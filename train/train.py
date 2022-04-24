import os
os.environ['WANDB_CONSOLE'] = 'off'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import torch.optim as optim
import torch.nn.functional as F
from batchers.wiki_sent_correctness import WikiSentCorrectnessBatch
from models.sentence_correctness_classifier import SentCorrClassActor, SentCorrClassCritic
from datetime import datetime
import time
import sys
from configs import configs
from tqdm import tqdm
import numpy as np
import math
import random
import bitsandbytes as bnb
from torch.distributions import Categorical
import copy

num_embeddings = 50265

scaler = torch.cuda.amp.GradScaler()

print(int(sys.argv[1]))
config_idx = int(sys.argv[1])
config = configs[int(sys.argv[1])]

config['training']['log'] = True

class LabelSmoothingCrossEntropy(torch.nn.Module):
    # based on https://github.com/seominseok0429/label-smoothing-visualization-pytorch
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.2):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss

if config['training']['log']:
    import wandb
    wandb.init(project='sentence_correctness_classifier', entity='danielkalicki', config=config)

def train(models, device, loader, epoch, worker_idx):
    loss = run(models, device, loader, epoch, mode="train", worker_idx=worker_idx)
    return loss

def test(models, device, loader, epoch, worker_idx):
    loss = run(models, device, loader, epoch, mode="test", worker_idx=worker_idx)
    return loss

def run(models, device, loader, epoch, mode="train", worker_idx=0):
    start = time.time()
    pbar = tqdm(total=len(loader), dynamic_ncols=True)
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
    criterion = LabelSmoothingCrossEntropy()

    for model in models:
        if mode == "test":
            model.eval()
        model.reset_stats()

    actor = models[0]
    critic = models[1]

    batch_cnt = 1e-6
    for batch_idx, (sent, sent_mask, label, info) in enumerate(loader):
        sent, sent_mask, label = sent.to(device), sent_mask.to(device), label.to(device)

        if mode == "train":
            for model in models:
                model.zero_grad()

        with torch.cuda.amp.autocast():
            # select action
            pred_label = actor.forward(sents=(sent, sent_mask))
            action_probs = F.softmax(pred_label, dim=-1)
            dist = Categorical(action_probs)
            action = dist.sample()
            action_one_hot = F.one_hot(action, num_classes=2).type(torch.FloatTensor).to(device)

            # calculate reward
            reward = criterion(action_one_hot, label.to(torch.long)).detach()

            # critic
            value = critic.forward(sents=(sent, sent_mask), action=action_one_hot.detach())
            critic_loss = torch.abs(reward-value).mean()

            # actor loss
            advantage = value.detach()-reward.detach()
            actor_loss = -(dist.log_prob(action)*advantage).mean()

        if not math.isnan(float(actor_loss)) and not math.isnan(float(critic_loss)):
            batch_cnt += 1
            if mode == "train":
                scaler.scale(actor_loss).backward()
                scaler.step(actor.optimizer)
                scaler.update()

                scaler.scale(critic_loss).backward()
                scaler.step(critic.optimizer)
                scaler.update()

                # torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clipnorm'])

        _, pred = torch.max(pred_label, 1)
        correct_ = (float)(torch.sum(pred == label))
        total_ = (float)(pred.shape[0])
        actor.add_batch_stats(loss=actor_loss.detach(), correct=correct_, total=total_)
        critic.add_batch_stats(loss=critic_loss.detach(), correct=0, total=0)

        pbar.update(1)

    pbar.close()
    end = time.time()
    print("")
    print('Epoch {}:'.format(epoch))
    print('\t\t' + mode + ' time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        wandb.log({mode: {worker_idx: {'loss_actor': actor.get_loss(),
                                       'loss_critic': critic.get_loss(),
                                       'accuracy_actor': actor.get_accuracy(),
                                       'accuracy_critic': critic.get_accuracy()}}, 
                         'epoch': epoch}, commit=True)

    return actor.get_accuracy()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
workers = []
start_epoch = 1

# model
for _ in range(4):
    actor = SentCorrClassActor(config)
    critic = SentCorrClassCritic(config)
    actor.to(device)
    critic.to(device)
    workers.append([actor, critic])

dataset_train = WikiSentCorrectnessBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=4)

dataset_test = WikiSentCorrectnessBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1,
    shuffle=False, num_workers=0)

tokenizer = dataset_train.tokenizer()

for epoch in range(start_epoch, config['training']['epochs'] + start_epoch):
    best_worker = {'idx': -1, 'acc': 0}
    for idx, models in enumerate(workers):
        train(models, device, data_loader_train, epoch, idx)
        test_acc = test(models, device, data_loader_test, epoch, idx)
        if test_acc > best_worker['acc']:
            best_worker['acc'] = test_acc
            best_worker['idx'] = idx

    for idx, models in enumerate(workers):
        if idx != best_worker['idx']:
            models[0].load_state_dict(workers[best_worker['idx']][0].state_dict())
            models[1].load_state_dict(workers[best_worker['idx']][1].state_dict())

    dataset_train.on_epoch_end()

    for models in workers:
        for model in models:
            model.scheduler_step()
