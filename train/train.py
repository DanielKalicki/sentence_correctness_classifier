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

def train(models, device, loader, optimizers, epoch, scheduler):
    loss = run(models, device, loader, optimizers, epoch, mode="train", scheduler=scheduler)
    return loss

def test(models, device, loader, optimizers, epoch, scheduler):
    loss = run(models, device, loader, optimizers, epoch, mode="test", scheduler=scheduler)
    return loss

def calculate_mask_word_loss(batch_idx, info, pred_sentence, label_sent, label_mask, criterion):
    try:
        if batch_idx % 20 == 0:
            pred_tokens = (list(torch.max(pred_sentence, 2)[1][0].cpu().detach()))
            pred_tokens = [int(str(x).replace('tensor(', '').replace(')', '')) for x in pred_tokens]
            pred_out = ''
            input_sent_text = info['input_sentence'][0]
            label_sent_text = info['label_sentence'][0]
            pred_sent_text = ' '.join(tokenizer.convert_ids_to_tokens(pred_tokens))
            for idx in range(0, 8): #len(label_sent_text.split(" "))):
                pred_out += label_sent_text.split(" ")[idx]
                if not label_mask[0].tolist()[idx]:
                    pred_out += "[" + str(not label_mask[0].tolist()[idx])[0] + " " + pred_sent_text.split(" ")[idx] + "]" + " "
                else:
                    pred_out += " "
            print("sentence:")
            print(input_sent_text.replace('Ġ', ''))
            print("memory sentence:")
            print(' '.join(tokenizer.convert_ids_to_tokens(info['memory_words'][0])).replace('Ġ', ''))
            print("prediction:")
            print(pred_out.replace('Ġ', ''))
    except:
        pass

    shape_ = pred_sentence.shape
    pred_sentence = pred_sentence
    label_sent = label_sent
    label_mask = torch.logical_not(label_mask)

    pred_sentence = pred_sentence * torch.cat([label_mask.unsqueeze(-1)]*pred_sentence.shape[-1], dim=-1)
    label_sentence = label_sent * label_mask

    pred_sentence = pred_sentence.reshape(-1, num_embeddings)
    label_sentence = label_sentence.reshape(-1)
    label_mask = label_mask.reshape(-1)

    pred_loss = torch.sum(criterion(pred_sentence, label_sentence)*label_mask)/torch.sum(label_mask)
    batch_loss = (criterion(pred_sentence, label_sentence)*label_mask).reshape(shape_[0], shape_[1])
    return pred_loss, batch_loss, label_mask.reshape(shape_[0], shape_[1])

def run(models, device, loader, optimizers, epoch, mode="train", scheduler=None):
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
            for optimizer_ in optimizers:
                optimizer_.zero_grad()

        with torch.cuda.amp.autocast():
            pred_label = critic.forward(sents=(sent, sent_mask))

            # print(pred_label.shape)
            # print(label.shape)
            # print(pred_label[0])
            pred_loss = torch.mean(criterion(pred_label, label.to(torch.long)))
            # print(pred_loss.shape)

            # value_ = torch.sum(batch_loss, dim=-1)/torch.sum(label_mask, dim=-1)
            # # print((torch.sum(batch_loss, dim=-1)/torch.sum(label_mask, dim=-1)).shape)

            # critic_loss = torch.mean(torch.abs(value-value_.detach())) # -(torch.mean(memory*pred_loss.detach()))

            # log_prob = dist.log_prob(memory_words)
            # # print(log_prob.shape)
            # # print(value_[0])
            # # print(value[0])
            # # print(log_prob[0])
            # # print(torch.mean(value_-value, dim=-1))
            # # print(torch.mean(value_ - value, dim=-1).shape)
            # actor_loss = -(log_prob * torch.cat([torch.mean(value_-value, dim=-1).detach().unsqueeze(1)]*log_prob.shape[1], dim=1)).mean()

        if not math.isnan(float(pred_loss)):
            batch_cnt += 1
            if mode == "train":
                for optimizer in optimizers[1:]:
                    scaler.scale(pred_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clipnorm'])

        _, pred = torch.max(pred_label, 1)
        correct_ = (float)(torch.sum(pred == label))
        total_ = (float)(pred.shape[0])
        actor.add_batch_stats(loss=pred_loss.detach(), correct=correct_, total=total_)

        pbar.update(1)

    pbar.close()
    end = time.time()
    print("")
    print('Epoch {}:'.format(epoch))
    print('\t\t' + mode + ' time: {:.2f}'.format((end - start)))
    if config['training']['log']:
        wandb.log({mode: {'loss_actor': actor.get_loss(),
                          'loss_critic': critic.get_loss(),
                          'accuracy_actor': actor.get_accuracy(),
                          'accuracy_critic': critic.get_accuracy()}, 'epoch': epoch}, commit=True)

    return actor.get_loss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
actor = SentCorrClassActor(config)
critic = SentCorrClassCritic(config)
actor.to(device)
critic.to(device)
start_epoch = 1

models = [actor, critic]

dataset_train = WikiSentCorrectnessBatch(config)
data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_size=config['batch_size'],
    shuffle=True, num_workers=4)

dataset_test = WikiSentCorrectnessBatch(config, valid=True)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=3,
    shuffle=False, num_workers=0)

tokenizer = dataset_train.tokenizer()

optimizers = []
for model in models:
    optimizer = bnb.optim.Adam8bit(model.parameters(), config['training']['lr'], betas=(0.9, 0.995))
    optimizers.append(optimizer)

for epoch in range(start_epoch, config['training']['epochs'] + start_epoch):
    train(models, device, data_loader_train, optimizers, epoch, None)
    current_test_loss = test(models, device, data_loader_test, None, epoch, None)
