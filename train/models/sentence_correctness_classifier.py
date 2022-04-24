import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import Parameter
from torch.nn.utils import weight_norm
import numpy as np
import random
import bitsandbytes as bnb
from transformers import AutoModel, AutoModelForMaskedLM
from torch.distributions import Categorical

class SentCorrClassActor(nn.Module):
    def __init__(self, config):
        super(SentCorrClassActor, self).__init__()
        self.config = config
        self.batch_cnt, self.loss, self.total, self.correct = 1e-6, 0, 1e-6, 0

        # self.actor_roberta = AutoModel.from_pretrained("roberta-base")
        self.actor_roberta = AutoModel.from_pretrained("distilroberta-base")
        self.actor_roberta.train()
        self.actor_fc1 = nn.Linear(self.config['word_edim'], 512)
        self.actor_fc2 = nn.Linear(512, 2)

        self.optimizer = bnb.optim.Adam8bit(self.parameters(), config['training']['lr'], betas=(0.9, 0.995))
        lr_lambda = lambda epoch: config['training']['lr_gamma']**epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def forward(self, sents):
        batch_size = sents[0].shape[0]
        max_sent_len = sents[0].shape[1]
        sent = sents[0]
        sent_mask = sents[1]
        sent = self.actor_roberta(input_ids=sent, attention_mask=torch.logical_not(sent_mask))[0]
        # sent = F.softmax(sent, dim=-1)
        pred = self.actor_fc1(sent[:, 0])
        pred = F.relu(pred)
        pred = self.actor_fc2(pred)
        return pred

    def reset_stats(self):
        self.batch_cnt, self.loss, self.total, self.correct = 1e-6, 0, 1e-6, 0

    def add_batch_stats(self, loss, correct, total):
        self.batch_cnt += 1
        self.loss += loss
        self.correct += correct
        self.total += total

    def get_loss(self):
        return self.loss / self.batch_cnt

    def get_accuracy(self):
        return self.correct / self.total

    def scheduler_step(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

class SentCorrClassCritic(nn.Module):
    def __init__(self, config):
        super(SentCorrClassCritic, self).__init__()
        self.config = config
        self.batch_cnt, self.loss, self.total, self.correct = 1e-6, 0, 1e-6, 0

        # self.critic_roberta = AutoModel.from_pretrained("roberta-base")
        self.critic_roberta = AutoModel.from_pretrained("distilroberta-base")
        self.critic_roberta.train()

        self.critic_fc1 = nn.Linear(self.config['word_edim']+2, 512)
        self.critic_fc2 = nn.Linear(512, 1)

        self.optimizer = bnb.optim.Adam8bit(self.parameters(), config['training']['lr'], betas=(0.9, 0.995))
        lr_lambda = lambda epoch: config['training']['lr_gamma']**epoch
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def forward(self, sents, action):
        batch_size = sents[0].shape[0]
        max_sent_len = sents[0].shape[1]
        sent = sents[0]
        sent_mask = sents[1]
        sent = self.critic_roberta(input_ids=sent, attention_mask=torch.logical_not(sent_mask))[0]
        value = torch.cat([sent[:, 0], action], dim=1)
        value = self.critic_fc1(value)
        value = F.relu(value)
        value = self.critic_fc2(value)
        return value
    
    def reset_stats(self):
        self.batch_cnt, self.loss, self.total, self.correct = 1e-6, 0, 1e-6, 0
    
    def add_batch_stats(self, loss, correct, total):
        self.batch_cnt += 1
        self.loss += loss
        self.correct += correct
        self.total += total
    
    def get_loss(self):
        return self.loss / self.batch_cnt

    def get_accuracy(self):
        return self.correct / self.total

    def scheduler_step(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()
