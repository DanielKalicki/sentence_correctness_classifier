import os.path
import json
import bz2
import pickle
import random
import re
import copy
import math
from urllib.parse import unquote
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
print(tokenizer.vocab_size)

random.seed(0)
batch_train_data = []
batch_train_data_size = 0
batch_valid_data = []
batch_valid_data_size = 0
batch_full_data = {}

train_title_list = []
valid_title_list  = []
train_title_weight = []
valid_title_weight = []

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def find_html_links_from_wikipage(text):
    links = []
    # search_results = re.findall('\shref.*?>', text)
    search_results = re.findall("href=\"view-source:https://en.wikipedia.org/wiki/.*?>", text)
    search_results = re.findall("<a href=\"https://en.wikipedia.org/wiki/.*?\s", text)
    for link in search_results:
        links.append(unquote(
            link.replace("<a href=\"https://en.wikipedia.org/wiki/", "")[:-2]))
    return links

class WikiSentCorrectnessBatch(Dataset):
    def __init__(self, config, valid=False):
        self.config = config
        self.max_sent_len = float(self.config['max_sent_len'])

        self.valid = valid

        self.datasets_dir = "/home/kalickid/Projects/github/s2v_linker/datasets/hotpotqa/"

        self.batch_files = []
        self.batch_idx = 0
        self._init_batch()

    def _init_batch(self):
        if not self.valid:
            global batch_train_data, batch_valid_data, batch_full_data
            global batch_train_data_size, batch_valid_data_size
            batch_train_data = []
            batch_valid_data = []
            batch_full_data = {}

            files = []
            for folder in sorted(os.listdir(self.datasets_dir)):
                for file in sorted(os.listdir(self.datasets_dir+folder)):
                    if file.split(".")[-1] == 'bz2':
                        files.append(self.datasets_dir+folder+"/"+file)

            random.shuffle(files)
            for file in files[0:200]:
                with bz2.BZ2File(file, "r") as fp:
                    # print(file)
                    valid_cnt = 0 # first document in batch file is mark always as test
                    for line in fp:
                        data = json.loads(line)
                        title = data['title'].lower()
                        text = data['text'][1:-1] # skip first and last paragraph
                        if valid_cnt < 1:
                            batch_valid_data.append(title)
                            valid_cnt += 1
                        else:
                            batch_train_data.append(title)
                        batch_full_data[title] = []
                        for p_idx, paragraph in enumerate(text):
                            for l_idx, sentence in enumerate(paragraph):
                                sentence = " "+remove_html_tags(sentence).strip()
                                batch_full_data[title].append({'text': sentence, 'p_idx': p_idx})

            batch_train_data_size = 0
            for title in batch_train_data:
                batch_train_data_size += len(batch_full_data[title])
            batch_valid_data_size = 0
            for title in batch_valid_data:
                batch_valid_data_size += len(batch_full_data[title])
            print("\tTrain dataset size: " + str(batch_train_data_size))
            print("\tTest dataset size: " + str(batch_valid_data_size))

            global train_title_list, valid_title_list
            global train_title_weight, valid_title_weight
            train_title_list = []
            train_title_weight = []
            for title in batch_train_data:
                train_title_list.append(title)
                train_title_weight.append(len(batch_full_data[title]))
            valid_title_list = []
            valid_title_weight = []
            for title in batch_valid_data:
                valid_title_list.append(title)
                valid_title_weight.append(len(batch_full_data[title]))
            
    def on_epoch_end(self):
        self._init_batch()

    def __len__(self):
        if self.valid:
            return int(batch_valid_data_size//16)
        else:
            return int(batch_train_data_size//(64))

    def get_title_from_idx(self):
        global train_title_list, valid_title_list
        global train_title_weight, valid_title_weight
        title_list = valid_title_list if self.valid else train_title_list
        title_weight = valid_title_weight if self.valid else train_title_weight
        rnd_title = random.choices(title_list, weights=title_weight)[0]
        return rnd_title

    def __getitem__(self, idx):
        global batch_train_data, batch_valid_data
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sentences = torch.zeros((int(self.max_sent_len),), dtype=torch.long)
        sentences_mask = torch.ones((int(self.max_sent_len),), dtype=torch.bool)
        label = torch.zeros((1,), dtype=torch.long)

        info = {}

        title = self.get_title_from_idx()
        batch_data = batch_full_data[title]

        # input sentences
        rnd_input_sent_idx = random.randint(0, len(batch_data)-1)
        text = batch_data[rnd_input_sent_idx]['text']
        tokens_w = tokenizer.tokenize(text)
        tokens = tokenizer.convert_tokens_to_ids(tokens_w)
        if random.random() > 0.5:
            sentence_changed = False
            # for i_word, _ in enumerate(tokens):
            #     if random.random() > 0.9:
            #         tokens[i_word] = random.randint(0, tokenizer.vocab_size-1)
            #         sentence_changed = True
            if not sentence_changed:
                idx = random.randint(0, len(tokens)-1)
                tokens[idx] = random.randint(0, tokenizer.vocab_size-1)
            label = 0
            info['class'] = 'incorrect'
        else:
            label = 1
            info['class'] = 'correct'
        
        sentences[0:min(len(tokens), int(self.max_sent_len))] =\
            torch.LongTensor(tokens[0:min(len(tokens), int(self.max_sent_len))])
        sentences_mask[0:min(len(tokens), int(self.max_sent_len))] = torch.tensor(0.0)
        info['input_sentence'] = ' '.join(tokens_w[:int(self.max_sent_len)]).replace(' [PAD]', '')

        return sentences, sentences_mask, label, info

    def tokenizer(self):
        global tokenizer
        return tokenizer

def test():
    batcher = WikiSentCorrectnessBatch({
        's2v_dim': 4096,
        'doc_len': 400,
        'max_sent_len': 16
    })
    for i in range(100):
        x = batcher.__getitem__(i)
        # print(x)

test()
