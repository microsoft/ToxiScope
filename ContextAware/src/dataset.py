#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import torch
import json
import numpy as np
import re
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import nltk
from fuzzywuzzy import fuzz

def emailbody_to_tokens(email_body, min_sen_len=2):
    """"
        input: string
        output: list of list of tokens with lower format
    """
    email_body = re.sub(r"(\r\n)+", r"\n", email_body)
    email_body = re.sub(r" +", r" ", email_body)

    email_tokens = []
    for paragraph in email_body.strip().split("\n"):
        for sen in nltk.sent_tokenize(paragraph):
            tokens = [token.lower() for token in nltk.word_tokenize(sen)]
            if len(tokens) > min_sen_len:
                email_tokens.append(tokens)
    return email_tokens


def preprocess_meeting_data(in_file, out_file):
    """
        transform the email body into list of sentence which is a list of tokens
    """
    with open(in_file) as in_f, open(out_file, 'w') as out_f:
        for line in in_f:
            record = json.loads(line)
            sen_tokens = record["origin_tokens"]
            email_body = record["email_body"]
            email_tokens = emailbody_to_tokens(email_body)
            label = record["label"]

            new_record = {"sentence": sen_tokens, "context": email_tokens, "label": label}
            
            out_f.write(json.dumps(new_record, ensure_ascii=False) + "\n")

def preprocess_text_data(in_file, out_file, text_field="sentence", context_field=None, label_field="label"):
    with open(in_file) as in_f, open(out_file, "w") as out_f:
        for line in in_f:
            record = json.loads(line)
            sen_text = record[text_field]
            if isinstance(sen_text, str):
                sen_tokens = [token.lower() for token in nltk.word_tokenize(sen_text)]
            elif isinstance(sen_text, list):
                sen_tokens = sen_text

            if context_field is None:
                context_tokens = [sen_tokens]
            else:
                context_text = record[context_field]
                context_tokens = emailbody_to_tokens(context_text)

            label = record[label_field]
            new_record = {"sentence": sen_tokens, "context": context_tokens ,"label": label}

            out_f.write(json.dumps(new_record, ensure_ascii=False) + "\n")


class MeetingCorpus(object):
    def __init__(self, vocabulary={}, min_freq=1):
        self.vocabulary = vocabulary
        self.id2word = {}
        self.word_freq = {}
        self.min_freq = min_freq

    def fit(self, train_file):
        self.vocabulary = {}
        with open(train_file) as tf:
            for line in tf:
                record = json.loads(line)
                for sen in record["context"]:
                    for token in sen:
                        self.word_freq[token] = self.word_freq.get(token, 0) + 1
        # add new token <UNK> and <PADDING> into vocabulary
        self.vocabulary["<PADDING>"] = 0
        self.vocabulary["<UNK>"] = 1
        
        for word, count in self.word_freq.items():
            if count >= self.min_freq:
                self.vocabulary[word] = len(self.vocabulary)
        
        self.id2word = {v:k for k,v in self.vocabulary.items()}
        
    def fit_transform(self, train_file):
        self.fit(train_file)
        return self.transform(train_file)

    def fit_transform_context(self, train_file, win_size=5):
        self.fit(train_file)
        return self.transform_context(train_file, win_size)

    def transform(self, data_file):
        data = []
        tf = pd.read_csv(data_file)

        for idx, record in tf.iterrows():
            #record = json.loads(line)

            if isinstance(record["comments"], float):
                continue
            if isinstance(record["context"], float):
                record["context"] = ""
            text = record["comments"] + " " + record["context"]
            sentence_ids = [self.vocabulary.get(token, 1) for token in text]
            #if "gn" in data_file:
            #    print(len(record["context"]))

            if isinstance(record["context"], float):
                context_ids = [sentence_ids]
            else:
                context_ids = [[self.vocabulary.get(token, 1) for token in sen] for sen in record["context"]]

            if len(context_ids) == 0:
                context_ids = [sentence_ids]

            if isinstance(record["label"], str):
                record['label'] = (1 if record['label'] == 'y' else 0)
            data.append({"sentence": sentence_ids, "context": context_ids, "label": record["label"]})

        return data

    def transform_context(self, data_file, win_size=5):
        data = []
        with open(data_file) as tf:
            for line in tf:
                record = json.loads(line)
                sentence_ids = [self.vocabulary.get(token, 1) for token in record["sentence"]]
                context_ids = [[self.vocabulary.get(token, 1) for token in sen] for sen in record["context"]]

                if len(context_ids) == 0:
                    context_ids = [sentence_ids]
                else:
                    sentence_text = ' '.join(record['sentence'])
                    context_text = [' '.join(sen) for sen in record['context']]
                    best_match_idx = -1
                    best_score = 30
                    for c_id, context_sen in enumerate(context_text):
                        score = fuzz.ratio(sentence_text, context_sen)
                        if score > best_score:
                            best_score = score
                            best_match_idx = c_id
                    if best_match_idx == -1:
                        context_ids = context_ids[:2*win_size]
                    else:
                        start = max(0, best_match_idx - win_size)
                        end = best_match_idx + win_size + 1
                        context_ids = context_ids[start:end]
                data.append({'sentence': sentence_ids, 'context': context_ids, 'label': record['label']})
        return data 

class MeetingDataset(Dataset):
    def __init__(self, data):
        """
        data: list of json objs with {"sentence": [int], "context": [[int], [int]], "label": int}
        """
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def meeting_collate(batch):
    return {k:[b[k] for b in batch] for k in batch[0].keys()}

if __name__ == "__main__":
    pass

