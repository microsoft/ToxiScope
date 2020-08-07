#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import codecs

class BaseDataset:
    def __init__(self, vocabulary={}, min_freq=1):
        self.vocabulary = vocabulary
        self.min_freq = min_freq
        self.id2word={}
        self.word_freq={}

    def fit(self, train_file):
        self.vocabulary = {}
        with codecs.open(train_file) as f:
            for line in f:
                for token in self._tokenize(line):
                    token = token.lower()
                    self.word_freq[token] = self.word_freq.get(token, 0) + 1

        self.vocabulary["<PADDING>"] = 0
        self.vocabulary["<UNK>"] = 1

        for word, count in sorted(self.word_freq.items(), key=lambda x:x[1], reverse=True):
            if count < self.min_freq:
                break
            self.vocabulary[word] = len(self.vocabulary)
        
        self.id2word = {v:k for k,v in self.vocabulary.items()}

    def transform(self, train_file, data_loader, batch_size, shuffle):
        data = []
        with codecs.open(train_file) as tf:
            for line in tf:
                obj = self._generate_obj(line)
                data.append(obj)
        return data_loader(data, batch_size, shuffle)
    
    def fit_transform(self, train_file, data_loader, batch_size, shuffle):
        self.fit(train_file)
        return self.transform(train_file, data_loader, batch_size, shuffle)

    def _tokenize(self, line):
        raise NotImplementedError

    def _generate_obj(self, line):
        raise NotImplementedError

if __name__ == "__main__":
    pass

