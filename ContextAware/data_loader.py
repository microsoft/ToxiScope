#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import codecs
from sklearn.model_selection import train_test_split
import json
from base.base_data_loader import BaseDataLoader
from base.base_dataset import BaseDataset
import nltk

class EmailIntentDataset(BaseDataset):
    def __init__(self, vocabulary={}, min_freq=1):
        super(EmailIntentDataset, self).__init__(vocabulary, min_freq)

    def _tokenize(self, line):
        obj = json.loads(line)
        return [t for t in obj["sentence"]]

    def _generate_obj(self, line):
        record = json.loads(line)
        tokens = self._tokenize(line)
        ids = [self.vocabulary.get(t.lower(), 1) for t in tokens]
        return {"label": int(record["label"]), "sentence": ids, "tokens": tokens}

class EmailIntentDataLoader(BaseDataLoader):
    def __init__(self, data, batch_size, shuffle=False):
        super(EmailIntentDataLoader, self).__init__(data, batch_size, shuffle)

    def _unpack_data(self, batch):
        sentences, labels = zip(*[(b["sentence"], b["label"]) for b in batch])
        return sentences, labels


class EmailIntentContextDataset(BaseDataset):
    def __init__(self, vocabulary={}, min_freq=1):
        super(EmailIntentContextDataset, self).__init__(vocabulary, min_freq)

    def _tokenize(self, line):
        obj = json.loads(line)
        return [w for sen in obj["context"] for w in sen]

    def _generate_obj(self, line):
        record = json.loads(line)
        sen_ids = [self.vocabulary.get(t.lower(), 1) for t in record["sentence"]]
        context_ids = [[self.vocabulary.get(t.lower(),1) for t in sen] for sen in record["context"] if len(sen)>0]
        if len(context_ids) == 0:
            context_ids = [sen_ids]
        return {"label": int(record["label"]), "sentence": sen_ids, "context": context_ids}

class EmailIntentContextDataLoader(BaseDataLoader):
    def __init__(self, data, batch_size, shuffle=False):
        super(EmailIntentContextDataLoader, self).__init__(data, batch_size, shuffle)

    def _unpack_data(self, batch):
        sentences, contexts, labels = zip(*[(b["sentence"], b["context"], b["label"])  for b in batch])
        return (sentences, contexts), labels

class SentenceDataLoader(BaseDataLoader):
    def __init__(self, data, batch_size, shuffle=False):
        super(SentenceDataLoader, self).__init__(data, batch_size, shuffle)

    def _unpack_data(self, batch):
        sentences, sentence_ids, labels = zip(*[(b["sentence"], b["sentence_ids"], b["label"]) for b in batch])
        return [sentences, sentence_ids], labels

class MeetingSubintentLoader(BaseDataLoader):
    def __init__(self, data, batch_size, shuffle=False):
        super(MeetingSubintentLoader, self).__init__(data, batch_size, shuffle)

    def _unpack_data(self, batch):
        contexts, labels = zip(*[(b["context"], b["label"]) for b in batch])
        return contexts, labels

class SentenceDataset(BaseDataset):
    def __init__(self, vocabulary={}, min_freq=1):
        super(SentenceDataset, self).__init__(vocabulary, min_freq)

    def _tokenize(self, line):
        record = json.loads(line)
        return [t for t in record["sentence"]]

    def _generate_obj(self, line):
        record = json.loads(line)
        tokens = [t.lower() for t in self._tokenize(line)]
        ids = [self.vocabulary.get(t.lower(), 1) for t in tokens]
        label = int(record["label"])
        return {"label": label, "sentence_ids": ids, "sentence": tokens}


class MeetingSubintentDataset:
    def __init__(self, vocabulary={}, min_freq=1):
        self.vocabulary = vocabulary
        self.min_freq = min_freq
        self.id2word = {}
        self.word_freq = {}
    
    def fit(self, train_file, text_field="context"):
        self.vocabulary = {}
        with codecs.open(train_file) as f:
            for line in f:
                obj = json.loads(line)
                for sen in obj[text_field]:
                    for token in sen:
                        token = token.lower()
                        self.word_freq[token] = self.word_freq.get(token, 0) + 1

        # add <PADDING> and UNK into vocabulary
        self.vocabulary["<PADDING>"] = 0
        self.vocabulary["<UNK>"] = 1

        for word, count in sorted(self.word_freq.items(), key=lambda x:x[1], reverse=True):
            if count < self.min_freq:
                break
            self.vocabulary[word] = len(self.vocabulary)

        self.id2word = {v:k for k,v in self.vocabulary.items()}

    def transform(self, train_file, text_field="context", batch_size=64, shuffle=False):
        if len(self.vocabulary) == 0:
            raise Exception("Please fit the dataset before transform")
        
        data = []
        with codecs.open(train_file) as tf:
            for line in tf:
                obj = json.loads(line)
                context_ids = [[self.vocabulary.get(w, 1) for w in sen] for sen in obj[text_field]]
                if len(context_ids) == 0:
                    continue
                data.append({"context": context_ids, "label": int(obj["label"])})
       
        data = MeetingSubintentLoader(data, batch_size, shuffle)
        return data

    def transform_split(self, train_file, text_field="context", ratio=0.8, batch_size=64, shuffle=False):
        data = self.transform(train_file, text_field)
        labels = [o["label"] for o in data]
        train_split, valid_split = train_test_split(data, test_size=1-ratio, stratify=labels, shuffle=True)
        train_split = MeetingSubintentLoader(train_split, batch_size, shuffle)
        valid_split = MeetingSubintentLoader(valid_split, batch_size, False)
        return train_split, valid_split

    def fit_transform(self, train_file, text_field="context"):
        self.fit(train_file, text_field)
        return self.transform(train_file, text_field)


class EmailConvLoader(BaseDataLoader):
    def __init__(self, data, batch_size, shuffle=False):
        super(EmailConvLoader, self).__init__(data, batch_size, shuffle)

    def _unpack_data(self, batch):
        sent, reply = zip(*[(b["sent"], b["reply"]) for b in batch])
        return sent, reply


class FlatEmailConvDataset:
    def __init__(self, vocabulary={}, min_freq=1):
        self.vocabulary = vocabulary
        self.min_freq = min_freq
        self.id2word = {}
        self.word_freq = {}

    def fit(self, sent_file, reply_file):
        for f in [sent_file, reply_file]:
            with codecs.open(f) as itf:
                for line in itf:
                    for token in line.strip().split(" "):
                        token = token.lower()
                        self.word_freq[token] = self.word_freq.get(token, 0) + 1
        self.vocabulary["<PADDING>"] = 0
        self.vocabulary["<UNK>"] = 1

        for word, count in sorted(self.word_freq.items(), key=lambda x:x[1], reverse=True):
            if count < self.min_freq:
                break
            self.vocabulary[word] = len(self.vocabulary)

        self.id2word = {v:k for k,v in self.vocabulary.items()}

    def transform(self, sent_file, reply_file, batch_size, shuffle):
        data = []
        with codecs.open(sent_file) as sf, open(reply_file) as rf:
            for sent, reply in zip(sf, rf):
                sent_ids = [self.vocabulary.get(t.lower()) for t in sent.strip().split(" ")]
                reply_ids = [self.vocabulary.get(t.lower()) for t in reply.strip().split(" ")]
                data.append({"sent": sent_ids, "reply": reply_ids})
        return EmailConvLoader(data, batch_size, shuffle)

    def fit_transform(self, sent_file, reply_file, batch_size, shuffle):
        self.fit(sent_file, reply_file)
        return self.transform(sent_file, reply_file, batch_size, shuffle)



class EmailConvDataset:
    def __init__(self, vocabulary={}, min_freq=1):
        self.vocabulary = vocabulary
        self.min_freq = min_freq
        self.id2word = {}
        self.word_freq = {}

    def fit(self, sent_file, reply_file):
        with codecs.open(sent_file) as sf, codecs.open(reply_file) as rf:
            for sent, reply in zip(sf, rf):
                for body in [sent, reply]:
                    for sen in nltk.sent_tokenize(sent):
                        for token in nltk.word_tokenize(sen):
                            token = token.lower()
                            self.word_freq[token] = self.word_freq.get(token, 0) + 1
        # add UNK and PADDING
        self.vocabulary["<PADDING>"] = 0
        self.vocabulary["<UNK>"] = 1

        for word, count in sorted(self.word_freq.items(), key=lambda x:x[1], reverse=True):
            if count < self.min_freq:
                break
            self.vocabulary[word] = len(self.vocabulary)

        self.id2word = {v:k for k,v in self.vocabulary.items()}

    
    def transform(self, sent_file, reply_file, batch_size, shuffle):
        data = []
        with codecs.open(sent_file) as sf, codecs.open(reply_file) as rf:
            for sent, reply in zip(sf, rf):
                sent_ids = self._doc2ids(sent)
                reply_ids = self._doc2ids(reply)
                data.append({"sent": sent_ids, "reply": reply_ids})
        
        return EmailConvLoader(data, batch_size, shuffle)

    def fit_transform(self, sent_file, reply_file, batch_size, shuffle):
        self.fit(sent_file, reply_file)
        return self.transform(sent_file, reply_file, batch_size, shuffle)
    
    def _doc2ids(self, doc):
        doc = doc.strip()
        ids = []
        for sen in nltk.sent_tokenize(doc):
            ids.append([self.vocabulary.get(token.lower(), 1) for token in nltk.word_tokenize(sen)])
        return ids




if __name__ == "__main__":
    pass

