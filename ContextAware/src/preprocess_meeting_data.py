#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import dataset
import argparse
import numpy as np
import pickle
import time

start = time.time()
np.random.seed(1234)

ap = argparse.ArgumentParser()
ap.add_argument("--train_file", type=str)
ap.add_argument("--test_file", type=str)
ap.add_argument("--valid_file", type=str)
ap.add_argument("--out_folder", type=str)
ap.add_argument("--pretrained_wordvec", type=str)
ap.add_argument("--min_freq", type=int, default=3, help='the minimum freq for the word in vocabulary')
ap.add_argument("--word_embsize", type=int, default=300)
ap.add_argument("--txt_processed", action='store_true', help='whether process the text data')
ap.add_argument("--sen_field", type=str, help="the text field for sentence")
ap.add_argument("--context_field", type=str, help="the text field for context")
ap.add_argument("--label_field", default="label", type=str, help="the text field for the label")

args = ap.parse_args()

if not os.path.exists(args.out_folder):
    os.makedirs(args.out_folder)

# save train/valid/test file as train/valid/test.json in output folder
if args.train_file and not args.txt_processed:
    dataset.preprocess_text_data(args.train_file, os.path.join(args.out_folder, "train.json"), args.sen_field, args.context_field, args.label_field)

if args.valid_file and not args.txt_processed:
    dataset.preprocess_text_data(args.valid_file, os.path.join(args.out_folder, "valid.json"), args.sen_field, args.context_field, args.label_field)

if args.test_file and not args.txt_processed:
    dataset.preprocess_text_data(args.test_file, os.path.join(args.out_folder, "test.json"), args.sen_field, args.context_field, args.label_field)

print("Finish preprocess train/valid/test files using %d" % (time.time() - start))
start = time.time()
# generate dataset
meeting_corpus = dataset.MeetingCorpus(min_freq=args.min_freq)
meeting_corpus.fit(os.path.join(args.out_folder, "train.json"))

# load the pretrained word2vec
vocab_size = len(meeting_corpus.vocabulary)
embedding = np.random.uniform(-1, 1, size=(vocab_size, args.word_embsize))

num_matched_vocab = 0
with open(args.pretrained_wordvec) as pw:
    for line in pw:
        word, vec = line.strip().split(" ", 1)
        vec = np.fromstring(vec, sep=" ")
        if word in meeting_corpus.vocabulary:
            embedding[meeting_corpus.vocabulary[word]] = vec
            num_matched_vocab += 1
    print("Total %d words in Vocabulary, matched %d in word2ve" % (vocab_size, num_matched_vocab))

# saved the vocabulary and embedding
glove_file = os.path.join(args.out_folder, "glove.pkl")
with open(glove_file, 'wb') as gf:
    pickle.dump(embedding, gf)
    pickle.dump(meeting_corpus.vocabulary, gf)

print("Finish load word2vec using %d" % (time.time() - start))

if __name__ == "__main__":
    pass

