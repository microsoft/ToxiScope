#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
from trainer import Trainer
from models import DNNClassifier
import argparse
from data_loader import EmailIntentDataset, EmailIntentDataLoader, EmailIntentContextDataset, EmailIntentContextDataLoader
import torch.nn as nn
import json
import torch
import numpy as np
import random
from evaluator import Evaluator
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, help="The training file")
    parser.add_argument("--valid_file", type=str, help="The validation file")
    parser.add_argument("--test_file", type=str, help="The test file")
    parser.add_argument("--config_file", type=str, help="The model configuration file")
    parser.add_argument("--resume", type=str, help="the saved checkpoint path")
    parser.add_argument("--saved_folder", type=str, help="the folder for model checkpoint")
    parser.add_argument("--gpuid", type=int, default=0, help="the gpu id")
    parser.add_argument("--use_context", action="store_true", help="whether use the context information")
    return parser.parse_args()


def main():
    args = parse_args()
    
    with open(args.config_file) as f:
        config = json.load(f)
    
    # set the random seed
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    if config["with_cuda"]:
        torch.cuda.manual_seed_all(config["seed"])
        torch.cuda.set_device(args.gpuid)
    
    if args.use_context:
        dataset = EmailIntentContextDataset
        dataloader = EmailIntentContextDataLoader
    else:
        dataset = EmailIntentDataset
        dataloader =EmailIntentDataLoader
    
    text_set = dataset(min_freq=config["min_word_freq"])
    
    train_set = text_set.fit_transform(args.train_file, dataloader, batch_size=config["batch_size"], shuffle=True)
    valid_set = text_set.transform(args.valid_file,  dataloader, batch_size=config["batch_size"], shuffle=False)
    
    config["vocab_size"] = len(text_set.vocabulary)
    config["saved_folder"] = args.saved_folder

    if not os.path.exists(os.path.join(args.saved_folder, config["name"])):
        os.makedirs(os.path.join(args.saved_folder, config["name"]))

    # dump the vocabulary
    with open(os.path.join(args.saved_folder, config["name"],"vocabulary.pkl"), "wb") as otf:
        pickle.dump(text_set.vocabulary, otf, protocol=pickle.HIGHEST_PROTOCOL)


    if "word2vec_file" in config:
        num_matched_vocab = 0
        # laod the pretrained word2vec
        word_embedding = np.random.uniform(-1, 1, size=[config["vocab_size"], config["word_emb_size"]])
        with open(config["word2vec_file"]) as tf:
            for line in tf:
                word, vec = line.strip().split(" ",1)
                vec = np.fromstring(vec, sep=" ")
                if word in text_set.vocabulary:
                    word_embedding[text_set.vocabulary[word]] = vec
                    num_matched_vocab += 1

        config["pretrained_word_embedding"] = word_embedding
        print("Total %d words in Vocabulary, matched %d in word2vec" % (config["vocab_size"], num_matched_vocab))


    classifier = DNNClassifier(config)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    
    if config["with_cuda"]:
        classifier.cuda()

    model_trainer = Trainer(classifier, loss_fn, config["metrics"], args.resume, config, train_set, valid_set)
    model_trainer.train()

    # evaluate on validation and test set
    # laod the best classifier configuation
    best_model_path = os.path.join(config["saved_folder"], config["name"],"checkpoint_best.pth.tar")
    checkpoint = torch.load(best_model_path)
    classifier.load_state_dict(checkpoint["state_dict"])

    estimator = Evaluator(classifier, labels=range(config["num_classes"]), target_names=config["target_names"])
    valid_report,_ ,_,_ = estimator.evaluate(valid_set)

    print(valid_report)
    if args.test_file:
        test_set = text_set.transform(args.test_file, dataloader, batch_size=config["batch_size"], shuffle=False)
        test_report,_,_,_ = estimator.evaluate(test_set)
        print('Test Performance Report')
        print(test_report)

if __name__ == "__main__":
    main()
