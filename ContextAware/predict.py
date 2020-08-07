#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import pickle
import torch.nn as nn
from data_loader import EmailIntentDataset, EmailIntentDataLoader, EmailIntentContextDataset, EmailIntentContextDataLoader
import argparse
from models import DNNClassifier
import json
import torch
from evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, help='test file')
    parser.add_argument("--vocab_file", type=str, help="the vocabulary file")
    parser.add_argument("--config_file", type=str, help='the model config file')
    parser.add_argument("--model_path", type=str, help='the saved model path')
    parser.add_argument("--use_context", action='store_true', help='whether use context iformation')
    parser.add_argument("--predictions", type=str, help="the prediction output file")
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config_file) as f:
        config = json.load(f)
    
    with open(args.vocab_file, 'rb') as f:
        vocabulary = pickle.load(f)

    if args.use_context:
        dataset = EmailIntentContextDataset
        dataloader = EmailIntentContextDataLoader
    else:
        dataset = EmailIntentDataset
        dataloader = EmailIntentDataLoader

    text_set = dataset(vocabulary=vocabulary, min_freq=config["min_word_freq"])

    test_set = text_set.transform(args.test_file, dataloader, batch_size=config["batch_size"], shuffle=False)
    config["vocab_size"] = len(vocabulary)

    classifier = DNNClassifier(config)
    checkpoint = torch.load(args.model_path, map_location={'cuda:1':'cuda:0'})
    classifier.load_state_dict(checkpoint["state_dict"])#
    if config["with_cuda"]:
        classifier.cuda()

    estimator = Evaluator(classifier, labels=range(config["num_classes"]), target_names=config["target_names"])
    report,preds,targets,probs = estimator.evaluate(test_set)
    print(report)
    
    with open(args.predictions, "w") as otf:
        for prob, target in zip(probs, targets):
            otf.write("%0.4f\t%d\n" % (prob, target))

if __name__ == "__main__":
    main()
