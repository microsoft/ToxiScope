#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import torch.nn as nn
import torch
from sklearn.metrics import classification_report

class Evaluator:
    def __init__(self, classifier, labels=None, target_names=None):
        self.classifier = classifier
        self.labels = labels
        self.target_names = target_names

    def predict(self, batch_x):
        self.classifier.eval()
        output = self.classifier(batch_x)
        return output["probs"]
    
    def evaluate(self, test_set):
        self.classifier.eval()
        
        full_preds = []
        full_targets = []
        full_probs = []
        for bid, (batch_x, target) in enumerate(test_set):
            output = self.classifier(batch_x)
            probs = list(output["probs"].cpu().data.numpy()[:,1])
            preds = list(output["predictions"].cpu().data.numpy())
            full_probs += probs
            full_preds += preds
            full_targets += target

        report = classification_report(full_targets, full_preds, labels=self.labels, target_names=self.target_names, digits=4)    
        return report, full_preds, full_targets, full_probs

if __name__ == "__main__":
    pass

