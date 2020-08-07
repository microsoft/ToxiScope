#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = "Wei Wang"
__email__ = "tskatom@gmail.com"

import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from collections import namedtuple
import json
import numpy as np
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support, auc

Data = namedtuple('Data', 'data target')

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="commitment")
    ap.add_argument("--data_folder", type=str)
    ap.add_argument("--result_folder", type=str)
    ap.add_argument("--exp_type", type=str, default="sentence_only")
    return ap.parse_args()

def benchmark_LR(train, valid, test, metric='f1'):
    tfidf = TfidfVectorizer(sublinear_tf=True, max_df=0.5, ngram_range=(1,3),stop_words='english', max_features=50000)

    Cs = [0.1, 1, 10, 100, 1000]
    best_model = None
    best_f1 = 0.
    best_c = -1
    best_penalty = "l2"

    for c in Cs:
        for penalty in ["l1", "l2"]:
            pipeline = Pipeline(
                    [
                        ("feature_extraction", tfidf), 
                        ("classifier", LogisticRegression(C=c, penalty=penalty))
                    ]
                    )
            pipeline.fit(train.data, train.target)
            valid_pred = pipeline.predict(valid.data)

            if metric == "f1":
                valid_f1 = f1_score(valid.target, valid_pred, pos_label=1)
            elif metric == "accuracy":
                valid_f1 = accuracy_score(valid.target, valid_pred, pos_label=1)

            if valid_f1 > best_f1:
                best_f1 = valid_f1
                best_model = pipeline
                best_c = c
                best_penalty = penalty

    test_pred = best_model.predict(test.data)
    test_prob = best_model.predict_proba(test.data)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(test.target, test_prob)
    test_auc = auc(recalls, precisions)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(test.target, test_pred, average='binary')
    
    print("Precision: %0.4f, Recall: %0.4f, F1-Score: %0.4f, AUPRC: %0.4f Best_c: %f, Best_penalty: %s" % ( test_prec, test_rec, test_f1, test_auc, best_c, best_penalty))
    return test_prec, test_rec, test_f1, test_auc, best_c, best_penalty, test_prob, test.target

def load_data(train_f, valid_f, test_f):
    dataset = []
    for f in [train_f, valid_f, test_f]:
        with open(f) as df:
            texts = []
            targets = []
            for line in df:
                info = json.loads(line)
                texts.append(' '.join(info['sentence']))
                targets.append(info['label'])

            data = Data(data=texts, target=targets)
            dataset.append(data)
    return dataset

def main():
    args = parse_args()
    result_folder = args.result_folder
    data_folder = args.data_folder

    out_name = "prediction.txt"
    result_name = "result.txt" 
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    out_name = os.path.join(result_folder, out_name)
    result_name = os.path.join(result_folder, result_name)
    
    # load the dataset
    if args.exp_type == "sentence_only":
        train_f = os.path.join(data_folder, "train.json")
        valid_f = os.path.join(data_folder, "valid.json")
        test_f = os.path.join(data_folder, "test.json")

        train, valid, test = load_data(train_f, valid_f, test_f)
    
    test_prec, test_rec, test_f1, test_auc, best_c, best_penalty, test_prob, test_label = benchmark_LR(train, valid, test)

    with open(out_name, 'w') as otf, open(result_name, 'w') as rtf:
        for p,l in zip(test_prob, test_label):
            otf.write("%0.4f\t%d\n" % (p, l))

        r_str = "precision: %0.4f recall: %0.4f f1: %0.4f auprc: %0.4f c: %f penality: %s" % (test_prec, test_rec, test_f1, test_auc, best_c, best_penalty)
        rtf.write("%s\n" % r_str)
    

if __name__ == "__main__":
    main()
