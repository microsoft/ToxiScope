#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pickle
import dataset
import argparse
from ContextModels import HierarchyAttention, DynamicHierarchyAttention, SentenceAttention, ContextAttention
import time
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--data_folder", type=str, default="/data/MeetingData/processed/meeting_request", 
        help="the input data folder")
ap.add_argument("--result_folder", type=str, default="../result/meeting_request/hierarchy_attention",
        help="the folder to save the results and model")
ap.add_argument("--word_emb_size", type=int, default=300, help='the word embedding dim')
ap.add_argument("--batch_size", type=int, default=64, help="batch size")
ap.add_argument("--epochs", type=int, default=50, help='the max number of epochs to train')
ap.add_argument("--sentence_hidden_size", type=int, default=50, help='the dimention of sentence rnn hidden units')
ap.add_argument("--message_hidden_size", type=int, default=50, help='the dimention of message rnn hidden units')
ap.add_argument("--sentence_context_size", type=int, default=50, help='the context size in sentence rnn')
ap.add_argument("--message_context_size", type=int, default=50, help='the context size in message rnn')
ap.add_argument("--sentence_dropout_p", type=float, default=0.2, help='the dropout p for sentence embedding')
ap.add_argument("--message_dropout_p", type=float, default=0.2, help='the dropout p for message embedding')
ap.add_argument("--output_dropout_p", type=float, default=0.2, help='the dropout in final MLP layer')
ap.add_argument("--word_vec", type=str, default="glove.pkl", help='the preprocessed word2vec and vocab')
ap.add_argument("--num_classes", type=int, default=2, help='the number of classes')
ap.add_argument("--cuda", action='store_true', help="use GPU or not")
ap.add_argument("--fine_tune", action='store_true', help="fine tune or not")
ap.add_argument("--gpuid", type=int, default=0, help="gpu id to use")
ap.add_argument("--l2", type=float, default=0., help="the l2 regularization")
ap.add_argument("--lr", type=float, default=0.0001, help='the initial learning rate')
ap.add_argument("--model_type", type=str, default='HierarchyAttention', help='the name of the model being run')
ap.add_argument("--win_size", type=int, help='the size of the context')
ap.add_argument("--sufix", type=str, default='',help='the sufix of experiment data')
ap.add_argument("--update_word_emb", action='store_true', help='update word embedding or not')
ap.add_argument('--glove_file', default='./')
ap.add_argument('--train_or_test', default="test", type=str)
args = ap.parse_args()

args.result_folder = args.result_folder + "%s_H%d_B%d_lt%fD%fU%s_lr%f" % (args.sufix,args.message_context_size, args.batch_size,args.l2, args.sentence_dropout_p, args.update_word_emb, args.lr)

if not os.path.exists(args.result_folder):
    os.makedirs(args.result_folder)

log_file = os.path.join(args.result_folder, "log.txt")
log = open(log_file, 'w')
log.write("train_loss\ttrain_acc\tvalid_loss\tvalid_acc\n")

if args.cuda:
    torch.cuda.set_device(args.gpuid)

"""
Load Train/Valid/Test file
"""
train_file = os.path.join(args.data_folder, "train.csv")
#valid_file = os.path.join(args.data_folder, "valid%s.json" % args.sufix)
test_file = os.path.join(args.data_folder, "test.csv")

glove_file = os.path.join(args.glove_file, args.word_vec)
'''
with open(glove_file, 'rb') as gf:
    word_embedding = pickle.load(gf) # numpy array
    vocabulary = pickle.load(gf) # diction {word:word_id}
'''
vocabulary = dict()
word_embedding = []
'''
with open(glove_file, 'r', encoding="utf8") as glove_in:
        for line in glove_in.readlines():
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            wordEmbedding = np.array([float(value) for value in values[1:]])
            embeddings_index[word] = coefs
            #print(wordEmbedding.shape)
            word_embedding.append(coefs)
vocabulary = embeddings_index
word_embedding = embeddings_index
'''
word_embedding = np.random.uniform(-1, 1, size=[400001, 300])
with open(glove_file, 'r', encoding="utf8") as glove_in:
    num_matched_vocab = 0
    for line in glove_in.readlines():
        word, vec = line.strip().split(" ",1)
        vec = np.fromstring(vec, sep=" ")
        if word in vocabulary:
            word_embedding[vocabulary[word]] = vec
            num_matched_vocab += 1
'''
with open(glove_file, 'rb') as gf:
    word_embedding = pickle.load(gf) # numpy array
    vocabulary = pickle.load(gf) # diction {word:word_id}
'''
print("Load word2vec with size ", word_embedding.shape)

"""
Transform the sentence and context into word ids
train_data = [{"sentence": [[...]], "context": [sentences], "label":0|1}, {}]
"""
corpus = dataset.MeetingCorpus(vocabulary=vocabulary)
if args.win_size is None:
    train_data = corpus.transform(train_file)
    #valid_data = corpus.transform(valid_file)
    test_data = corpus.transform(test_file)
else:
    train_data = corpus.transform_context(train_file, args.win_size)
    #valid_data = corpus.transform_context(valid_file, args.win_size)
    test_data = corpus.transform_context(test_file, args.win_size)

print("Train %d, Test %d" % (len(train_data), len(test_data)))

"""
Wrap with Dataset
"""
if args.train_or_test == 'train':
    train_data = dataset.MeetingDataset(train_data)
    #valid_data = dataset.MeetingDataset(valid_data)
test_data = dataset.MeetingDataset(test_data)

if args.train_or_test == 'train':
    train_loader = dataset.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.meeting_collate)
    #valid_loader = dataset.DataLoader(valid_data, batch_size=args.batch_size, collate_fn=dataset.meeting_collate)
test_loader = dataset.DataLoader(test_data, batch_size=args.batch_size, collate_fn=dataset.meeting_collate)

#print("Batch_Size %d: Train Batches %d, Test Batches %d" % (args.batch_size, len(train_loader), len(test_loader)))

"""
Build Model
"""
vocab_size = len(vocabulary)
word_embedding = torch.FloatTensor(word_embedding)

model = eval(args.model_type)(vocab_size=vocab_size, word_emb_size=args.word_emb_size, 
        pretrained_word_embedding=word_embedding, freeze_word_emb=not args.update_word_emb, 
        sentence_hidden_size=args.sentence_hidden_size, message_hidden_size=args.message_hidden_size,
        sentence_context_size=args.sentence_context_size, message_context_size=args.message_context_size,
        num_classes=args.num_classes, sentence_dropout_p=args.sentence_dropout_p,
        message_dropout_p=args.message_dropout_p, output_dropout_p=args.output_dropout_p)

"""
Define loss function
"""
loss_fn = nn.CrossEntropyLoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

if args.cuda:
    model.cuda()
    loss_fn.cuda()

"""
Training
"""
val_f1_best = -1e10
val_acc_best = -1e10
stop_training = False
best_model_name = ""

def train_epoch(epoch):
    print('\nTraining: Epoch ' + str(epoch))
    if args.fine_tune:
        save_name = os.path.join(args.result_folder, "best_model_%03d.pt" % epoch)
        best_model_name = save_name
        model.load(best_model_name)
    model.train()
    all_costs = []
    logs = []
    log_cost = 0.
    
    last_time = time.time()
    correct = 0.
    pos_correct = 0.
    
    sample_count = 0
    pos_pred = 0
    pos = 0

    for bid, train_batch in enumerate(train_loader):
        b_start = time.time()
        # prepare batch
        sentence_batch = train_batch["sentence"]
        target_batch = train_batch["label"]
        
        k = len(target_batch) # actual batch_size

        if args.cuda:
            target_batch = Variable(torch.LongTensor(target_batch).cuda())
        else:
            target_batch = Variable(torch.LongTensor(target_batch))

        # model forward
        if args.cuda():
            output, sentence_attention = model(sentence_batch.cuda())
        else:
            output, sentence_attention = model(sentence_batch)
        loss = loss_fn(output, target_batch)
        print(loss.item())
        all_costs.append(loss.item())
        log_cost += loss.item()
        pred = output.data.max(1)[1]
        
        #compute correct and pos_correct
        correct += (target_batch.data == pred).sum()

        #backward
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        # statistic
        sample_count += k

        if len(all_costs) == 100:
            logs.append('{0}/{1}; loss {2}; sentence/s {3} accuracy train : {4} '.format(
                bid+1, len(train_loader), round(np.mean(all_costs), 2),
                int(sample_count / (time.time() - last_time)), 
                round(100. * correct.item() / sample_count, 4)
                ))
            print(logs[-1])
            all_costs = []
        b_end = time.time()
        
    train_acc = round(100 * correct.item() / sample_count)
    
    train_loss = log_cost/sample_count

    print('Results : epoch {0}; Train Accuracy {1} Loss{2}'.format(epoch, train_acc, train_loss))
    return train_acc, train_loss

def evaluate(epoch, eval_type='valid'):
    model.eval()
    correct = 0.
    
    global val_f1_best, best_model_name, val_acc_best

    if eval_type == 'valid':
        print('\n Valiation: epoch {0}'.format(epoch))
    
    
    probabilities = []
    labels = []
    sample_count = 0.
    log_cost = 0.
    data_loader = []
    #load batch
    if eval_type == "train":
        data_loader = train_loader
    elif eval_type == "test":
        data_loader = test_loader
    '''
    elif eval_type == "valid":
        data_loader = valid_loader
    '''


    for bid, data_batch in enumerate(data_loader):
        sentence_batch = data_batch["sentence"]
        target_batch = data_batch["label"]
        labels.extend(target_batch)
        sample_count += len(target_batch)

        if args.cuda:
            target_batch = Variable(torch.LongTensor(target_batch).cuda())
        else:
            target_batch = Variable(torch.LongTensor(target_batch))
        
        output, sentence_attention = model(sentence_batch)
        loss = loss_fn(output, target_batch)
        
        log_cost += loss.item()
        pred = output.data.max(1)[1] # torch.Tensor

        correct += pred.long().eq(target_batch.data.long()).cpu().sum()

        prob = torch.nn.functional.softmax(output, dim=1).data.cpu().numpy().tolist()
        probabilities.extend(prob)

    # compute precision, recall, f1, accu
    acc = correct / sample_count
    val_cost = log_cost / sample_count
    y_pred = np.amax(probabilities, axis=1)
    auc = roc_auc_score(labels, y_pred)
    print('Val ROC AUC: %.3f' % auc)

    print('Epoch {0} : {1} acc{2} loss{3}'.format(
            epoch, eval_type, acc, val_cost
        ))
    
    # save the best model
    #revert later => if eval_type == "valid":
    if eval_type == "train":
        if acc > val_acc_best:
            val_acc_best = acc
            save_name = os.path.join(args.result_folder, "best_model_%03d.pt" % epoch)
            best_model_name = save_name
            torch.save(model, save_name)

    return acc, probabilities, labels, val_cost


"""
Start Training Epoch
"""
epoch = 1




if args.train_or_test == 'train':
    while not stop_training and epoch <= args.epochs:
        st_time = time.time()
        train_acc,  train_loss = train_epoch(epoch)
        train_acc, _, _, train_loss = evaluate(epoch, "train")
        #val_acc, val_prob, val_label, val_loss = evaluate(epoch, "train")
        # recrod the intermediate results
        log.write("%0.4f\t%0.4f\n" % (train_loss,  train_acc))
        epoch += 1
        ed_time = time.time()
        print('-'*80, 'time elapsed %d' % (ed_time - st_time))
        log.flush()

    log.flush()
    log.close()

if args.train_or_test == 'test':
    save_name = os.path.join(args.result_folder, "best_model_%03d.pt" % epoch)
    best_model_name = save_name

#Save the last model
last_model_name = os.path.join(args.result_folder, "last_model.pt")
torch.save(model, last_model_name)
acc, probs, labels, _ = evaluate(epoch, "test")
print('Last Model Predict Test: Epoch {0} acc {1}'.format(epoch, acc))

with open(os.path.join(args.result_folder, "last_predictions.txt"), 'w') as pf:
    for p_l, t_l in zip(probs, labels):
        pf.write("%s\t%d\n" % ('\t'.join(["%0.4f" % p for p in p_l]), t_l))
# Run the best model on the test set
del model
model = torch.load(best_model_name)


acc, probs, labels, _ = evaluate(epoch, "test")
y_pred = np.amax(probs, axis=1)

auc = roc_auc_score(labels, y_pred)
print('Test ROC AUC: %.3f' % auc)
print('Best Model Predict Test: Epoch {0}  acc {1}'.format(epoch, acc))

with open(os.path.join(args.result_folder, "predictions.txt"), 'w') as pf:
    for p_l, t_l in zip(probs, labels):
        pf.write("%s\t%d\n" % ('\t'.join(["%0.4f" % p for p in p_l]), t_l))

if __name__ == "__main__":
    pass

