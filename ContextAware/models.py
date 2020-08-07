#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentionRNN(nn.Module):
    def __init__(self, input_size, hidden_size, attention_vec_size):
        super(AttentionRNN, self).__init__()

        self.encoder = nn.GRU(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.u = nn.Parameter(torch.Tensor(attention_vec_size,1).uniform_(-0.1,0.1))
        self.projection = nn.Linear(hidden_size*2, attention_vec_size)
        self.nonlinearity = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, sequence_lens):
        is_cuda = next(self.parameters()).is_cuda

        if is_cuda:
            sorted_seq_lens, sorted_ids = torch.sort(torch.Tensor(sequence_lens).cuda(), descending=True)
            sorted_seq_lens = sorted_seq_lens.cpu().numpy().astype(int)
        else:
            sorted_seq_lens, sorted_ids = torch.sort(torch.Tensor(sequence_lens), descending=True)
            sorted_seq_lens = sorted_seq_lens.numpy().astype(int)
        
        unsorted_ids = torch.sort(sorted_ids)[1]
        x = x[sorted_ids]
        packed = pack_padded_sequence(x, sorted_seq_lens, batch_first=True)
        output, _ = self.encoder(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # reverse the outpyut
        output = output[unsorted_ids, :, :]

        att_tensors = Variable(torch.zeros((output.size(0), output.size(1))))
        weighted_tensors = Variable(torch.zeros((output.size(0), output.size(2))))

        if is_cuda:
            att_tensors = att_tensors.cuda()
            weighted_tensors = weighted_tensors.cuda()

        for i in range(output.size(0)):
            tensors = output[i,:sequence_lens[i],:]
            proj = self.projection(tensors)
            proj = self.nonlinearity(proj)

            attention = torch.mm(proj, self.u)
            attention = self.softmax(attention.t())
            attention = attention.view(-1)

            weighted_tensors[i,:] = tensors.transpose(0,1).mv(attention)
            att_tensors[i,:sequence_lens[i]] = attention

        return weighted_tensors, att_tensors, output


class WordToSentenceRNN(nn.Module):
    def __init__(self, config):
        super(WordToSentenceRNN, self).__init__()

        self.config = config
        
        self.word_embedding = nn.Embedding(config["vocab_size"], config["word_emb_size"])
        if "pretrained_word_embedding" in config:
            self.word_embedding.weight.data = torch.Tensor(config["pretrained_word_embedding"])

        if config["freeze_word_emb"]:
            for p in self.word_embedding.parameters():
                p.require_grad = False
        
        self.word_emb_dropout = nn.Dropout(p=config["word_emb_dropout"])
        self.sen_dropout = nn.Dropout(p=config["sen_dropout"])

        rnn_input_size = config["word_emb_size"]
        self.sen_encoder = AttentionRNN(rnn_input_size, config["sen_hidden_size"], config["sen_attention_size"])

    def forward(self, batch_x):
        """
        batch_x: 
            list of list of int
            [[1,2], [3,4,5]]
        """
        
        padded_sentences, sentence_lens = pad_batch_sentence(batch_x, self.config["max_sen_size"])
        if self.config["with_cuda"]:
            padded_sentences = padded_sentences.cuda()
        embedding = self.word_embedding(padded_sentences)
        dropout_embedding = self.word_emb_dropout(embedding)
        sen_tensors, sen_att_tensors, _ = self.sen_encoder(dropout_embedding, sentence_lens)
        sen_tensors = self.sen_dropout(sen_tensors)

        return {"representation": sen_tensors, "attention": sen_att_tensors}



class SentenceToMessageRNN(nn.Module):
    def __init__(self, config):
        super(SentenceToMessageRNN, self).__init__()
        self.config = config

        self.sentence_encoder = WordToSentenceRNN(config)
        
        mess_rnn_input_size = config["sen_hidden_size"] * 2
        self.message_encoder = AttentionRNN(mess_rnn_input_size, config["mess_hidden_size"], config["mess_attention_size"])
        self.mess_dropout = nn.Dropout(p=config["mess_dropout"])
    
    def forward(self, batch_x):
        """
        batch_x:
            list of list of list of int
            [
                [[1,2], [2,3,4]],
                [[1,2], [2,3,4]]
            ]
        """
        messages = Variable(torch.zeros((len(batch_x), self.config["mess_hidden_size"]*2)))
        if self.config["with_cuda"]:
            messages = messages.cuda()

        attentions = []
        for i, doc in enumerate(batch_x):
            results = self.sentence_encoder(doc)
            sen_tensors, sen_attention_tensors = results["representation"], results["attention"]
            sen_tensors = torch.unsqueeze(sen_tensors, 0)
            message_tensors, message_attention_tensors, _ = self.message_encoder(sen_tensors, [len(doc)])
            messages[i, :] = torch.squeeze(message_tensors, 0)
            attentions.append({"mess_att": message_attention_tensors, "sen_att": sen_attention_tensors})
        
        messages = self.mess_dropout(messages)

        return {"representation": messages, "attention": attentions}

class WordToSentenceCNN(nn.Module):
    def __init__(self, config):
        super(WordToSentenceCNN, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(config["vocab_size"], config["word_emb_size"])
        if "pretrained_word_embedding" in config:
            self.embedding.weight.data = torch.Tensor(config["pretrained_word_embedding"])

        if config["freeze_word_emb"]:
            for p in self.word_embedding.parameters():
                p.require_grad = False

        self.word_emb_dropout = nn.Dropout(p=config["word_emb_dropout"])
        # conv layers
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=config["kernel_num"], kernel_size=[kernel_size, config["word_emb_size"]]) for kernel_size in config["kernel_sizes"]])
        self.sen_dropout = nn.Dropout(p=config["sen_dropout"])
    
    def forward(self, batch_x):
        """
            batch_x: list of list of word ids
            [
                [1,2],
                [2,3,4]
            ]
        """
        padded_sentences, sentence_lens = pad_batch_sentence(batch_x, self.config["max_sen_size"])

        if self.config["with_cuda"]:
            padded_sentences = padded_sentences.cuda()

        x = self.embedding(padded_sentences) # N * W * D
        x = self.word_emb_dropout(x)

        x = x.unsqueeze(1) # N * 1 * W * D
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        
        x = self.sen_dropout(x)
        
        return {"representation": x}


class FlatMessageCNN(nn.Module):
    """consider message as a long text"""
    def __init__(self, config):
        super(FlatMessageCNN, self).__init__()
        
        self.text_encoder = WordToSentenceCNN(config)
    
    def forward(self, batch_x):
        # unsqueeze doc to a long text
        batch_x = [ [w for sen in doc for w in sen] for doc in batch_x]
        representation = self.text_encoder(batch_x)["representation"]

        return {"representation": representation}

class FlatMessageCNNContextOnly(nn.Module):
    def __init__(self, config):
        super(FlatMessageCNNContextOnly, self).__init__()
        self.text_encoder = WordToSentenceCNN(config)

    def forward(self, batch_x):
        sen_x, context_x = batch_x
        context_x = [[w for sen in doc for w in sen] for doc in context_x]
        representation = self.text_encoder(context_x)["representation"]
        return {"representation": representation}

class FlatMessageRNN(nn.Module):
    """consider message as a long text"""
    def __init__(self, config):
        super(FlatMessageRNN, self).__init__()
        
        self.text_encoder = WordToSentenceRNN(config)
    
    def forward(self, batch_x):
        # unsqueeze doc to a long text
        batch_x = [ [w for sen in doc for w in sen] for doc in batch_x]
        representation = self.text_encoder(batch_x)["representation"]

        return {"representation": representation}


class SenMessCNN(nn.Module):
    """
        concatnate the sent and mess representations
    """
    def __init__(self, config):
        super(SenMessCNN, self).__init__()
        self.text_encoder = WordToSentenceCNN(config)

    def forward(self, batch_x):
        sen_x, context_x = batch_x
        # flat the context_x to a long text
        context_x = [[w for sen in doc for w in sen] for doc in context_x]
        sen_representation = self.text_encoder(sen_x)["representation"]
        mess_representation = self.text_encoder(context_x)["representation"]

        representation = torch.cat([sen_representation, mess_representation], 1)
        return {"representation": representation}

class BiAttentionSenMessRNN(nn.Module):
    """
    Bi-Attention version of sentence message interaction
    """
    def __init__(self, config):
        super(BiAttentionSenMessRNN, self).__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config["vocab_size"], config["word_emb_size"])
        if "pretrained_word_embedding" in config:
            self.word_embedding.weight.data = torch.Tensor(config["pretrained_word_embedding"])

        self.word_emb_dropout = nn.Dropout(p=config["word_emb_dropout"])
        self.sen_dropout = nn.Dropout(p=config["sen_dropout"])

        self.sen_encoder = AttentionRNN(config["word_emb_size"], config["sen_hidden_size"], config["sen_attention_size"])
        self.proj = nn.Linear(config["sen_hidden_size"]*2, config["sen_hidden_size"])
        
        self.att_u = nn.Parameter(torch.Tensor(config["sen_attention_size"],1).uniform_(-0.1,0.1))
        self.att_nonlinearity = nn.Tanh()
        self.att_proj = nn.Linear(config["sen_hidden_size"]*4, config["sen_attention_size"])

        if config["attention_type"] == "dot_attention":
            self.sen_mess_attention = self._dot_attention
        elif config["attention_type"] == "con_attention":
            self.sen_mess_attention = self._con_attention
            self.v = nn.Parameter(torch.Tensor(config["sen_hidden_size"]*6,1).uniform_(-0.1,0.1))

    def _dot_attention(self, s_states, m_states):
        sen_proj = F.relu(self.proj(s_states))
        m_proj = F.relu(self.proj(m_states))
        weight_matrix = F.softmax(torch.mm(sen_proj, m_proj.t()), dim=1)
        m_info = torch.mm(weight_matrix, m_states)
        return m_info, weight_matrix

    def _con_attention(self, s_states, m_states):
        context_info = Variable(torch.zeros([s_states.size(0), m_states.size(1)]))
        weight_matrix = []
        if self.config["with_cuda"]:
            context_info = context_info.cuda()

        for i in range(s_states.size(0)):
            num_sen = m_states.size(0)
            word_rep = s_states[i].repeat(num_sen, 1)
            word_rep = torch.cat([word_rep, m_states, word_rep * m_states], dim=1)
            
            attention = F.softmax(torch.mm(word_rep, self.v).t(), dim=1).view(-1)
            
            mess_info = m_states.transpose(0,1).mv(attention)
            context_info[i,:] = mess_info
            weight_matrix.append(attention)
        
        return context_info, weight_matrix

    def forward(self, batch_x):
        sen_x, context_x = batch_x
        
        is_cuda = next(self.parameters()).is_cuda

        padded_sentences, sentence_lens = pad_batch_sentence(sen_x, self.config["max_sen_size"])
        if self.config["with_cuda"]:
            padded_sentences = padded_sentences.cuda()

        sen_embedding = self.word_embedding(padded_sentences)
        dropout_sen_embedding = self.word_emb_dropout(sen_embedding)
        _, _, sen_output = self.sen_encoder(dropout_sen_embedding, sentence_lens)

        weighted_tensors = Variable(torch.zeros((sen_output.size(0), sen_output.size(2)*2)))
        sen_att_tensors = Variable(torch.zeros((sen_output.size(0), sen_output.size(1))))
        context_att_tensors = []

        if is_cuda:
            weighted_tensors = weighted_tensors.cuda()
            sen_att_tensors = sen_att_tensors.cuda()

        for i in range(sen_output.size(0)):
            sen_states = sen_output[i,:sentence_lens[i],:]
            
            # compute sentence representations in 
            context = context_x[i]
            padded_context, context_lens = pad_batch_sentence(context, self.config["max_sen_size"])
            if is_cuda:
                padded_context = padded_context.cuda()
            context_embedding = self.word_embedding(padded_context)
            dropout_context_embedding = self.word_emb_dropout(context_embedding)
            context_tensors, _, _ = self.sen_encoder(dropout_context_embedding, context_lens)

            # compute the interactiom between word and sentence
            context_info, attention_weight = self.sen_mess_attention(sen_states, context_tensors)

            sen_states = torch.cat([sen_states, context_info], 1)
            sen_states = self.sen_dropout(sen_states)
            
            sen_att_proj = self.att_proj(sen_states)
            sen_att_proj = self.att_nonlinearity(sen_att_proj)

            attention = torch.mm(sen_att_proj, self.att_u)
            attention = F.softmax(attention.t(), dim=1).view(-1)

            weighted_sen = sen_states.transpose(0,1).mv(attention)

            weighted_tensors[i,:] = weighted_sen
            sen_att_tensors[i,:sentence_lens[i]] = attention
            context_att_tensors.append(attention_weight)

        return {"representation": weighted_tensors, "sen_attention": sen_att_tensors, "mess_attention": context_att_tensors}



def pad_doc(doc_batch, max_doc_len=15, max_sen_size=40):
    doc_lens = np.array([min(len(d), max_doc_len) for d in doc_batch])# number of sentences in each doc
    max_doc = max(doc_lens)
    
    padded_doc = []
    doc_sen_lens = []
    for doc in doc_batch:
        new_doc = []
        sen_lens = [min(len(s), max_sen_size) for s in doc[:max_doc]] + [0] * max(0, max_doc - len(doc))
        max_sen = max(sen_lens)
        for s in doc[:max_doc]:
            s = s[:max_sen] + [0]*max(0, max_sen - len(s))
            new_doc.append(s)
        
        new_doc = new_doc + [[0]*max_sen] * max(0, max_doc - len(doc))
        
        padded_doc.append(new_doc)
        doc_sen_lens.append(sen_lens)

    return padded_doc, doc_lens, doc_sen_lens



def pad_batch_sentence(sentence_batch, max_sen_size=40, min_sen_size=4):
    """
    sentence_batch: list of list of word ids
    """
    sen_lens = np.array([min(len(s), max_sen_size) for s in sentence_batch])
    max_len = max(min_sen_size, max(sen_lens))
    
    # we assume zero is the padding word id
    new_sentence_batch = np.zeros((len(sen_lens), max_len))
    for i, sen in enumerate(sentence_batch):
        new_sentence_batch[i, :sen_lens[i]] = np.array(sen[:sen_lens[i]])

    new_sentence_batch = new_sentence_batch.astype(int)
    
    return Variable(torch.LongTensor(new_sentence_batch)), sen_lens


class DNNClassifier(nn.Module):
    def __init__(self, config):
        super(DNNClassifier, self).__init__()
        self.config = config

        self.model = eval(config["model_name"])(config)

        self.likelihood = nn.Linear(config["feature_size"], config["num_classes"])

    def forward(self, batch_x):
        features = self.model(batch_x)
        logits = self.likelihood(features["representation"])
        preds = torch.max(logits, dim=1)[1]
        
        output = {k:v for k, v in features.items() if k != "representation"}
        output["logits"] = logits
        output["predictions"] = preds
        output["probs"] = F.softmax(logits, dim=1)
        
        return output

if __name__ == "__main__":
    pass

