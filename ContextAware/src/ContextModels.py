#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

np.random.seed(100)
torch.manual_seed(1234)

class HierarchyAttention(nn.Module):
    """
    Define two level of Attention Layer to context encoding:
        first layer: word2sentence
        second layer: sentence2message
    """
    def __init__(self, vocab_size, word_emb_size, pretrained_word_embedding, freeze_word_emb,
            sentence_hidden_size, message_hidden_size, 
            sentence_context_size, message_context_size, num_classes, 
            sentence_dropout_p=0.0, message_dropout_p=0.0, output_dropout_p=0.0, 
            sentence_proj_nonlinearity=nn.ReLU, message_proj_nonlinearity=nn.ReLU, 
            word_to_sen_rnn=nn.GRU, sen_to_message_rnn=nn.GRU):
        """
            word_emb_size: the dimention of word embedding (int)
            pretrained_word_embedding: the pretrained word embedding (None or FloatTensor)
            freeze_word_emb: True|False , freeze the word embedding layer or not
            sentence_hidden_size: the number of hidden units in word2sen rnn layer (int)
            message_hidden_size: the number of hidden units in sen2message rnn layer (int)
            sentence_context_size: the dimention of word2sen context vector (int)
            message_context_size: the dimention of sen2message context vector (int)
            sentence_dropout_p: the dropout p in word2sen rnn (float)
            message_dropout_p: the dropout p in sen2wrod rnn (float)
            output_dropout_p: the dropout p in final MLP layer (float)
            sentence_proj_nonlinearity: the non-linear activation for sentence proj (nn.function)
            message_proj_nonlinearity: the non-linear activation for message proj (nn.function)
            word_to_sen_rnn: the RNN type for word2sen (nn.rnn)
            sen_to_message_rnn: the RNN type for sen2message (nn.rnn)
        """
        super(HierarchyAttention, self).__init__()
        
        self.sentence_hidden_size = sentence_hidden_size
        self.message_hidden_size = message_hidden_size
        self.sentence_context_size = sentence_context_size
        self.message_context_size = message_context_size
        self.num_classes = num_classes

        # dropout layer in the final MLP
        self.output_dropout = nn.Dropout(p=output_dropout_p)

        # the word to sentence encoder
        self.word2sen = WordToSentence(vocab_size, word_emb_size, pretrained_word_embedding, freeze_word_emb,
                sentence_hidden_size, sentence_context_size, sentence_dropout_p, 
                sentence_proj_nonlinearity, word_to_sen_rnn)

        #self.word2sen = nn.DataParallel(self.word2sen)

        # the sentence to message encoder
        self.sen2message = SentenceToMessage(sentence_hidden_size * 2, message_hidden_size,
                message_context_size, message_dropout_p, 
                message_proj_nonlinearity, sen_to_message_rnn)
        
        self.full_conn = nn.Linear(message_hidden_size * 2 + sentence_hidden_size * 2, message_hidden_size)
        # the last MLP Layer
        self.likelihood = nn.Linear(message_hidden_size, self.num_classes)
        #self.likelihood = nn.Linear(sentence_hidden_size * 2, self.num_classes)

    def is_cuda(self):
        return "cuda" in str(type(self.likelihood.weight.data))

    def _sen2var(self, sen_batch):
        sen_lens = [len(s) for s in sen_batch]
        max_sen_len = max(sen_lens)
        batch_size = len(sen_lens)

        var_sen_lens = Variable(torch.LongTensor(sen_lens))
        var_sentence_batch = Variable(torch.zeros(batch_size, max_sen_len).long())
        for i in range(batch_size):
            var_sentence_batch[i,:sen_lens[i]] = torch.LongTensor(sen_batch[i])
        
        return var_sentence_batch.cuda(), var_sen_lens.cuda(), max_sen_len

    def forward(self, sentence_batch, context_batch):
        """ 
        inputs:
            sentence_batch: numpy.array with list of sentence, each sentence is a list of word ids
            context_batch: numpy.array with list of doc, each doc is a list of sentence
        outputs:
            output: N * num_classes
            sentence_attentions : N * max_sen_len in sentence batch
            context_attentions  : N * max_num_sen in context batch
        
        example:
            inputs: 
                sentence_batch: [[1,2,3], [1,2,3,4]]
                context_batch: np.array([[[1,2,3], [1,2,3,4]], [[1,2,3], [1,2,3,4]]])
            outputs:
                output: [[-0.0360 -0.2134], [0.0368 -0.1594]]
                sentence_attentions: [[ 0.3402  0.3222  0.3376  0.0000], [0.2569  0.2433  0.2517  0.2481]]
                context_attentions: [[0.5002  0.4998], 0.5002  0.4998]
        """
        batch_size = len(sentence_batch)
        context_len = [len(context_batch[i]) for i in range(batch_size)]
        max_context_len = max(context_len)
        sen_lens = [len(s) for s in sentence_batch]
        max_sen_len = max(sen_lens)
        
        # save sentence batch to Variable
        var_sentence_batch, var_sentence_lens, max_len = self._sen2var(sentence_batch)
        # save context batch into Variable
        var_context_batch = []
        for context in context_batch:
            var_context_batch.append(self._sen2var(context[:10]))
        
        # define a placeholder for message representation
        if self.is_cuda():
            messages = Variable(torch.zeros(batch_size, self.message_hidden_size * 2).cuda())
            context_attentions = Variable(torch.zeros(batch_size, max_context_len).cuda())
        else:
            messages = Variable(torch.zeros(batch_size, self.message_hidden_size * 2))
            context_attentions = Variable(torch.zeros(batch_size, max_context_len))

        for i in range(batch_size):
            context_sentence_tensors, context_sentence_attentions = self.word2sen(var_context_batch[i][0], var_context_batch[i][1],var_context_batch[i][2])
            message, message_attentions = self.sen2message(context_sentence_tensors)
            messages[i, :] = torch.squeeze(message, 0)
            context_attentions[i, :min(10,context_len[i])] = torch.squeeze(message_attentions, 0)
        # get the representation for the target sentence
        sentences, sentences_attentions = self.word2sen(var_sentence_batch, var_sentence_lens, max_len)
        
        # concatenate the sentence rep with context rep
        self.final_sentences = torch.cat([sentences, messages], dim=1)
        self.final_sentences = self.output_dropout(self.final_sentences)
        self.final_sentences = nn.ReLU()(self.full_conn(self.final_sentences))
        self.final_sentences = self.output_dropout(self.final_sentences)
        
        output = self.likelihood(self.final_sentences)

        return output, sentences_attentions, context_attentions

class ContextAttention(nn.Module):
    """
    Define two level of Attention Layer to context encoding:
        first layer: word2sentence
        second layer: sentence2message
    """
    def __init__(self, vocab_size, word_emb_size, pretrained_word_embedding, freeze_word_emb,
            sentence_hidden_size, message_hidden_size, 
            sentence_context_size, message_context_size, num_classes, 
            sentence_dropout_p=0.0, message_dropout_p=0.0, output_dropout_p=0.0, 
            sentence_proj_nonlinearity=nn.Tanh, message_proj_nonlinearity=nn.Tanh, 
            word_to_sen_rnn=nn.GRU, sen_to_message_rnn=nn.GRU):
        """
            word_emb_size: the dimention of word embedding (int)
            pretrained_word_embedding: the pretrained word embedding (None or FloatTensor)
            freeze_word_emb: True|False , freeze the word embedding layer or not
            sentence_hidden_size: the number of hidden units in word2sen rnn layer (int)
            message_hidden_size: the number of hidden units in sen2message rnn layer (int)
            sentence_context_size: the dimention of word2sen context vector (int)
            message_context_size: the dimention of sen2message context vector (int)
            sentence_dropout_p: the dropout p in word2sen rnn (float)
            message_dropout_p: the dropout p in sen2wrod rnn (float)
            output_dropout_p: the dropout p in final MLP layer (float)
            sentence_proj_nonlinearity: the non-linear activation for sentence proj (nn.function)
            message_proj_nonlinearity: the non-linear activation for message proj (nn.function)
            word_to_sen_rnn: the RNN type for word2sen (nn.rnn)
            sen_to_message_rnn: the RNN type for sen2message (nn.rnn)
        """
        super(ContextAttention, self).__init__()
        
        self.sentence_hidden_size = sentence_hidden_size
        self.message_hidden_size = message_hidden_size
        self.sentence_context_size = sentence_context_size
        self.message_context_size = message_context_size
        self.num_classes = num_classes

        # dropout layer in the final MLP
        self.output_dropout = nn.Dropout(p=output_dropout_p)

        # the word to sentence encoder
        self.word2sen = WordToSentence(vocab_size, word_emb_size, pretrained_word_embedding, freeze_word_emb,
                sentence_hidden_size, sentence_context_size, sentence_dropout_p, 
                sentence_proj_nonlinearity, word_to_sen_rnn)

        #self.word2sen = nn.DataParallel(self.word2sen)

        # the sentence to message encoder
        self.sen2message = SentenceToMessage(sentence_hidden_size * 2, message_hidden_size,
                message_context_size, message_dropout_p, 
                message_proj_nonlinearity, sen_to_message_rnn)

        # the last MLP Layer
        self.likelihood = nn.Linear(message_hidden_size * 2, self.num_classes)
        #self.likelihood = nn.Linear(sentence_hidden_size * 2, self.num_classes)

    def is_cuda(self):
        return "cuda" in str(type(self.likelihood.weight.data))

    def _sen2var(self, sen_batch):
        sen_lens = [len(s) for s in sen_batch]
        max_sen_len = max(sen_lens)
        batch_size = len(sen_lens)

        var_sen_lens = Variable(torch.LongTensor(sen_lens))
        var_sentence_batch = Variable(torch.zeros(batch_size, max_sen_len).long())
        for i in range(batch_size):
            var_sentence_batch[i,:sen_lens[i]] = torch.LongTensor(sen_batch[i])
        
        return var_sentence_batch.cuda(), var_sen_lens.cuda(), max_sen_len

    def forward(self, sentence_batch, context_batch):
        """ 
        inputs:
            context_batch: numpy.array with list of doc, each doc is a list of sentence
        outputs:
            output: N * num_classes
            sentence_attentions : N * max_sen_len in sentence batch
            context_attentions  : N * max_num_sen in context batch
        
        example:
            inputs: 
                sentence_batch: [[1,2,3], [1,2,3,4]]
                context_batch: np.array([[[1,2,3], [1,2,3,4]], [[1,2,3], [1,2,3,4]]])
            outputs:
                output: [[-0.0360 -0.2134], [0.0368 -0.1594]]
                sentence_attentions: [[ 0.3402  0.3222  0.3376  0.0000], [0.2569  0.2433  0.2517  0.2481]]
                context_attentions: [[0.5002  0.4998], 0.5002  0.4998]
        """
        batch_size = len(context_batch)
        context_len = [len(context_batch[i]) for i in range(batch_size)]
        max_context_len = max(context_len)
        
        # save context batch into Variable
        var_context_batch = []
        for context in context_batch:
            var_context_batch.append(self._sen2var(context[:10]))
        
        # define a placeholder for message representation
        if self.is_cuda():
            messages = Variable(torch.zeros(batch_size, self.message_hidden_size * 2).cuda())
            context_attentions = Variable(torch.zeros(batch_size, max_context_len).cuda())
        else:
            messages = Variable(torch.zeros(batch_size, self.message_hidden_size * 2))
            context_attentions = Variable(torch.zeros(batch_size, max_context_len))

        for i in range(batch_size):
            context_sentence_tensors, context_sentence_attentions = self.word2sen(var_context_batch[i][0], var_context_batch[i][1],var_context_batch[i][2])
            message, message_attentions = self.sen2message(context_sentence_tensors)
            messages[i, :] = torch.squeeze(message, 0)
            context_attentions[i, :min(10,context_len[i])] = torch.squeeze(message_attentions, 0)
        
        # concatenate the sentence rep with context rep
        messages = self.output_dropout(messages)
        
        output = self.likelihood(messages)

        return output, None, context_attentions


class SentenceAttention(nn.Module):
    """
    Define two level of Attention Layer to context encoding:
        first layer: word2sentence
        second layer: sentence2message
    """
    def __init__(self, vocab_size, word_emb_size, pretrained_word_embedding, freeze_word_emb,
            sentence_hidden_size, message_hidden_size, 
            sentence_context_size, message_context_size, num_classes, 
            sentence_dropout_p=0.0, message_dropout_p=0.0, output_dropout_p=0.0, 
            sentence_proj_nonlinearity=nn.Tanh, message_proj_nonlinearity=nn.Tanh, 
            word_to_sen_rnn=nn.GRU, sen_to_message_rnn=nn.GRU):
        """
            word_emb_size: the dimention of word embedding (int)
            pretrained_word_embedding: the pretrained word embedding (None or FloatTensor)
            freeze_word_emb: True|False , freeze the word embedding layer or not
            sentence_hidden_size: the number of hidden units in word2sen rnn layer (int)
            message_hidden_size: the number of hidden units in sen2message rnn layer (int)
            sentence_context_size: the dimention of word2sen context vector (int)
            message_context_size: the dimention of sen2message context vector (int)
            sentence_dropout_p: the dropout p in word2sen rnn (float)
            message_dropout_p: the dropout p in sen2wrod rnn (float)
            output_dropout_p: the dropout p in final MLP layer (float)
            sentence_proj_nonlinearity: the non-linear activation for sentence proj (nn.function)
            message_proj_nonlinearity: the non-linear activation for message proj (nn.function)
            word_to_sen_rnn: the RNN type for word2sen (nn.rnn)
            sen_to_message_rnn: the RNN type for sen2message (nn.rnn)
        """
        super(SentenceAttention, self).__init__()
        
        self.sentence_hidden_size = sentence_hidden_size
        self.message_hidden_size = message_hidden_size
        self.sentence_context_size = sentence_context_size
        self.message_context_size = message_context_size
        self.num_classes = num_classes

        # dropout layer in the final MLP
        self.output_dropout = nn.Dropout(p=output_dropout_p)

        # the word to sentence encoder
        self.word2sen = WordToSentence(vocab_size, word_emb_size, pretrained_word_embedding, freeze_word_emb,
                sentence_hidden_size, sentence_context_size, sentence_dropout_p, 
                sentence_proj_nonlinearity, word_to_sen_rnn)
        
        self.sentence_nonlinearity = nn.SELU()
        # the last MLP Layer
        self.likelihood = nn.Linear(sentence_hidden_size * 2, self.num_classes)

    def is_cuda(self):
        return "cuda" in str(type(self.likelihood.weight.data))

    def _sen2var(self, sen_batch):
        sen_lens = [len(s) for s in sen_batch]
        max_sen_len = max(sen_lens)
        batch_size = len(sen_lens)

        var_sen_lens = Variable(torch.LongTensor(sen_lens))
        var_sentence_batch = Variable(torch.zeros(batch_size, max_sen_len).long())
        for i in range(batch_size):
            var_sentence_batch[i,:sen_lens[i]] = torch.LongTensor(sen_batch[i])
       
        if self.is_cuda():
            print("CUDA available")
            var_sentence_batch = var_sentence_batch.cuda()
            var_sen_lens = var_sen_lens.cuda()

        return var_sentence_batch, var_sen_lens, max_sen_len

    def forward(self, sentence_batch, context_batch=None):
        """ 
        inputs:
            sentence_batch: numpy.array with list of sentence, each sentence is a list of word ids
            context_batch: numpy.array with list of doc, each doc is a list of sentence
        outputs:
            output: N * num_classes
            sentence_attentions : N * max_sen_len in sentence batch
            context_attentions  : N * max_num_sen in context batch
        
        example:
            inputs: 
                sentence_batch: [[1,2,3], [1,2,3,4]]
                context_batch: np.array([[[1,2,3], [1,2,3,4]], [[1,2,3], [1,2,3,4]]])
            outputs:
                output: [[-0.0360 -0.2134], [0.0368 -0.1594]]
                sentence_attentions: [[ 0.3402  0.3222  0.3376  0.0000], [0.2569  0.2433  0.2517  0.2481]]
                context_attentions: [[0.5002  0.4998], 0.5002  0.4998]
        """
        batch_size = len(sentence_batch)
        sen_lens = [len(s) for s in sentence_batch]
        max_sen_len = max(sen_lens)

        #sentence_batch = np.asarray(sentence_batch, dtype='float32')
        #print(type(sentence_batch))
        # save sentence batch to Variable
        var_sentence_batch, var_sentence_lens, max_len = self._sen2var(sentence_batch)

        # get the representation for the target sentence
        sentences, sentences_attentions = self.word2sen(var_sentence_batch, var_sentence_lens, max_len)
        
        sentences = self.sentence_nonlinearity(sentences)
        output = self.likelihood(sentences)

        return output, sentences_attentions

class DynamicHierarchyAttention(nn.Module):
    """
    Define two level of Attention Layer to context encoding:
        first layer: word2sentence
        second layer: sentence2message
    """
    def __init__(self, vocab_size, word_emb_size, pretrained_word_embedding, freeze_word_emb,
            sentence_hidden_size, message_hidden_size, 
            sentence_context_size, message_context_size, num_classes, 
            sentence_dropout_p=0.0, message_dropout_p=0.0, output_dropout_p=0.0, 
            sentence_proj_nonlinearity=nn.Tanh, message_proj_nonlinearity=nn.Tanh, 
            word_to_sen_rnn=nn.GRU, sen_to_message_rnn=nn.GRU):
        """
            word_emb_size: the dimention of word embedding (int)
            pretrained_word_embedding: the pretrained word embedding (None or FloatTensor)
            freeze_word_emb: True|False , freeze the word embedding layer or not
            sentence_hidden_size: the number of hidden units in word2sen rnn layer (int)
            message_hidden_size: the number of hidden units in sen2message rnn layer (int)
            sentence_context_size: the dimention of word2sen context vector (int)
            message_context_size: the dimention of sen2message context vector (int)
            sentence_dropout_p: the dropout p in word2sen rnn (float)
            message_dropout_p: the dropout p in sen2wrod rnn (float)
            output_dropout_p: the dropout p in final MLP layer (float)
            sentence_proj_nonlinearity: the non-linear activation for sentence proj (nn.function)
            message_proj_nonlinearity: the non-linear activation for message proj (nn.function)
            word_to_sen_rnn: the RNN type for word2sen (nn.rnn)
            sen_to_message_rnn: the RNN type for sen2message (nn.rnn)
        """
        super(DynamicHierarchyAttention, self).__init__()
        
        self.sentence_hidden_size = sentence_hidden_size
        self.message_hidden_size = message_hidden_size
        self.sentence_context_size = sentence_context_size
        self.message_context_size = message_context_size
        self.num_classes = num_classes

        # dropout layer in the final MLP
        self.output_dropout = nn.Dropout(p=output_dropout_p)

        # the word to sentence encoder
        self.word2sen = WordToSentence(vocab_size, word_emb_size, pretrained_word_embedding, freeze_word_emb,
                sentence_hidden_size, sentence_context_size, sentence_dropout_p, 
                sentence_proj_nonlinearity, word_to_sen_rnn)

        #self.word2sen = nn.DataParallel(self.word2sen)

        # the sentence to message encoder
        self.sen2message = DynamicSentenceToMessage(sentence_hidden_size * 2, message_hidden_size,
                message_context_size, message_dropout_p, 
                message_proj_nonlinearity, sen_to_message_rnn)

        # the last MLP Layer
        self.likelihood = nn.Linear(message_hidden_size * 2 + sentence_hidden_size * 2, self.num_classes)
        #self.likelihood = nn.Linear(sentence_hidden_size * 2, self.num_classes)

    def is_cuda(self):
        return "cuda" in str(type(self.likelihood.weight.data))

    def _sen2var(self, sen_batch):
        sen_lens = [len(s) for s in sen_batch]
        max_sen_len = max(sen_lens)
        batch_size = len(sen_lens)

        var_sen_lens = Variable(torch.LongTensor(sen_lens))
        var_sentence_batch = Variable(torch.zeros(batch_size, max_sen_len).long())
        for i in range(batch_size):
            var_sentence_batch[i,:sen_lens[i]] = torch.LongTensor(sen_batch[i])
        
        return var_sentence_batch.cuda(), var_sen_lens.cuda(), max_sen_len

    def forward(self, sentence_batch, context_batch):
        """ 
        inputs:
            sentence_batch: numpy.array with list of sentence, each sentence is a list of word ids
            context_batch: numpy.array with list of doc, each doc is a list of sentence
        outputs:
            output: N * num_classes
            sentence_attentions : N * max_sen_len in sentence batch
            context_attentions  : N * max_num_sen in context batch
        
        example:
            inputs: 
                sentence_batch: [[1,2,3], [1,2,3,4]]
                context_batch: np.array([[[1,2,3], [1,2,3,4]], [[1,2,3], [1,2,3,4]]])
            outputs:
                output: [[-0.0360 -0.2134], [0.0368 -0.1594]]
                sentence_attentions: [[ 0.3402  0.3222  0.3376  0.0000], [0.2569  0.2433  0.2517  0.2481]]
                context_attentions: [[0.5002  0.4998], 0.5002  0.4998]
        """
        batch_size = len(sentence_batch)
        context_len = [len(context_batch[i]) for i in range(batch_size)]
        max_context_len = max(context_len)
        sen_lens = [len(s) for s in sentence_batch]
        max_sen_len = max(sen_lens)
        
        # save sentence batch to Variable
        var_sentence_batch, var_sentence_lens, max_len = self._sen2var(sentence_batch)
        # save context batch into Variable
        var_context_batch = []
        for context in context_batch:
            var_context_batch.append(self._sen2var(context[:10]))
        
        # define a placeholder for message representation
        if self.is_cuda():
            messages = Variable(torch.zeros(batch_size, self.message_hidden_size * 2).cuda())
            context_attentions = Variable(torch.zeros(batch_size, max_context_len).cuda())
        else:
            messages = Variable(torch.zeros(batch_size, self.message_hidden_size * 2))
            context_attentions = Variable(torch.zeros(batch_size, max_context_len))

        sentences, sentences_attentions = self.word2sen(var_sentence_batch, var_sentence_lens, max_len)

        for i in range(batch_size):
            context_sentence_tensors, context_sentence_attentions = self.word2sen(var_context_batch[i][0], var_context_batch[i][1],var_context_batch[i][2])
            message, message_attentions = self.sen2message(sentences[i], context_sentence_tensors)
            messages[i, :] = torch.squeeze(message, 0)
            context_attentions[i, :min(10,context_len[i])] = torch.squeeze(message_attentions, 0)
        
        # concatenate the sentence rep with context rep
        self.final_sentences = torch.cat([sentences, messages], dim=1)
        self.final_sentences = self.output_dropout(self.final_sentences)
         
        output = self.likelihood(self.final_sentences)
        return output, sentences_attentions, context_attentions

class DynamicSentenceToMessage(nn.Module):
    def __init__(self, sentence_emb_size, message_hidden_size, 
            message_context_size, message_dropout_p,
            message_proj_nonlinearity, sen_to_message_rnn):
        super(DynamicSentenceToMessage, self).__init__()

        self.sentence_emb_size = sentence_emb_size
        self.message_hidden_size = message_hidden_size
        self.message_context_size = message_context_size
        self.message_proj_nonlinearity = message_proj_nonlinearity()

        self.message_encoder = sen_to_message_rnn(sentence_emb_size, message_hidden_size, batch_first=True,
                bidirectional=True)
        
        self.target_sentence_proj = nn.Linear(sentence_emb_size, self.message_context_size)
        self.message_proj = nn.Linear(message_hidden_size * 2, message_context_size)
        self.dropout = nn.Dropout(p=message_dropout_p)
        self.softmax = nn.Softmax()
    
    def forward(self, target_sentences, sentence_tensors):
        if sentence_tensors.dim() == 2:
            sentence_tensors = torch.unsqueeze(sentence_tensors, 0)
        if target_sentences.dim() == 1:
            target_sentences = torch.unsqueeze(target_sentences, 0)

        target_projs = self.target_sentence_proj(target_sentences)
        target_projs = self.message_proj_nonlinearity(target_projs)

        sentence_tensors = self.dropout(sentence_tensors)
        output, _ = self.message_encoder(sentence_tensors)
        
        messages = Variable(torch.zeros(output.size(0), self.message_hidden_size * 2).cuda())
        message_attentions = Variable(torch.zeros(output.size(0), output.size(1)).cuda())

        for i in range(output.size(0)):
            sentence_tensors = output[i] # size is T * message_hidden_size 2
            sentence_proj = self.message_proj(sentence_tensors)
            sentence_proj = self.message_proj_nonlinearity(sentence_proj)

            # project the target sentence
            target_proj = target_projs[i] # vector
            attention = torch.mv(sentence_proj, target_proj)
            attention = self.softmax(attention)
            
            message_attentions[i, :] = attention
            messages[i, :] = sentence_tensors.transpose(1,0).mv(attention)

        return messages, message_attentions

class SentenceToMessage(nn.Module):
    def __init__(self, sentence_emb_size, message_hidden_size, 
            message_context_size, message_dropout_p, 
            message_proj_nonlinearity, sen_to_message_rnn):
        super(SentenceToMessage, self).__init__()

        self.sentence_emb_size = sentence_emb_size
        self.message_hidden_size = message_hidden_size
        self.message_context_size = message_context_size
        
        self.dropout = nn.Dropout(p=message_dropout_p)
        self.message_context = nn.Parameter(torch.FloatTensor(message_context_size, 1).uniform_(-0.1, 0.1))

        self.message_encoder = sen_to_message_rnn(sentence_emb_size, message_hidden_size, batch_first=True,
                bidirectional=True)
        self.message_proj = nn.Linear(message_hidden_size * 2, message_context_size)
        self.message_proj_nonlinearity = message_proj_nonlinearity()

        self.softmax = nn.Softmax(dim=1)

    def is_cuda(self):
        return "cuda" in str(type(self.message_encoder.bias_hh_l0.data))

    def forward(self, sentence_tensors):
        """
        sentence_tensors: Variable with 2 or 3 dimension
            example: 
                input: 
                    [[1,2,3,4], [2,3,4,5]]
                outputs:
                    (
                        [[5,6,7,8]],
                        [[0.2, 0.8]]
                    )
        """
        # check the dim of input sentence_tensors
        if sentence_tensors.dim() == 2:
            # add one dim for batch
            sentence_tensors = torch.unsqueeze(sentence_tensors, 0)
        
        # apply dropout
        sentence_tensors = self.dropout(sentence_tensors)
        output, _ = self.message_encoder(sentence_tensors)
        
        
        # define messages place holder
        if self.is_cuda():
            messages = Variable(torch.zeros(output.size(0), self.message_hidden_size*2).cuda())
            message_attentions = Variable(torch.zeros(output.size(0), output.size(1)).cuda())
        else:
            messages = Variable(torch.zeros(output.size(0), self.message_hidden_size*2))
            message_attentions = Variable(torch.zeros(output.size(0), output.size(1)))

        for i in range(output.size(0)):
            sentence_tensors = output[i]
            #messages[i,:] = torch.max(sentence_tensors, dim=0)[0]
            proj = self.message_proj(sentence_tensors)
            proj = self.message_proj_nonlinearity(proj)
            attention = torch.mm(proj, self.message_context)
            attention = self.softmax(attention.t())
            attention = attention.view(-1)
            messages[i,:] = sentence_tensors.transpose(0,1).mv(attention)
            message_attentions[i, :] = attention
        return messages, message_attentions

class WordToSentence(nn.Module):
    def __init__(self, vocab_size, word_emb_size, pretrained_word_embedding, 
            freeze_word_emb, sentence_hidden_size, sentence_context_size, 
            sentence_dropout_p, sentence_proj_nonlinearity, word_to_sen_rnn):
        super(WordToSentence, self).__init__()


        self.word_emb_size = word_emb_size
        
        self.word_embedding = nn.Embedding(vocab_size, word_emb_size)
        if pretrained_word_embedding is not None:
            self.word_embedding.weight.data = pretrained_word_embedding
        
        if freeze_word_emb:
            for p in self.word_embedding.parameters():
                p.require_grad = False

        self.dropout = nn.Dropout(p=sentence_dropout_p)

        self.sentence_hidden_size = sentence_hidden_size
        self.sentence_context_size = sentence_context_size

        self.sentence_encoder = word_to_sen_rnn(word_emb_size, sentence_hidden_size, 
                bidirectional=True, batch_first=True)

        self.sentence_context = nn.Parameter(torch.Tensor(sentence_context_size, 1).uniform_(-0.1, 0.1))

        self.sentence_proj = nn.Linear(sentence_hidden_size * 2, sentence_context_size)
        self.sentence_proj_nonlinearity = sentence_proj_nonlinearity()

        # attention weight 
        self.softmax = nn.Softmax(dim=1)

    def is_cuda(self):
        return "cuda" in str(type(self.sentence_encoder.bias_hh_l0.data))

    def forward(self, var_sentence_batch, var_sentence_lens, batch_max_len):
        """
        sentence_batch: np.array with list of sentence, which is a list of word ids
        sentence_len_batch: list of sentence length
            example:
                input: 
                    [[1,2,3], [1,2,3,4]]
                output:
                    (
                        [[3,4], [5,6]],
                        [[0.25, 0.25, 0.5, 0], [0.25,0.35, 0.15, 0.25]]
                    )
        """
        # padding the sentence
        sen_lens = var_sentence_lens.cpu().data.numpy()
        padded_sentence_batch = var_sentence_batch
        sorted_sen_lens, sorted_ids = torch.sort(var_sentence_lens, descending=True)

        # sort the padded sentence
        unsorted_ids = torch.sort(sorted_ids)[1]
        
        padded_sentence_batch = padded_sentence_batch[sorted_ids]
        # word embedding lookup
        sen_embedding = self.word_embedding(padded_sentence_batch) 
        
        # pack the padded sequence
        packed = pack_padded_sequence(sen_embedding, sorted_sen_lens.cpu().data.numpy(), batch_first=True)
        output, _ = self.sentence_encoder(packed)
        
        # unpack the paddes sequnece
        output, _ = pad_packed_sequence(output, batch_first=True)

        # reverse the output
        output = output[unsorted_ids, :, :]
       
        output = self.dropout(output)
        if self.is_cuda():
            sentence_tensors = Variable(torch.zeros((output.size(0), output.size(2))).cuda())
            sentence_attentions = Variable(torch.zeros(output.size(0), batch_max_len).cuda())
        else:
            sentence_tensors = Variable(torch.zeros((output.size(0), output.size(2))))
            sentence_attentions = Variable(torch.zeros(output.size(0), batch_max_len))
       

        # project the output to context space
        for i in range(output.size(0)):
            #compute the attention for each sentences
            word_tensors = output[i,:sen_lens[i],:]
        
            proj = self.sentence_proj(word_tensors)
            proj = self.sentence_proj_nonlinearity(proj)

            # compute the attention
            attention = torch.mm(proj, self.sentence_context)
            attention = self.softmax(attention.t()) # make sure the softmax apply on the last dim
            attention = attention.view(-1)

            sentence_tensors[i, :] = word_tensors.transpose(0, 1).mv(attention)
            sentence_attentions[i, :sen_lens[i]] = attention
        sentence_tensors = self.dropout(sentence_tensors)
        return sentence_tensors, sentence_attentions

def pad_sentence(sentence_batch):
    """
    sentence_batch: list of list of word ids
    """
    sen_lens = np.array([len(s) for s in sentence_batch])
    max_len = max(sen_lens)
    
    # we assume zero is the padding word id
    new_sentence_batch = np.zeros((len(sen_lens), max_len))
    for i, sen in enumerate(sentence_batch):
        new_sentence_batch[i, :sen_lens[i]] = np.array(sen)

    new_sentence_batch = new_sentence_batch.astype(int)
    return new_sentence_batch, sen_lens
    
if __name__ == "__main__":
    pass

