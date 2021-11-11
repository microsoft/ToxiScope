import gzip
import os
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
#from torchtext.utils import download_from_url
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForTokenClassification, BertConfig
from transformers.modeling_roberta import RobertaClassificationHead
from torch.nn import CrossEntropyLoss, MSELoss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=5, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            loss_fct = CrossEntropyLoss()
            BCE_loss = loss_fct(inputs, targets)
            #BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, labels, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(labels, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class BERT_Classification(nn.Module):
    def __init__(self,
                 ):
        super(BERT_Classification, self).__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        #self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2) 
        #self.apply(self.init_weights)
        print("BERT loaded!!!!")
        # tokens_tensor = []
        # segments_tensor = []

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.model(input_ids=flat_input_ids, attention_mask=flat_attention_mask) #, attention_mask=attention_mask) #, token_type_ids=flat_token_type_ids, position_ids=flat_position_ids)
        sequence_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        logits = F.softmax(self.classifier(sequence_output))
        #reshaped_logits = logits.view(-1, num_choices)
        #outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            #loss_fct = FocalLoss()
            #print(logits)
            loss = loss_fct(logits, labels)
            print("loss:", loss)
            outputs = (logits,) + outputs[2:]
        return ((loss,) + outputs) if loss is not None else outputs   # (loss), reshaped_logits, (hidden_states), (attentions)


class BERT_MLP(nn.Module):
    def __init__(self,
                 ):
        super(BERT_MLP, self).__init__()
        config = BertConfig.from_pretrained("bert-large-uncased")
        self.weights_add = Variable(torch.Tensor(config.hidden_size), requires_grad=True).cuda()
        #self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model = BertModel(config)
        for param in self.model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.oneLayer = nn.Linear(config.hidden_size, 128)
        self.classifier = nn.Linear(128, 2) 
        #self.apply(self.init_weights)
        self.weights_add.data.uniform_(-2.0, 2.0)
        print("BERT loaded!!!!")
        # tokens_tensor = []
        # segments_tensor = []

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
        num_choices = input_ids.shape[1]
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        outputs = self.model(input_ids=flat_input_ids, attention_mask=flat_attention_mask) #, attention_mask=attention_mask) #, token_type_ids=flat_token_type_ids, position_ids=flat_position_ids)
        sequence_output = outputs[1]
        mlp_output = self.oneLayer(nn.Tanh()(self.weights_add*sequence_output))
        sequence_output = self.dropout(mlp_output)
        logits = self.classifier(sequence_output)
        #reshaped_logits = logits.view(-1, num_choices)
        #outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss(size_average=False)
            loss = loss_fct(logits, labels)
            #outputs = (logits,) + outputs[2:]
            print(loss.item())
        #return ((loss,))
            if loss is not None:
                return logits, ((loss,)+outputs)
            else:
                return logits, outputs
        #return logits, ((loss,) + outputs) if loss is not None else logits, outputs   # (loss), reshaped_logits, (hidden_states), (attentions)
