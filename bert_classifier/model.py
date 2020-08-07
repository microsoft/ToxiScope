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
        logits = self.classifier(sequence_output)
        #reshaped_logits = logits.view(-1, num_choices)
        #outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            outputs = (logits,) + outputs[2:]
        return ((loss,) + outputs) if loss is not None else outputs   # (loss), reshaped_logits, (hidden_states), (attentions)
