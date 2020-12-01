# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#"""PyTorch BERT model. """


import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
#from .activations import ACT2FN
from pytorch_transformers import BertPreTrainedModel,BertConfig
'''from file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
'''
#from .utils import logging


#logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]

from pytorch_transformers import BertPreTrainedModel, BertConfig, \
    ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP, BertModel, BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from pytorch_transformers.modeling_bert import BertForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import NLLLoss, LogSoftmax
import torch
from torch.functional import F

class BertForToxicChoice(BertPreTrainedModel):
     config_class = BertConfig
     pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
     base_model_prefix = "bert-base-uncased"

     def __init__(self, config):
         super(BertForToxicChoice, self).__init__(config)

         self.bert = BertModel(config)
         
         modules = [self.bert.embeddings, *self.bert.encoder.layer[:9]] #Replace 7 by what you want
         for module in modules:
             for param in module.parameters():
                 param.requires_grad = False
        
         self.num_labels = 2
         self.tanh = nn.Tanh()
         self.linear = nn.Linear(config.hidden_size, 128)
         self.classifier = nn.Linear(config.hidden_size, 2)
         #self.classifier = BertForSequenceClassification(config)
         self.dropout = nn.Dropout(p=0.1)
         self.init_weights()
     
     def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None):
    
    #    """
    #    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
    #        Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
    #        config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
    #        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    #    """
         #return_dict = return_dict if return_dict is not None else self.config.use_return_dict
         flat_input_ids = input_ids.view(-1, input_ids.size(-1))
         flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
         flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
         flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
         outputs = self.bert(flat_input_ids, attention_mask=flat_attention_mask)
         '''
         outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
         )
         '''
         pooled_output = outputs[1]
         #pooled_output = self.dropout(pooled_output)
         logits = self.classifier(self.dropout(self.tanh(pooled_output)))
         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

         if labels is not None:
             if self.num_labels == 1:
                #  We are doing regression
                 loss_fct = MSELoss()
                 loss = loss_fct(logits.view(-1), labels.view(-1))
             else:
                 loss_fct = CrossEntropyLoss()
                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
             outputs = ((loss,) + outputs)

         return outputs  # (loss), logits, (hidden_states), (attentions)
         '''
         loss = 0.0
         if labels is not None:
             if self.num_labels == 1:
                 #  We are doing regression
                 loss_fct = MSELoss()
                 loss = loss_fct(logits.view(-1), labels.view(-1))
             else:
                 loss_fct = CrossEntropyLoss()
                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
             print(loss)
         return loss
         if not return_dict:
             output = (logits,) + outputs[2:]
             #return loss
             return ((loss,) + output) if loss is not None else output
         '''

'''
class BertForToxicClassification(BertPreTrainedModel):
    #tokenizer_class=_TOKENIZER_FOR_DOC,
    checkpoint="bert-base-uncased",
    config_class=BertConfig,

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2 

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    #@add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    #@add_code_sample_docstrings(
    #    tokenizer_class=_TOKENIZER_FOR_DOC,
    #    checkpoint="bert-base-uncased",
    #    output_type=SequenceClassifierOutput,
    #    config_class=_CONFIG_FOR_DOC,
    #)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
    #    """
    #    labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
    #        Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
    #        config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
    #        If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    #    """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(torch.nn.Tanh(pooled_output))

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
'''
