from pytorch_pretrained_bert.tokenization import BertTokenizer, WordpieceTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining, BertPreTrainedModel, BertModel, BertConfig, BertForMaskedLM, BertForSequenceClassification
from transformers import BertModel, BertConfig
from pathlib import Path
from models import BertForToxicChoice
import torch
import re
import utils
from torch import Tensor
from torch.nn import BCEWithLogitsLoss
#from fastai.text import Tokenizer, Vocab
import pandas as pd
import collections
import os
import pdb
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import logging
from utils import InputExample, convert_examples_to_features
from sklearn.metrics import *
#import apex
from sklearn.model_selection import train_test_split
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from sklearn.metrics import roc_curve, auc


from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bert.optimization import BertAdam
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH=Path('./toxicity-detector/data/training/')
DATA_PATH.mkdir(exist_ok=True)

PATH=Path('./toxicity-detector/data/training')
PATH.mkdir(exist_ok=True)

CLASS_DATA_PATH=PATH
CLASS_DATA_PATH.mkdir(exist_ok=True)

model_state_dict = None
best_f1_so_far = 0
# BERT_PRETRAINED_PATH = Path('../trained_model/')
BERT_PRETRAINED_PATH = Path('./bert/pretrained-weights/uncased_L-12_H-768_A-12/')
# BERT_PRETRAINED_PATH = Path('../../complaints/bert/pretrained-weights/cased_L-12_H-768_A-12/')
# BERT_PRETRAINED_PATH = Path('../../complaints/bert/pretrained-weights/uncased_L-24_H-1024_A-16/')


# BERT_FINETUNED_WEIGHTS = Path('../trained_model/toxic_comments')

PYTORCH_PRETRAINED_BERT_CACHE = BERT_PRETRAINED_PATH/'cache_github_mlp/'
PYTORCH_PRETRAINED_BERT_CACHE.mkdir(parents=True, exist_ok=True)

# output_model_file = os.path.join(BERT_FINETUNED_WEIGHTS, "pytorch_model.bin")

# Load a trained model that you have fine-tuned
# model_state_dict = torch.load(output_model_file)

args = {
    "train_size": -1,
    "val_size": -1,
    "full_data_dir": DATA_PATH,
    "data_dir": PATH,
    "task_name": "toxic_label",
    "bert_model": 'bert-base-uncased',
    "output_dir": CLASS_DATA_PATH/'output',
    "max_seq_length": 512,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "learning_rate": 3e-5,
    "num_train_epochs": 5,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 3,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config, num_labels=2):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss
        else:
            return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

'''
class TextProcessor(DataProcessor):

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None

    def get_train_examples(self, data_dir, size=-1):
        filename = 'train_med.tsv'
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
        if size == -1:
            data_df = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df, "train")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df.sample(size), "train")

    def get_dev_examples(self, data_dir, size=-1):
        """See base class."""
        filename = 'val.tsv'
        if size == -1:
            data_df = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df, "dev")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
            #             data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
            return self._create_examples(data_df.sample(size), "dev")

    def get_test_examples(self, data_dir, data_file_name, size=-1):
        data_df = pd.read_csv(os.path.join(data_dir, data_file_name), sep='\t')
        #         data_df['comment_text'] = data_df['comment_text'].apply(cleanHtml)
        if size == -1:
            return self._create_examples(data_df, "test")
        else:
            return self._create_examples(data_df.sample(size), "test")

    def get_labels(self):
        """See base class."""
        if self.labels == None:
            self.labels = list(pd.read_csv(os.path.join(self.data_dir, "classes.txt"), header=None)[0].values)
        return self.labels

    def _create_examples(self, df, set_type, labels_available=True):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, row) in enumerate(df.values):
            guid = row[0]
            text_a = row[7]
            if row[3]:
                text_b = row[3]
            else:
                text_b = None
            if labels_available:
                labels = row[2]
            else:
                labels = []
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, labels=labels))
        return examples
'''
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b and isinstance(example.text_b, str):
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        labels_ids = []
        #for label in example.labels:
        labels_ids.append(int(example.label))

        if ex_index < 0:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (example.label, labels_ids))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=labels_ids))
        return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def accuracy_thresh(y_pred:Tensor, y_true:Tensor, thresh:float=0.5, sigmoid:bool=True):
    "Compute accuracy when `y_pred` and `y_true` are the same size."
    import numpy
    if sigmoid: y_pred = y_pred.sigmoid()
#     return ((y_pred>thresh)==y_true.byte()).float().mean().item()
    y = numpy.amax(y_pred.detach().cpu().numpy(), axis=1)
    count = 0
    for i in range(len(y)):
        if y[i] > thresh:
            if y_true[i] == 1:
                count +=1
        else:
            if y_true[i] == 0:
                count += 1
    return count/len(y)


def fbeta(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=2, eps:float=1e-9, sigmoid:bool=True):
    "Computes the f_beta between `preds` and `targets`"
    beta2 = beta ** 2
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred>thresh).float()
    y_true = y_true.float()
    TP = (y_pred*y_true).sum(dim=1)
    prec = TP/(y_pred.sum(dim=1)+eps)
    rec = TP/(y_true.sum(dim=1)+eps)
    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)
    return res.mean().item()


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

processors = {
    "toxic_label": utils.BinaryProcessor
}

# Setup GPU parameters

if args["local_rank"] == -1 or args["no_cuda"]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    n_gpu = torch.cuda.device_count()
#     n_gpu = 1
else:
    torch.cuda.set_device(args['local_rank'])
    device = torch.device("cuda", args['local_rank'])
    n_gpu = 1
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.distributed.init_process_group(backend='nccl')
logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args['local_rank'] != -1), args['fp16']))

args['train_batch_size'] = int(args['train_batch_size'] / args['gradient_accumulation_steps'])

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
if n_gpu > 0:
    torch.cuda.manual_seed_all(args['seed'])

task_name = args['task_name'].lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = utils.BinaryProcessor()
label_list = processor.get_labels()
num_labels = len(label_list)
print("Number of labels:", num_labels)
tokenizer = BertTokenizer.from_pretrained(args['bert_model'], do_lower_case=args['do_lower_case'])

train_examples = None
num_train_steps = None
if args['do_train']:
    train_examples = processor.get_train_examples(args['full_data_dir'])
#     train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
    num_train_steps = int(
        len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps'] * args['num_train_epochs'])
    config = BertConfig(
        #args["bert_model"],
        #num_labels=2, # if args.model_type in ["roberta_mc"] else num_labels,
        #finetuning_task=args["task_name"],
    )

    # Prepare model
    def get_model():
        #     pdb.set_trace()
        #if model_state_dict:
            #model = BertForMultiLabelSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels,
        #     model = BertForSequenceClassification(args['bert_model'], num_labels=num_labels,state_dict=model_state_dict) #                                                              state_dict=model_state_dict)
        #else:
        #model = BertForSequenceClassification.from_pretrained(args['bert_model'], num_labels=num_labels)
        model = BertForToxicChoice.from_pretrained(args['bert_model'], num_labels=num_labels)
       
        return model


    model = get_model()

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args['local_rank'] != -1:
        t_total = t_total // torch.distributed.get_world_size()

    optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args['learning_rate'],
                             warmup=args['warmup_proportion'],
                             t_total=t_total)

# Eval Fn
eval_examples = processor.get_dev_examples(args['data_dir']) #, size=args['val_size'])
def eval():
    args['output_dir'].mkdir(exist_ok=True)

    eval_features = utils.convert_examples_to_features(
        eval_examples, label_list, args['max_seq_length'], tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args['eval_batch_size'])
    all_logits = None
    all_labels = None
    all_logits_sigmoid = None
    y_pred, y_true = [], []
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)
        #         logits = logits.detach().cpu().numpy()
        #         label_ids = label_ids.to('cpu').numpy()
        #         tmp_eval_accuracy = accuracy(logits, label_ids)
        import numpy as np
        logits = logits[0]
        logits_arg = np.argmax(logits.detach().cpu().numpy(), axis=1)
        tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
        if all_logits_sigmoid is None:
            all_logits_sigmoid = logits.sigmoid().detach().cpu().numpy()
        else:
            all_logits_sigmoid = np.concatenate((all_logits_sigmoid, logits.sigmoid().detach().cpu().numpy()), axis=0)


        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)
        y_true.append(all_labels.ravel())
        y_pred.append(all_logits.ravel())
        eval_loss += tmp_eval_loss[0].mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    #     ROC-AUC calcualation
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    ''' 
    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    '''
    # Compute micro-average ROC curve and ROC area
    import numpy as np
    y_pred = np.argmax(all_logits_sigmoid, axis=1)
    auc = roc_auc_score(all_labels, y_pred)
    gold = all_labels
    y_label = np.amax(all_logits_sigmoid, axis=1)
    print('Eval ROC AUC: %.3f' % auc)
    #print('Best Model Predicted: Epoch {0}  acc {1}'.format(epoch, acc))
    labels = []
    print("***********************")
    if True:
        for i in range(len(y_label)):
            if y_label[i] >= 0.5:
                labels.append(1)
            else:
                labels.append(0)
        pos = []
        neg = []
        neg_t = []
        pos_t = []
        labels = y_pred
        for idx, i in enumerate(gold):
            if i == 1:
                pos_t.append(1)
                pos.append(labels[idx])
            else:
                neg_t.append(0)
                neg.append(labels[idx])

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import f1_score, classification_report
        f1 = f1_score(gold, labels, average='macro')
        print("Accuracy (neg) class:", accuracy_score(neg_t, neg))
        print("Accuracy (pos) class:", accuracy_score(pos_t, pos))
        print("Overall accuracy:", accuracy_score(gold, labels))
        print("F1 macro score: ", f1_score(gold, labels, average='macro'))
        print("F1 micro score: ", f1_score(gold, labels, average='micro'))
        print("Pos F1 macro score: ", f1_score(pos_t, pos, average="macro"))
        print("Neg F1 macro score: ", f1_score(neg_t, neg, average="macro"))
        print(classification_report(gold, labels, digits=4))
    global best_f1_so_far
    if f1 > best_f1_so_far:
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "finetuned_pytorch_model_val.bin")
        torch.save(model_to_save.state_dict(), output_model_file)
        best_f1_so_far = f1
        
    '''
    with open(os.path.join("./", "predictions.txt"), 'w') as pf:
        for p_l, t_l in zip(gold, labels):
            pf.write("%s\t%d\n" % ('\t'.join(["%0.4f" % p for p in p_l]), t_l))
    '''
    '''
    import numpy as np
    y_label1 = np.argmax(y_pred, axis=1)
    print(y_label1, y_pred)
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true, y_label1)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              #               'loss': tr_loss/nb_tr_steps,
              'roc_auc': roc_auc}

    output_eval_file = os.path.join(args['output_dir'], "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    '''
train_features = utils.convert_examples_to_features(
    train_examples, label_list, args['max_seq_length'], tokenizer)

logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", args['train_batch_size'])
logger.info("  Num steps = %d", num_train_steps)
all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
if args['local_rank'] == -1:
    train_sampler = RandomSampler(train_data)
else:
    train_sampler = DistributedSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])

from tqdm import tqdm_notebook as tqdm
def fit(num_epocs=args['num_train_epochs']):
    global_step = 0
    model.to(device)
    model.train()
    for i_ in (range(int(num_epocs))):

        tr_loss, best_loss = 0, 1000.0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate((train_dataloader)):

            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = model(input_ids, segment_ids, input_mask, labels=label_ids)
            #loss = loss.mean()
            loss = outputs[0]
            print(loss)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                optimizer.backward(loss)
            else:
                loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
    #             scheduler.batch_step()
                # modify learning rate with special warm up BERT uses
                lr_this_step = args['learning_rate'] * warmup_linear(global_step/t_total, args['warmup_proportion'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        logger.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
        if tr_loss < best_loss:
        # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "finetuned_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            best_loss = tr_loss

        logger.info('Eval after epoc {}'.format(i_+1))
        eval()

#model.module.unfreeze_bert_encoder()
fit()
'''
# Save a trained model
model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "finetuned_pytorch_model.bin")
torch.save(model_to_save.state_dict(), output_model_file)
'''
# Load a trained model that you have fine-tuned
output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "finetuned_pytorch_model.bin")
model_state_dict = torch.load(output_model_file)
model = BertForToxicChoice.from_pretrained(args['bert_model'], num_labels = num_labels, state_dict=model_state_dict)
model.to(device)

model_val = BertForToxicChoice.from_pretrained(args['bert_model'], num_labels = num_labels, state_dict=model_state_dict)
model_val.to(device)


eval()


def predict(model, path, test_filename='test.tsv'):
    predict_processor = utils.BinaryProcessor()
    test_examples = predict_processor.get_test_examples(path) #, test_filename, size=-1)

    # Hold input data for returning it
    input_data = [{'id': input_example.guid, 'comment_text': input_example.text_a} for input_example in test_examples]

    test_features = utils.convert_examples_to_features(
        test_examples, label_list, args['max_seq_length'], tokenizer)

    logger.info("***** Running prediction *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args['eval_batch_size'])

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
     
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

    # Run prediction for full data
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args['eval_batch_size'])

    all_logits = None

    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for step, batch in enumerate((test_dataloader)):
        input_ids, input_mask, segment_ids = batch
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask)
            print(logits)
            logits = logits[0]
            #logits = logits.sigmoid()
            print(logits)
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
        else:
            all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)
        
        '''
        if all_labels is None:
            all_labels = label_ids.detach().cpu().numpy()
        else:
            all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)
        '''
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
    

    return pd.merge(pd.DataFrame(input_data), pd.DataFrame(all_logits, columns=label_list), left_index=True,
                    right_index=True), all_logits


result, y_sigmoid = predict(model, DATA_PATH)
print("Predictions for the best trained model")
import numpy as np
if True:
    y_pred = np.argmax(y_sigmoid, axis=1)
    df = pd.read_csv(os.path.join(DATA_PATH,"test.tsv"), sep='\t')
    all_labels = np.asarray(df['label'].tolist())
    auc = roc_auc_score(all_labels, y_pred)
    gold = all_labels
    y_label = np.amax(y_sigmoid, axis=1)
    print('Test ROC AUC: %.3f' % auc)
    #print('Best Model Predicted: Epoch {0}  acc {1}'.format(epoch, acc))
    labels = []
    print("***********************")
    if True:
        for i in range(len(y_label)):
            if y_label[i] >= 0.5:
                labels.append(1)
            else:
                labels.append(0)
        pos = []
        neg = []
        neg_t = []
        pos_t = []
        labels = y_pred
        for idx, i in enumerate(gold):
            if i == 1:
                pos_t.append(1)
                pos.append(labels[idx])
            else:
                neg_t.append(0)
                neg.append(labels[idx])

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import f1_score, classification_report
        print("Accuracy (neg) class:", accuracy_score(neg_t, neg))
        print("Accuracy (pos) class:", accuracy_score(pos_t, pos))
        print("Overall accuracy:", accuracy_score(gold, labels))
        print("F1 macro score: ", f1_score(gold, labels, average='macro'))
        print("F1 micro score: ", f1_score(gold, labels, average='micro'))
        print("Pos F1 macro score: ", f1_score(pos_t, pos, average="macro"))
        print("Neg F1 macro score: ", f1_score(neg_t, neg, average="macro"))
        print(classification_report(gold, labels, digits=4))
    '''
    with open(os.path.join("./", "test_predictions.txt"), 'w') as pf:
        for p_l, t_l in zip(gold, labels):
            pf.write("%s\t%d\n" % ('\t'.join(["%0.4f" % p for p in p_l]), t_l))
    '''
result.to_csv(os.path.join(PYTORCH_PRETRAINED_BERT_CACHE,"./test_predictions.tsv"), sep='\t', index=False)

output_model_file = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, "finetuned_pytorch_model_val.bin")
model_state_dict = torch.load(output_model_file)
model_val = BertForToxicChoice.from_pretrained(args['bert_model'], num_labels = num_labels, state_dict=model_state_dict)
model_val.to(device)

result, y_sigmoid = predict(model_val, DATA_PATH)
print("Predictions for the best validation model")
import numpy as np
if True:
    y_pred = np.argmax(y_sigmoid, axis=1)
    df = pd.read_csv(os.path.join(DATA_PATH,"test.tsv"), sep='\t')
    all_labels = np.asarray(df['label'].tolist())
    auc = roc_auc_score(all_labels, y_pred)
    gold = all_labels
    y_label = np.amax(y_sigmoid, axis=1)
    print('Test ROC AUC: %.3f' % auc)
    #print('Best Model Predicted: Epoch {0}  acc {1}'.format(epoch, acc))
    labels = []
    print("***********************")
    if True:
        for i in range(len(y_label)):
            if y_label[i] >= 0.5:
                labels.append(1)
            else:
                labels.append(0)
        pos = []
        neg = []
        neg_t = []
        pos_t = []
        labels = y_pred
        for idx, i in enumerate(gold):
            if i == 1:
                pos_t.append(1)
                pos.append(labels[idx])
            else:
                neg_t.append(0)
                neg.append(labels[idx])

        from sklearn.metrics import accuracy_score
        from sklearn.metrics import f1_score, classification_report
        print("Accuracy (neg) class:", accuracy_score(neg_t, neg))
        print("Accuracy (pos) class:", accuracy_score(pos_t, pos))
        print("Overall accuracy:", accuracy_score(gold, labels))
        print("F1 macro score: ", f1_score(gold, labels, average='macro'))
        print("F1 micro score: ", f1_score(gold, labels, average='micro'))
        print("Pos F1 macro score: ", f1_score(pos_t, pos, average="macro"))
        print("Neg F1 macro score: ", f1_score(neg_t, neg, average="macro"))
        print(classification_report(gold, labels, digits=4))
    '''
    with open(os.path.join("./", "test_predictions.txt"), 'w') as pf:
        for p_l, t_l in zip(gold, labels):
            pf.write("%s\t%d\n" % ('\t'.join(["%0.4f" % p for p in p_l]), t_l))
    '''
result.to_csv(os.path.join(PYTORCH_PRETRAINED_BERT_CACHE,"./test_predictions_val.tsv"), sep='\t', index=False)


