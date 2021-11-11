#Modified HuggingFace transformers

from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import json
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from sklearn.metrics import f1_score, roc_auc_score
from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup

from utils import (convert_examples_to_features,
                   output_modes, processors)
from model import BERT_Classification, BERT_MLP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
'''
args = {
    'data_dir': 'data/avocado/',
    'model_type':  'bert',
    'model_name': 'bert-base-uncased',
    'task_name': 'binary',
    'output_dir': 'outputs_noupsample/',
    'cache_dir': 'cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'output_mode': 'classification',
    'train_batch_size': 64,
    'eval_batch_size': 64,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 100,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 1000,
    'evaluate_during_training': True,
    'save_steps': 2000,
    'eval_all_checkpoints': True,

    'overwrite_output_dir': True,
    'reprocess_input_data': True,
    'notes': 'For exploration'
}
'''
args = dict()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('args.json', 'r') as f:
    args = json.load(f)

MODEL_CLASSES = {
    'bert': (BertConfig, BERT_MLP, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]
config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task=args['task_name'])
tokenizer = tokenizer_class.from_pretrained(args['model_name'])
#model = model_class.from_pretrained(args['model_name'])
#model = BERT_Classification()
model = BERT_MLP()
model.to(device)

task = args['task_name']

if task in processors.keys() and task in output_modes.keys():
    processor = processors[task]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
else:
    raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')

def load_and_cache_examples(task, tokenizer, evaluate=False, test=False):
    processor = processors[task]()
    output_mode = args['output_mode']
    if test:
        mode = 'test'
    else:
        mode = 'val' if evaluate else 'train'
    cached_features_file = os.path.join(args['data_dir'],
                                        f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")

    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)

    else:
        logger.info("Creating features from dataset file at %s", args['data_dir'])
        label_list = processor.get_labels()
        if test:
            examples = processor.get_test_examples(args['data_dir'])
        else: 
            examples = processor.get_dev_examples(args['data_dir']) if evaluate else processor.get_train_examples(
            args['data_dir'])

        if __name__ == "__main__":
            features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer,
                                                    output_mode,
                                                    cls_token_at_end=bool(args['model_type'] in ['xlnet']),
                                                    # xlnet has a cls token at the end
                                                    cls_token=tokenizer.cls_token,
                                                    cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
                                                    sep_token=tokenizer.sep_token,
                                                    sep_token_extra=bool(args['model_type'] in ['roberta']),
                                                    # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                    pad_on_left=bool(args['model_type'] in ['xlnet']),
                                                    # pad on the left for xlnet
                                                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                    pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def train(train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()
    #classcount = np.bincount(train_dataset.label).tolist()
    classcount = [100, 2]
    weights = 1. / torch.tensor(classcount, dtype=torch.float)
    ''' 
    train_sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=args['train_batch_size'],
        replacement=True)
    '''
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])

    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    warmup_steps = math.ceil(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps = -1)

    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    min_val_loss = 5
    epoch = args['num_train_epochs']
    early_stop = False
    epochs_to_improve, n_epochs_stop = 0, 2
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            probs, outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            #print("\r%f" % loss, end='')

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])

            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            #print("training loss: ", tr_loss/global_step)
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                   # if not os.path.exists(output_dir):
                   #     os.makedirs(output_dir)
                   # model_to_save = model.module if hasattr(model,
                    #                                        'module') else model  # Take care of distributed/parallel training
                   # torch.save(model.state_dict(),os.path.join(output_dir,"model.pt"))
                    #model_to_save.save_pretrained(output_dir)
                   # logger.info("Saving model checkpoint to %s", output_dir)

                    if args[
                        'evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(model, tokenizer, eval_output_dir=output_dir)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args['logging_steps'], global_step)
                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model,
                                                            'module') else model  # Take care of distributed/parallel training
                    torch.save(model.state_dict(),output_dir+"model.pt")
                    #model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                ''' 
                if loss < min_val_loss:
                    epochs_no_improve = 0
                    min_val_loss = loss
		else:
		    epochs_no_improve += 1
		if epoch > 5 and epochs_no_improve == n_epochs_stop:
		    print('Early stopping!' )
		    early_stop = True
		    break
		else:
		    continue
		break
		
	    if early_stop:
		print("Stopped")
		break
		'''
                print("training loss: ", tr_loss/global_step)

    return global_step, tr_loss / global_step


from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr


def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args['data_dir'])
    print("printing predictions:")
    for (i,v) in zip(examples, preds):
        print(i.text_a, i.label, v)
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    return wrong


def get_eval_report(labels, preds):
    preds_l = (preds > 0.5)
    wrong = get_mismatched(labels, preds)
    examples = processor.get_dev_examples(args['data_dir'])
    parent = []
    text = []
    #body = []
    for (ex, val) in zip(examples, preds):
        parent.append(ex.text_b)
        text.append(ex.text_a)
    df = pd.DataFrame(data={'text':text, 'parent':parent, 'probs':preds})
    df.to_csv(os.path.join(args['output_dir'],"test.tsv"),sep='\t')    
    #mcc = matthews_corrcoef(labels, preds)
    f1 = f1_score(labels, preds_l, average='weighted')
    auc = roc_auc_score(labels, preds )#, average='weighted')
    tn, fp, fn, tp = confusion_matrix(labels, preds_l).ravel()
    logger.info
    return {
              # "mcc": mcc,
               "tp": tp,
               "tn": tn,
               "fp": fp,
               "fn": fn,
               "f1": f1,
               "roc_auc": auc
           }, get_mismatched(labels, preds_l)


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)


def evaluate(model, tokenizer, prefix="", test=False, eval_output_dir=args['output_dir']):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    #eval_output_dir = args['output_dir']

    results = {}
    EVAL_TASK = args['task_name']
    if test:
        eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, test=True)
    else:
        eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, evaluate=True)
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,
                      # XLM don't use segment_ids
                      'labels': batch[3]}
            probs, outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        print("logits:")
        pred = probs.data.max(1)[1] # torch.Tensor
        print(probs)
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs['labels'].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args['output_mode'] == "classification":
        #y_pred = np.amax(probs, axis=1)
        pred_label = np.amax(probs.detach().cpu().numpy(), axis=1)
    elif args['output_mode'] == "regression":
        preds = np.squeeze(preds)
    result, wrong = compute_metrics(EVAL_TASK, preds, out_label_ids)
    _, wrong_logits = compute_metrics(EVAL_TASK, pred_label, out_label_ids)
    results.update(result)
    print("Test mode:", test) 
    if test:
        print("Test results")
        output_eval_file = os.path.join(eval_output_dir, "test_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Test results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    else:
        print("Evaluation")
        output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            writer.write(" %s = %s\n" % ("val_loss", str(eval_loss)))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results, wrong
if args['do_train']:
    train_dataset = load_and_cache_examples(task, tokenizer)
    global_step, tr_loss = train(train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

if args['do_train']:
    if not os.path.exists(args['output_dir']):
        os.makedirs(args['output_dir'])
    logger.info("Saving model checkpoint to %s", args['output_dir'])

    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    #model_to_save.save_pretrained(args['output_dir'])
    #torch.save(model.state_dict(), args['output_dir']+ 'pytorch_model.pt')
    tokenizer.save_pretrained(args['output_dir'])
    torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))

results = {}
if args['do_eval']:
    checkpoints = [args['output_dir']]
    logger.info("Weights name: %s", WEIGHTS_NAME)
    if args['eval_all_checkpoints']:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' +  'model.pt', recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        ckpt = torch.load(os.path.join(checkpoint, "model.pt"))
        model.load_state_dict(ckpt)
        model.to(device)
        result, wrong_preds = evaluate(model, tokenizer, prefix=global_step, test=True,eval_output_dir=checkpoint)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)
'''ckpt = torch.load(args['output_dir']+"pytorch_model.pt")
model.load_state_dict(ckpt)
model.to(device)
result, wrong_preds = evaluate(model, tokenizer, prefix=global_step, test=True)
result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
results.update(result)

print(results)
'''
