#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import torch.optim as optim
import math
import numpy as np
import torch
import json

class BaseTrainer():
    def __init__(self, model, loss, metrics, resume, config):
        self.config = config
        self.model = model
        self.loss = loss
        
        self.epochs = config["epochs"]
        self.optimizer = getattr(optim, config["optimizer_type"])(model.parameters(),
                **config["optimizer_params"])

        
        self.monitor = config["monitor"]
        self.monitor_mode = config["monitor_mode"]
        self.monitor_best = math.inf if self.monitor_mode == "min" else -math.inf

        self.start_epoch = 1
        self.name = config["name"]
        self.checkpoint_dir = os.path.join(config["saved_folder"], self.name)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if resume:
            self._resume_checkpoint(resume)

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            log = {"epoch": epoch}
            for k,v in result.items():
                if k == "metrics":
                    for m, v in zip(self.metrics, v):
                        log[m] = v
                elif k == "val_metrics":
                    for m, v in zip(self.metrics, v):
                        log["val_"+m] = v
                else:
                    log[k] = v
            print("%d: %s" % (epoch, json.dumps(log)))
            if (self.monitor_mode == "min" and log[self.monitor] < self.monitor_best) or (self.monitor_mode == "max" and log[self.monitor] > self.monitor_best):
                self.monitor_best = log[self.monitor]
                self._save_checkpoint(epoch)

    def _train_epoch(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch):
        arch = type(self.model).__name__
        state = {
                'arch': arch,
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'monitor_best': self.monitor_best
                }

        filename = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth.tar')
        torch.save(state, filename)


    def _resume_checkpoint(self, resume_path):
        print("Loadding checkpoint %s" % resume_path)
        checkpoint = torch.load(resume_path)

        self.start_epoch = checkpoint["epoch"] + 1
        self.monitor_best = checkpoint["monitor_best"]
        self.model.load_state_dict(checkpoint["state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.with_cuda:
            for state in self.optimizer.state.values():
                for k,v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()


def acc(preds, target):
    # compute accuracy
    correct = np.sum(preds == target)
    return correct / len(preds)

class Trainer(BaseTrainer):
    def __init__(self, model, loss, metrics, resume, config, data_loader, valid_data_loader=None):
        super(Trainer, self).__init__(model, loss, metrics, resume, config)

        self.config = config
        self.metrics = metrics
        self.batch_size = data_loader.batch_size
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.valid = True if self.valid_data_loader is not None else False
    
    def _to_var(self, tensor, tensor_type):
        var = torch.autograd.Variable(tensor_type(tensor))
        if self.config["with_cuda"]:
            var = var.cuda()
        return var

    def _eval_metrics(self, logits, target):
        acc_metrics = np.zeros(len(self.metrics))
        output = logits.cpu().data.numpy()
        target = target.cpu().data.numpy()

        output = np.argmax(output, axis=1)
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += eval(metric)(output, target)
        return acc_metrics

    def _train_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.data_loader):

            self.optimizer.zero_grad()
            output = self.model(data)
            target = self._to_var(target, torch.LongTensor)
            loss = self.loss(output["logits"], target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.cpu().data.item()
            total_metrics += self._eval_metrics(output["logits"], target)

        
        log = {
                'loss': total_loss / len(self.data_loader),
                'metrics': (total_metrics / len(self.data_loader)).tolist()
            }

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log
    
    def _valid_epoch(self):
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))

        for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            target = self._to_var(target, torch.LongTensor)
            output = self.model(data)
            loss = self.loss(output["logits"], target)

            total_val_loss += loss.cpu().data.item()
            total_val_metrics += self._eval_metrics(output["logits"], target)

        return {
                "val_loss": total_val_loss / len(self.valid_data_loader),
                "val_metrics": (total_val_metrics / len(self.valid_data_loader)).tolist()
                }



