#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
from copy import copy
import numpy as np

class BaseDataLoader:
    def __init__(self, data, batch_size, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        # return number of batches
        num_batches = int(np.ceil(1.*len(self.data)/self.batch_size))
        return num_batches

    @property
    def n_sampels(self):
        return len(self.data)

    def __iter__(self):
        assert self.__len__() > 0
        self.batch_idx = 0
        if self.shuffle:
            self._shuffle_data()

        return self

    def _shuffle_data(self):
        np.random.shuffle(self.data)

    def __next__(self):
        if self.batch_idx < self.__len__():
            batch = self.data[self.batch_size*self.batch_idx:self.batch_size*(self.batch_idx+1)]
            self.batch_idx = self.batch_idx + 1
            return self._unpack_data(batch)
        else:
            raise StopIteration

    def _unpack_data(self, batch):
        raise NotImplementedError



if __name__ == "__main__":
    pass

