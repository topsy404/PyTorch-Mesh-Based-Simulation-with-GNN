#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/4 下午7:10
import pdb

import torch
from torch import nn

class Normalizer(torch.nn.Module):
    def __init__(self,
                 size,
                 max_accumulations=10 ** 6,
                 std_epsilon=1e-8,
                 name="Normalizer"
                 ):
        super(Normalizer, self).__init__()
        self._max_accumulations = torch.tensor(max_accumulations, dtype = torch.float32)
        self._std_epsilon = torch.tensor(std_epsilon, dtype = torch.float32)
        self._acc_count = torch.tensor(0, dtype=torch.float32)
        self._num_accumulations = torch.tensor(0, dtype=torch.float32)
        self._acc_sum = torch.zeros(size, dtype=torch.float32)
        self._acc_sum_squared = torch.zeros(size, dtype=torch.float32)


    def forward(self, batched_data, accumulate = True):
        if accumulate:
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data)
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """
        shape ?, 7
        :param batched_data:
        :return:
        """
        #todo: is reasonable that the count is equal to the number of nodes?
        count = torch.tensor(batched_data.shape[0], dtype=torch.float32)
        data_sum = torch.sum(batched_data, dim = 0)
        squared_data_sum = torch.sum(batched_data ** 2, dim = 0)
        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        pdb.set_trace()
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0))
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean()  ** 2)
        return torch.maximum(std, self._std_epsilon)



