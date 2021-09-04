#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/4 下午7:10

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
        self._max_accumulations = max_accumulations
        self._std_epsilon = std_epsilon

    def forward(self, batched_data, accumulate = True):
        return None


