#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/5 3:46 PM

"""
test list
test in core_model
test MLP
test GraphNetBlock
test EncoderProcessDecoder

test in normalization
test Normalizer

test in cloth_model
test Model in cloth model
"""
import torch

from py_meshgraphnet import  core_model

import numpy as np
np.random.seed(0)

test_mlp = core_model.MLP(latent_size= 128,
                          num_layers= 2,
                          output_size= 3)


print(test_mlp)
test_feature_np = np.random.rand(10,3)
test_feature_tensor = torch.Tensor(test_feature_np)
output_mlp = test_mlp(test_feature_tensor)
print(output_mlp)


