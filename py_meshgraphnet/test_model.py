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

import core_model

import numpy as np

np.random.seed(0)

"""
test MLP
"""
# test_mlp = core_model.MLP(latent_size=128,
#                           num_layers=2,
#                           output_size=3)
# print(test_mlp)
# test_feature_np = np.random.rand(10, 3)
# test_feature_tensor = torch.Tensor(test_feature_np)
# output_mlp = test_mlp(test_feature_tensor)
# print(output_mlp)

"""
test Model in cloth_model
"""

import cloth_model
import core_model
import common
#
test_core_model = core_model.EncoderProcessDecode(output_size=3,
                                                  latent_size=128,
                                                  num_layers=2,
                                                  message_passing_steps=15)

test_cloth_model = cloth_model.Model(learned_model=test_core_model)
print("test core model: ", test_core_model )
print("test cloth model: ", test_cloth_model)


inputs = {}
inputs["cells"] = np.random.randint(0, 1000, (3028, 3))
inputs["mesh_pos"] = np.random.randn(1579, 2)
inputs["node_type"] = np.random.randint(0, 3, (1579, 1))
inputs["world_pos"] = np.random.randn(1579, 3)
inputs["prev|world_pos"] = np.random.randn(1579, 3)
inputs["target|world_pos"] = np.random.randn(1579, 3)
for key, val in inputs.items():
    inputs[key] = torch.tensor(inputs[key], dtype=torch.float32)
inputs["cells"] = inputs["cells"].to(torch.int64)
inputs["node_type"] = inputs["node_type"].to(torch.int64)

print("inputs: ", inputs)
test_cloth_model_output = test_cloth_model.forward(inputs)
print("test_cloth_model_output: ", test_cloth_model_output.shape)


"""
test GraphNetBlock
"""

"""
test EncoderProcessDecoder
"""

"""
learned_model(EncoderProcessDecoder) -> Model in cloth model -> build graph, graph model, update model
"""
