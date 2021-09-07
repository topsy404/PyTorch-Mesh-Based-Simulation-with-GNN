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

import  torch
import torch.nn

net_lst = []
for i in range(5):
    net_lst.append(core_model.MLP(input_size=3,
                                  latent_size=128,
                                  num_layers=2,
                                  output_size=3))
new_net = torch.nn.Sequential(*net_lst)
for net in new_net:
    print(len(list(net.parameters())))
# test_mlp = core_model.MLP(input_size= 3,
#                           latent_size=128,
#                           num_layers=2,
#                           output_size=3)
# print(test_mlp)
#
# for e in list(test_mlp.parameters()):
#     print(e.shape)
#
# test_feature_np = np.random.rand(10, 3)
# test_feature_tensor = torch.Tensor(test_feature_np)
# output_mlp = test_mlp(test_feature_tensor)
# print(output_mlp)

"""
tensor([[ 1.0006,  0.3606, -1.3613],
        [ 1.1554,  0.1244, -1.2798],
        [ 0.7578,  0.6507, -1.4085],
        [ 0.8105,  0.5935, -1.4041],
        [ 0.6810,  0.7226, -1.4036],
        [ 0.7225,  0.6876, -1.4102],
        [ 1.1617,  0.1135, -1.2752],
        [ 1.2434, -0.0422, -1.2012],
        [ 0.6635,  0.7441, -1.4076],
        [ 1.1917,  0.0594, -1.2510]], grad_fn=<NativeLayerNormBackward>)

"""

"""
test Model in cloth_model
"""
#
# import cloth_model
# import core_model
# import common
#
#
#
#
# test_core_model = core_model.EncoderProcessDecode(output_size=3,
#                                                   latent_size=128,
#                                                   num_layers=2,
#                                                   message_passing_steps=15)
#
# test_cloth_model = cloth_model.Model(learned_model=test_core_model)
# print("test core model: ", test_core_model )
# print("test cloth model: ", test_cloth_model)
# for e in list(test_core_model.parameters()):
#     print(e.shape)

# inputs = {}
# inputs["cells"] = np.random.randint(0, 1000, (3028, 3))
# inputs["mesh_pos"] = np.random.randn(1579, 2)
# inputs["node_type"] = np.random.choice([0,3], (1579, 1))
# inputs["world_pos"] = np.random.randn(1579, 3)
# inputs["prev|world_pos"] = np.random.randn(1579, 3)
# inputs["target|world_pos"] = np.random.randn(1579, 3)
# for key, val in inputs.items():
#     inputs[key] = torch.tensor(inputs[key], dtype=torch.float32)
# inputs["cells"] = inputs["cells"].to(torch.int64)
# inputs["node_type"] = inputs["node_type"].to(torch.int64)
#
# print("inputs: ", inputs)
# test_cloth_model_output = test_cloth_model.forward(inputs)
#
# print("test_cloth_model_output: ", test_cloth_model_output)
# loss = test_cloth_model.loss(inputs)
# print("loss: ", loss)



"""
test GraphNetBlock
"""

"""
test EncoderProcessDecoder
"""

"""
learned_model(EncoderProcessDecoder) -> Model in cloth model -> build graph, graph model, update model
"""
