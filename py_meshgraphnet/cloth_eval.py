#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/7 下午4:38


import torch
import common


def _rollout(model, state, prev_pos, cur_pos):
    normal_node_mask = torch.eq(state["node_type"][:, 0],
                                torch.full(state["node_type"][:, 0].shape, common.NodeType.NORMAL))

    prediction = model({**state,
                        'prev|world_pos': prev_pos,
                        'world_pos': cur_pos })
    next_pos = torch.where(normal_node_mask.reshape(-1, 1).repeat(1,3), prediction, cur_pos)
    return next_pos

def evaluate(model, inputs):
    """
    :param model: model with trained parameters
    :param inputs: tensor
    :return:
    """
    num_steps = inputs["cells"].shape[0]
    inital_state = inputs
    for k, val in inital_state.items():
        print("k : {}, val : {}".format(k, val.shape))
    prev_pos = inital_state["prev|world_pos"]
    cur_pos = inital_state["world_pos"]
    trajectory = []
    for step in range(num_steps):
        next_pos = _rollout(model, inital_state, prev_pos, cur_pos)
        trajectory.append(next_pos)
        prev_pos = cur_pos
        cur_pos = next_pos
    return torch.cat(trajectory, dim = 0)





