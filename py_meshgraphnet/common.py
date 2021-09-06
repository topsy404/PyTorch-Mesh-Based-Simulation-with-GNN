#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/4 4:55 pm

import enum
import torch

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def triangles_to_edges(faces):
    """

    :param faces:
    :return:
    """
    # get 3 edges from a triangle face: (e1, e2), (e2, e3), (e3, e1)
    edges = torch.cat([faces[:, 0:2],
                       faces[:, 1:3],
                       torch.stack([faces[:, 2], faces[:, 0]], dim=1),
                       ], dim=0)
    # remove the duplicate edges
    receivers = torch.min(edges, dim = 1).values
    senders = torch.max(edges, dim = 1).values

    packed_edges = torch.stack([senders, receivers], dim = 1).type(torch.int64)
    # todo: verify the method to find unique edges
    unique_edges = torch.unique(packed_edges, dim = 0 )
    senders, receivers = unique_edges[:,0], unique_edges[:,1]
    senders_and_receivers = torch.cat([senders, receivers], dim = 0)
    receivers_and_senders =  torch.cat([receivers, senders], dim = 0)
    return senders_and_receivers, receivers_and_senders

# faces = torch.tensor([[1,2,3], [2,3,4],[3,4,1]], dtype = torch.int64)
# print(faces)
# senders, receivers  = triangles_to_edges(faces)
# print(senders, receivers )

def grather_index_from_tf(index, num_of_dims):
    """

    :param index:
    :param num_of_dims:
    :return:
    """
    return index.unsqueeze(-1).repeat(1, num_of_dims)
