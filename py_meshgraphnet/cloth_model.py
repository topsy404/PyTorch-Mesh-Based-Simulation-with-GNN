#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/4 4:43 pm


import torch
from torch import nn
import numpy as np

from py_meshgraphnet import common
from py_meshgraphnet import core_model
from py_meshgraphnet import normalization


class Model(torch.nn.Module):
    def __init__(self, learned_model, name='Model'):
        super(Model, self).__init__(name=name)
        self._learned_model = learned_model
        self._learned_model = learned_model  # learned_model = core_model.EncodeProcessDecode
        self._output_normalizer = normalization.Normalizer(
            size=3, name='output_normalizer')
        self._node_normalizer = normalization.Normalizer(
            size=3 + common.NodeType.SIZE, name='node_normalizer')
        self._edge_normalizer = normalization.Normalizer(
            size=7, name='edge_normalizer')  # 2D coord + 3D coord + 2*length = 7

    def _node_normalizer(self, node_features, is_training):
        return torch.norm(node_features, dim=-1, keepdim=True)

    def _build_graph(self, inputs, is_training):
        """
        :param inputs:
            key:  cells ;  shape:  (3028, 3)
            key:  mesh_pos ;  shape:  (1579, 2)　＃relative position
            key:  node_type ;  shape:  (1579, 1) # fixed or not
            key:  world_pos ;  shape:  (1579, 3)
            key:  prev|world_pos ;  shape:  (1579, 3)
            key:  target|world_pos ;  shape:  (1579, 3)
        :param is_training:
        :return:
        """
        # inputs = {}
        # inputs["cells"] = np.random.randint(0, 1000, (3028, 3))
        # inputs["mesh_pos"] = np.random.randn(1579, 2)
        # inputs["node_type"] = np.random.randint(0, 3, (1579, 1))
        # inputs["world_pos"] = np.random.randn(1579, 3)
        # inputs["prev|world_pos "] = np.random.randn(1579, 3)
        # inputs["target|world_pos"] = np.random.randn(1579, 3)
        velocity = inputs["world_pos"] - inputs["prev|word_pos"]  # shape: #nodes, 3
        node_type = torch.nn.functional.one_hot(inputs["node_type"],
                                                num_classes=common.NodeType.SIZE)  # shape: #nodes, SIZE
        node_features = torch.cat([velocity, node_type], dim=-1)
        senders, receivers = common.triangles_to_edges(inputs["cells"])
        relative_world_pos = torch.gather(inputs["world_pos"], dim=0, index=senders) - \
                             torch.gather(inputs["world_pos"], dim=0, index=receivers)
        relative_mesh_pos = torch.gather(inputs["mesh_pos"], dim=0, index=senders) - \
                            torch.gather(inputs["mesh_pos"], dim=0, index=receivers)
        edge_features = torch.cat([relative_world_pos,
                                   torch.norm(relative_world_pos, dim=-1, keepdim=True),
                                   relative_mesh_pos,
                                   torch.norm(relative_mesh_pos, dim=-1, keepdim=True),
                                   ])
        mesh_edge = core_model.EdgeSet(name="mesh_edges",
                                       features=edge_features,
                                       receivers=receivers,
                                       senders=senders)
        # todo: implement the custom normalization method
        node_features = self._node_normalizer(node_features, is_training)
        edge_sets = [mesh_edge]
        return core_model.MultiGraph(node_features=node_features,
                                     edge_sets=edge_sets)

    def loss(self, inputs):
        """

        :param inputs:
        :return:
        """
        graph = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(graph)

        # build target acceleration
        cur_position = inputs["world_pos"]
        prev_position = inputs["prev|world_pos"]
        target_position = inputs["target|world_pos"]
        target_acceleration = target_position - 2 * cur_position + prev_position
        target_normalized = self._output_normalizer(target_acceleration)

        # build loss
        loss_mask = torch.equal(inputs["node_type"][:, 0], torch.tensor(common.NodeType.NORMAL))
        error = torch.sum((target_normalized - network_output) ** 2, dim=1)
        loss = torch.mean(error[loss_mask])
        return loss

    def _update(self, inputs, per_node_network_output):
        acceleration = self._output_normalizer.inverse(per_node_network_output)
        # integrate forward
        cur_position = inputs['world_pos']
        prev_position = inputs['prev|world_pos']
        position = 2 * cur_position + acceleration - prev_position
        return position

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        graph = self._build_graph(inputs, is_training=True)
        per_node_network_output = self._learned_model(graph)
        return self._update(inputs, per_node_network_output)
