#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/4

import collections
import functools

import torch
from torch import nn
from torch_scatter import scatter

EdgeSet = collections.namedtuple("EdgeSet", ["name", "features", "senders", "receivers"])

MultiGraph = collections.namedtuple("Grahp", ["node_features", "edge_sets"])


class MLP(nn.Module):
    def __init__(self,
                 latent_size,
                 num_layers,
                 output_size,
                 layer_norm=True,
                 activate_final=False):
        super(MLP, self).__init__()
        self._latent_size = latent_size
        self._num_layers = num_layers
        self._output_size = output_size
        self._layer_norm = layer_norm
        widths = [self._latent_size] * self._num_layers + [output_size]
        self._networks = []
        # networks += [nn.Linear(input_size, widths[0]), nn.ReLU())

        for i in range(len(widths) - 1):
            input_size, output_size = widths[i], widths[i + 1]
            self._networks.append(nn.Linear(input_size, output_size))
            if i == len(widths) - 2 and not activate_final:
                continue
            else:
                self._networks.append(nn.ReLU())

        if activate_final:
            self._networks.append(nn.ReLU())
        if self._layer_norm:
            # todo : determine which axis to be normalized
            self._networks.append(nn.LayerNorm(output_size))

    def forward(self, features):
        """

        :param features: shape: #nodes, ?
        :return:
        """
        input_size = features.shape[-1]
        self._networks = [nn.Linear(input_size, self._latent_size), nn.ReLU()] + self._networks
        self._mlp = nn.Sequential(*self._networks)
        return self._mlp(features)


class GraphNetBlock(nn.Module):
    def __init__(self, model_fn, name="GraphNetBlock"):
        super(GraphNetBlock, self).__init__()
        self._model_fn = model_fn

    def _update_edge_features(self, node_features, edge_set):
        sender_features = torch.gather(node_features, dim = 0, index = edge_set.senders)  # index should be a torch.Tensor with type int64
        receiver_features = torch.gather(node_features, dim = 0, index = edge_set.receivers)
        features_lst = [sender_features, receiver_features, edge_set.features]
        edge_features =  torch.cat(features_lst, dim = -1)
        return self._model_fn()(edge_features)

    def _update_node_features(self, node_features, edge_sets):
        """

        :param node_features:
        :param edge_set:
        :return:
        """
        num_nodes = node_features.shape[0]
        features = [node_features]
        for edge_set in edge_sets:
            edge_set_features = edge_set.features
            edge_set_receivers = edge_set.receivers
            seg_sum = torch.zeros((num_nodes, edge_set_features.shape[-1]))
            scatter(edge_set_features, edge_set_receivers, 0, seg_sum)
            features.append(seg_sum)
        return self._model_fn()(torch.cat(features, dim = -1))

    def forward(self, graph):
        """
        Replicate from _build
        update edge feature -> update node feature -> apply residual connection
        :param graph:
        :return:
        """
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features,
                                                          edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(graph.node_features,
                                                       new_edge_sets)

        # add residual connections
        new_node_features += graph.node_features
        new_edge_sets = [es._replace(features=es.features + old_es.features)
                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        return MultiGraph(new_node_features, new_edge_sets)

class EncoderProcessDecode(nn.Module):
    def __init__(self,
                 output_size,
                 latent_size,
                 num_layers,
                 message_passing_steps,
                 name="EncoderProcessDecode"):
        """

        :param output_size:
        :param latent_size:
        :param num_layers:
        :param message_passing_steps:
        :param name:
        """
        super(EncoderProcessDecode, self).__init__(name=name)
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps

    def _make_mlp(self, output_size, layer_norm=True):
        # build an MLP
        network = MLP(output_size, self._latent_size, self._num_layers, layer_norm)
        return network

    def _encoder(self, graph):
        node_latents = self._make_mlp(self._latent_size)(graph.node_features)
        new_edges_sets = []
        for edge_set in graph.edge_sets:
            latent = self._make_mlp(self._latent_size)(edge_set.features)
            new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets)

    def _decoder(self, graph):
        decoder = self._make_mlp(self._output_size, layer_norm=False)
        return decoder(graph.node_features)

    def forward(self, graph):
        """
        Replicate from _build
        encode graph -> message propagation -> decoder
        :param graph:
        :return:
        """
        model_fn = functools.partial(self._make_mlp, output_size=self._latent_size)
        latent_graph = self._encoder(graph)
        for _ in range(self._message_passing_steps):
            latent_graph = GraphNetBlock(model_fn)(latent_graph)
        return self._decoder(latent_graph)
