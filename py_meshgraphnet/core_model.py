#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/4

import collections
import functools
import pdb

import torch
from torch import nn
from torch_scatter import scatter

EdgeSet = collections.namedtuple("EdgeSet", ["name", "features", "senders", "receivers"])

MultiGraph = collections.namedtuple("Grahp", ["node_features", "edge_sets"])


class MLP(nn.Module):
    def __init__(self,
                 input_size,
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
        self._networks = [nn.Linear(input_size, self._latent_size), nn.ReLU()]
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
        self._mlp = nn.Sequential(*self._networks)

    def forward(self, features):
        """

        :param features: shape: #nodes, ?
        :return:
        """
        input_size = features.shape[-1]
        self._networks = [nn.Linear(input_size, self._latent_size), nn.ReLU()] + self._networks
        # pdb.set_trace()
        return self._mlp(features)


class GraphNetBlock(nn.Module):
    def __init__(self, edge_features_size, node_feature_size, latent_size, num_layers, output_size, layer_norm=True):
        super(GraphNetBlock, self).__init__()
        self._edge_encoder = MLP(edge_features_size, latent_size, num_layers, output_size, layer_norm)
        self._node_encoder = MLP(node_feature_size, latent_size, num_layers, output_size, layer_norm)

    def _update_edge_features(self, node_features, edge_set):
        num_of_dims_of_features = node_features.shape[-1]
        sender_features = torch.gather(node_features, dim=0,
                                       index=edge_set.senders.unsqueeze(-1).repeat(1,
                                                                                   num_of_dims_of_features))  # index should be a torch.Tensor with type int64
        receiver_features = torch.gather(node_features, dim=0,
                                         index=edge_set.receivers.unsqueeze(-1).repeat(1, num_of_dims_of_features))
        features_lst = [sender_features, receiver_features, edge_set.features]  # (128 + 128 + 128)
        edge_features = torch.cat(features_lst, dim=-1)
        #pdb.set_trace()
        return self._edge_encoder(edge_features)

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
        new_features = torch.cat(features, dim=-1)
        return self._node_encoder(new_features)

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


class GraphNetBlockBack(nn.Module):
    def __init__(self, model_fn, name="GraphNetBlock"):
        super(GraphNetBlock, self).__init__()
        self._model_fn = model_fn

    def _update_edge_features(self, node_features, edge_set):
        num_of_dims_of_features = node_features.shape[-1]
        sender_features = torch.gather(node_features, dim=0,
                                       index=edge_set.senders.unsqueeze(-1).repeat(1,
                                                                                   num_of_dims_of_features))  # index should be a torch.Tensor with type int64
        receiver_features = torch.gather(node_features, dim=0,
                                         index=edge_set.receivers.unsqueeze(-1).repeat(1, num_of_dims_of_features))
        features_lst = [sender_features, receiver_features, edge_set.features]
        edge_features = torch.cat(features_lst, dim=-1)
        return self._model_fn(input_size=edge_features.shape[-1])(edge_features)

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
        new_features = torch.cat(features, dim=-1)
        return self._model_fn(input_size=new_features.shape[-1])(new_features)

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
        super(EncoderProcessDecode, self).__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._node_encoder_net = MLP(input_size=12,
                                     latent_size=128,
                                     num_layers=2,
                                     output_size=128)
        self._edge_encoder_net = MLP(input_size=7,
                                     latent_size=128,
                                     num_layers=2,
                                     output_size=128)
        self._node_decoder_net = MLP(input_size=128,
                                     latent_size=128,
                                     num_layers=2,
                                     output_size=output_size)
        self._edge_decoder_net = MLP(input_size=128,
                                     latent_size=128,
                                     num_layers=2,
                                     output_size=output_size)
        self._process_net_lst = []
        for step in range(self._message_passing_steps):
            self._process_net_lst.append(
                GraphNetBlock(edge_features_size=128 * 3, node_feature_size=128 * 2, latent_size=latent_size,
                              num_layers=num_layers, output_size=128))
        self._process_net = nn.Sequential(*self._process_net_lst)

    def _make_mlp(self, input_size, output_size, layer_norm=True):
        # build an MLP
        network = MLP(input_size, self._latent_size, self._num_layers, output_size, layer_norm)
        return network

    def _encoder(self, graph):
        node_latents = self._node_encoder_net(graph.node_features)
        new_edges_sets = []
        for edge_set in graph.edge_sets:
            latent = self._edge_encoder_net(edge_set.features)
            new_edges_sets.append(edge_set._replace(features=latent))

        return MultiGraph(node_latents, new_edges_sets)

    def _decoder(self, graph):
        node_latents = self._node_decoder_net(graph.node_features)
        new_edges_sets = []
        for edge_set in graph.edge_sets:
            latent = self._edge_decoder_net(edge_set.features)
            new_edges_sets.append(edge_set._replace(features=latent))

        return MultiGraph(node_latents, new_edges_sets)

    def _process(self, graph):
        for index in range(self._message_passing_steps):
            net = self._process_net[index]
            graph = net(graph)
            #self._show_graph_detail(str(index) + "_graph", graph)
        return graph

    def _show_graph_detail(self, name, graph):
        print("------------------------------\n")
        print("{} node_features shape {}".format(name, graph.node_features.shape))
        for edge in graph.edge_sets:
            print("name: {}, shape {}".format(edge.name, edge.features.shape))

    def forward(self, graph):
        """
        Replicate from _build
        encode graph -> message propagation -> decoder
        :param graph:
        :return:
        """
        latent_graph = self._encoder(graph)
        #self._show_graph_detail("latent graph", latent_graph)
        process_graph = self._process(latent_graph)
        decoder_graph = self._decoder(process_graph)
        return decoder_graph.node_features


class EncoderProcessDecodeBack(nn.Module):
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
        super(EncoderProcessDecodeBack, self).__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps

    def _make_mlp(self, input_size, output_size, layer_norm=True):
        # build an MLP
        network = MLP(input_size, self._latent_size, self._num_layers, output_size, layer_norm)
        return network

    def _encoder(self, graph):
        node_latents = self._make_mlp(graph.node_features.shape[-1], self._latent_size)(graph.node_features)
        new_edges_sets = []
        for edge_set in graph.edge_sets:
            latent = self._make_mlp(edge_set.features.shape[-1], self._latent_size)(edge_set.features)
            new_edges_sets.append(edge_set._replace(features=latent))
        return MultiGraph(node_latents, new_edges_sets)

    def _decoder(self, graph):
        decoder = self._make_mlp(graph.node_features.shape[-1], self._output_size, layer_norm=False)
        return decoder(graph.node_features)

    def _show_graph_detail(self, name, graph):
        print("------------------------------\n")
        print("{} node_features shape {}".format(name, graph.node_features.shape))
        for edge in graph.edge_sets:
            print("name: {}, shape {}".format(edge.name, edge.features.shape))
    def forward(self, graph):
        """
        Replicate from _build
        encode graph -> message propagation -> decoder
        :param graph:
        :return:
        """
        latent_graph = self._encoder(graph)
        self._show_graph_detail("latent graph", latent_graph)
        for i in range(self._message_passing_steps):
            latent_graph = GraphNetBlock(model_fn)(latent_graph)
            self._show_graph_detail(str(i) + "_graph", latent_graph)
        return self._decoder(latent_graph)
