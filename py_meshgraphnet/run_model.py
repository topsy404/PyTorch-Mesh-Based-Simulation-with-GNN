#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/6 下午4:05


import pickle
from absl import app
from absl import flags
from absl import logging
import numpy as np

import torch
import dataset
import cloth_model
import core_model

import pdb

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval'],
                  'Train model, or run evaluation.')
flags.DEFINE_enum('model', None, ['cfd', 'cloth'],
                  'Select model to run.')
flags.DEFINE_string('checkpoint_dir', None, 'Directory to save checkpoint')
flags.DEFINE_string('dataset_dir', None, 'Directory to load dataset from.')
flags.DEFINE_string('rollout_path', None,
                    'Pickle file to save eval trajectories')
flags.DEFINE_enum('rollout_split', 'valid', ['train', 'test', 'valid'],
                  'Dataset split to use for rollouts.')
flags.DEFINE_integer('num_rollouts', 10, 'No. of rollout trajectories')
flags.DEFINE_integer('num_training_steps', int(10e6), 'No. of training steps')


PARAMETERS = {
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=None)
}

def learner(model, params):
    """
    load data -> compute loss --> optimize
    :param model:
    :param params:
    :return:
    """
    # todo: create data loader

    # optimizer
    # print(model.parameters())
    # for e in model.parameters():
    #     print(e.shape)
    my_optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optimizer, gamma=0.1)
    num_epoches = 20000
    dataloder = dataset.get_dataloader()
    for epoch in range(num_epoches):
        for index, batched_data in enumerate(dataloder):

            inputs = {}
            for key, val in batched_data.items():
                inputs[key] = torch.squeeze(val, dim = 0)
            # for key, val in inputs.items():
            #     print("key: {}, val shape: {}".format(key, val.shape))

            my_optimizer.zero_grad()
            loss = model.loss(inputs)
            if index % 10 == 0:
                print("index {}, loss: {}".format(index, loss))

            loss.backward()
            my_optimizer.step()
        my_lr_scheduler.step()
        if epoch % 10 == 1:
            path = "train_epoch_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), path)
        if epoch % 1000 == 0:
            print("#epoch: {}; loss: {}".format(epoch, loss) )





def main(argv):
    params = PARAMETERS["cloth"]  #
    learned_model = core_model.EncoderProcessDecode(
        output_size=params['size'],  # 3
        latent_size=128,
        num_layers=2,
        message_passing_steps=15)
    model = params['model'].Model(learned_model)
    learner(model, params)

if __name__ == "__main__":
    app.run(main)
