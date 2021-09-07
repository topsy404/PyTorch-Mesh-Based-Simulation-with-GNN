#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/6 下午5:30
import tensorflow.compat.v1 as tf
#tf.enable_eager_execution()
import numpy as np
import pickle
from meshgraphnets import dataset, cfd_model, cfd_eval, cloth_model, cloth_eval
PARAMETERS = {
    'cfd': dict(noise=0.02, gamma=1.0, field='velocity', history=False,
                size=2, batch=2, model=cfd_model, evaluator=cfd_eval),
    'cloth': dict(noise=0.003, gamma=0.1, field='world_pos', history=True,
                  size=3, batch=1, model=cloth_model, evaluator=cloth_eval)
}

params = PARAMETERS["cloth"]
ds = dataset.load_dataset("meshgraphnets/dataset/flag_simple/", 'test')
print(ds)
ds = dataset.add_targets(ds, [params['field']], add_history=params['history'])
print(ds)
count = 0
with tf.Session() as sess:
    inputs_dict = {}
    fp = open("test10.pkl", 'ab')
    for i in range(10):
        inputs = tf.data.make_one_shot_iterator(ds).get_next()
        count += 1
        new_inputs = {}
        for key, val in inputs.items():
            new_inputs[key] = np.array(val.eval())
        inputs_dict[count] = new_inputs
        print("finished #{} ".format(i))
        pickle.dump(new_inputs, fp)


    # for i in range(5):
    #     inputs = tf.data.make_one_shot_iterator(ds).get_next()
    #     print(i)
    #     world_pos = inputs["world_pos"].eval()
    #     cells = inputs["cells"].eval()
    #     with open(str(i) + "_world_pos.pkl", 'wb') as fp:
    #         pickle.dump(np.array(world_pos), fp)
    #     with open(str(i) + "_cells.pkl", 'wb') as fp:
    #         pickle.dump(np.array(cells), fp)

