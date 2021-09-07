#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: topsy 
# Data: 2021/9/6 下午4:13


import torch
from torch.utils.data import DataLoader, Dataset
import os
import pickle


class ClothDataset(Dataset):
    def __init__(self, pkl_file_name, root_dir):
        super(ClothDataset, self).__init__()
        file_path = os.path.join(root_dir, pkl_file_name)
        print(file_path)
        with open(file_path, "rb") as fp:
            self._data = pickle.load(fp)

    def __len__(self):
        return len(self._data["cells"])

    def __getitem__(self, idx):
        item = {}
        for key, val in self._data.items():
            if key in ["cells", "node_type"]:
                item[key] = torch.tensor(val[idx], dtype=torch.int64)
            else:
                item[key] = torch.tensor(val[idx], dtype=torch.float32)
        return item


"""
iterating through the dataset
"""
clothdataset = ClothDataset( pkl_file_name="train10.pkl",
                             root_dir="/home/topsy/Documents/projects/PyTorch-Mesh-Based-Simulation-with-GNN/py_meshgraphnet/dataset",)


# for i in range(len(clothdataset)):
#     sample = clothdataset[i]
#     print("i: {}, cells shape: {}".format(i, sample["cells"].shape))

def get_dataloader():
    dataloder = DataLoader(clothdataset, batch_size=1, shuffle=True, num_workers=2)
    return dataloder

# for index, batched_data in enumerate(dataloder):
#     if index % 10 == 0:
#         print("index: {}, batched_data shape: {}".format(index, batched_data["cells"].shape))
