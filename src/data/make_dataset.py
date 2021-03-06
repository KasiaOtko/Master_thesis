# -*- coding: utf-8 -*-
import os

import numpy as np
import pandas as pd
import torch
import torch_geometric.data  # type: ignore
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

# sys.path.append("..")

def load_data(dataset_name: str, root: str) -> torch_geometric.data.Data:
    """
    - dataset_name (str) - name of the dataset to load
    - root (str) - path to thr root of the folder storing the dataset
    """

    if "ogb" in dataset_name:
        dataset = PygNodePropPredDataset(name=dataset_name, root=root, transform=T.ToSparseTensor())
        return dataset
    else:
        #meta_dict = torch.load(root+"/submission_ogbn_eujudgements/meta_dict.pt")
        #dataset = PygNodePropPredDataset("ogbn-eujudgements", meta_dict=meta_dict)#root=root)
        dataset = torch.load(root + "/data/processed/G_big_final.pt")
        # links = pd.read_csv(root + "/data/processed/links_gcc_final.csv", index_col = 0, parse_dates = [4, 10])
        # cases = pd.read_csv(root + "/data/processed/cases_gcc_final.csv", index_col = 0, parse_dates = [5, 6])
        # X = np.load(root + "/data/processed/paragraph_embeddings_final.npy")
        # y = np.load(root + "/data/processed/targets_final.npy")

        return dataset
