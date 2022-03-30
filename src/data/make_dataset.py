# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch_geometric.data  # type: ignore
from ogb.nodeproppred import PygNodePropPredDataset

# sys.path.append("..")

def load_data(dataset_name: str, root: str) -> torch_geometric.data.Data:
    """
    - dataset_name (str) - name of the dataset to load
    - root (str) - path to thr root of the folder storing the dataset
    """

    if "ogb" in dataset_name:
        dataset = PygNodePropPredDataset(name=dataset_name, root=root)
        return dataset
    else:
        print(os.getcwd())
        links = pd.read_csv(root + "/data/processed/eu_links_lang.csv", index_col = 0, parse_dates = [4, 10])
        cases = pd.read_csv(root + "/data/processed/eu_cases_03_16.csv", index_col = 0, parse_dates = [5, 6])
        X = np.load(root + "/data/processed/paragraph_embeddings.npy")
        y = links

        return X, links, cases
