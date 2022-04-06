# -*- coding: utf-8 -*-
import os
import pandas as pd
import torch
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
        dataset = torch.load("/data/processed/EU_graph.pt")
        print(os.getcwd())
        # links = pd.read_csv(root + "/data/processed/eu_links_lang_gcc.csv", index_col = 0, parse_dates = [4, 10])
        # cases = pd.read_csv(root + "/data/processed/eu_cases_lang_gcc.csv", index_col = 0, parse_dates = [5, 6])
        # X = np.load(root + "/data/processed/paragraph_embeddings.npy")
        # y = np.load(root + "/data/processed/targets.npy")

        return dataset
