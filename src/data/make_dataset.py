# -*- coding: utf-8 -*-
import pandas as pd
import torch_geometric.data  # type: ignore
from ogb.nodeproppred import PygNodePropPredDataset


def load_data(dataset_name: str, root: str) -> torch_geometric.data.Data:
    """
    - dataset_name (str) - name of the dataset to load
    - root (str) - path to thr root of the folder storing the dataset
    """

    if "ogb" in dataset_name:
        dataset = PygNodePropPredDataset(name=dataset_name, root=root)
        return dataset
    else:
        links = pd.read_csv("data/raw/interim/eu_links.csv", index_col = 0, parse_dates = [4, 10])
        cases = pd.read_csv("Data/EU_judgements/eu_cases.csv", index_col = 0, parse_dates = [5, 6])
        return links, cases
