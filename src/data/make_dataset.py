# -*- coding: utf-8 -*-
import logging
import torch_geometric.data  # type: ignore
from ogb.nodeproppred import PygNodePropPredDataset


def load_data(dataset_name: str, root: str) -> torch_geometric.data.Data:
    """ 
        - dataset_name (str) - name of the dataset to load
        - root (str) - path to thr root of the folder storing the dataset
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    if 'ogb' in dataset_name:
        dataset = PygNodePropPredDataset(name = dataset_name, root = root)
    else:
        dataset = None

    return dataset
