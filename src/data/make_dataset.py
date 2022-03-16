# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import argparse
from ogb.nodeproppred import PygNodePropPredDataset

parser = argparse.ArgumentParser()
parser.add_argument("-root", "--root-ogb-dataset", default = 'data/raw',
                    help="Root directory of a folder storing OGB dataset")
parser.add_argument("-d", "--dataset_name",
                    help = "Name of the dataset to use (possible: 'ogbn-products', 'ogbn-arxiv', 'EU-judgements')")
args = parser.parse_args()

def main(dataset_name):
    """ 
        - dataset_nameReturn dataset
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    if 'ogb' in args.d:
        dataset = PygNodePropPredDataset(name = dataset_name, root = args.root)
    else:
        dataset = None

    return dataset

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    data = main('ogbn-arxiv')
