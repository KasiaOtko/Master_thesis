import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import degree


def accuracy_per_degree(model, dataset):
    if dataset == "ogbn-products":
        G_pyg = PygNodePropPredDataset(name='ogbn-products')[0]
    elif dataset == "ogbn-arxiv":
        G_pyg = PygNodePropPredDataset(name='ogbn-arxiv')[0]
    else:
        G_pyg = torch.load("Data/EU_judgements/G_big_final.pt")
    preds = np.load("REPO/models/preds/{}_{}.npy".format(model, dataset)).reshape(-1)
    G_pyg.preds = preds
    G_pyg.degree = degree(G_pyg.edge_index[0], G_pyg.x.shape[0])
    
    df = pd.DataFrame((G_pyg.y.numpy(), G_pyg.preds, G_pyg.degree.numpy().astype(int))).T
    df.columns = ["y", "pred", "degree"]
    df["degree_"] = df["degree"].apply(lambda x: x if x <= 10 else "> 10")
    df["correct"] = df["y"] == df["pred"]
    summary = df.groupby("degree_").mean()["correct"]
    
    plt.figure(figsize = (10, 5))
    plt.bar(summary.index.astype(str), summary.values)
    plt.xlabel("Degree"); plt.ylabel("Accuracy")
    plt.title("Accuracy per node degree ({} model, {} dataset)".format(model.upper(), dataset))
    

def misclass_edge_types(model, dataset):
    if dataset == "ogbn-products":
        G_pyg = PygNodePropPredDataset(name='ogbn-products')[0]
    elif dataset == "ogbn-arxiv":
        G_pyg = PygNodePropPredDataset(name='ogbn-arxiv')[0]
    else:
        G_pyg = torch.load("Data/EU_judgements/G_big_final.pt")
    preds = np.load("REPO/models/preds/{}_{}.npy".format(model, dataset)).reshape(-1)
    G_pyg.preds = preds
    G_pyg.edge_index = G_pyg.edge_index.numpy()
    G_pyg.edge_type = G_pyg.edge_type.numpy()
    
    misclassified_nodes = (G_pyg.y.numpy() != G_pyg.preds).nonzero()[0]
    edge_idxs = [(G_pyg.edge_index[0,:] == node).nonzero()[0] for node in misclassified_nodes]
    edge_types = [G_pyg.edge_type[edge_idx] for edge_idx in edge_idxs]
    proportions_mis = [np.mean(edge_type) for edge_type in edge_types]
    
    classified_nodes = (G_pyg.y.numpy() == G_pyg.preds).nonzero()[0]
    edge_idxs = [(G_pyg.edge_index[0,:] == node).nonzero()[0] for node in classified_nodes]
    edge_types = [G_pyg.edge_type[edge_idx] for edge_idx in edge_idxs]
    proportions = [np.mean(edge_type) for edge_type in edge_types]
    
    return proportions, proportions_mis