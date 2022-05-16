import hydra
import numpy as np
import pandas as pd
import networkx as nx
import torch
from GAT_model import GATv2
from GCN_model import GCN, RGCN
from GraphSAGE_model import SAGE
from omegaconf import OmegaConf

from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils.convert import to_networkx

import wandb
from src.data.make_dataset import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path="../config", config_name="default_config.yaml")
def get_model_and_data(config):

    hparams = config.gcn.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = hparams["model"])
    orig_cwd = hydra.utils.get_original_cwd()
    
    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
        num_classes = len(np.unique(data[0].y))
        if hparams["model"] == "gcn":
            model = GCN(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"],
                        hparams["improved"], hparams["linear"]).to(device)
        elif hparams["model"] == "gat":
            model = GATv2(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["dropout_l"], hparams["heads"], 
                            hparams["negative_slope"],hparams["num_layers"], hparams["linear"]).to(device)
        elif hparams["model"] == "sage":
            model = SAGE(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"], hparams["aggr"], hparams["linear"]).to(device)    

    else:
        data = load_data(hparams.dataset.name, orig_cwd)
        num_classes = len(np.unique(data.y))
        if hparams["model"] == "gcn":
            model = GCN(data.x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"], 
            hparams["improved"], hparams["linear"]).to(device)
        if hparams["model"] == "rgcn":
            model = RGCN(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"],
                        hparams["num_bases"], hparams["num_layers"], False).to(device)
        elif hparams["model"] == "gat":
            model = GATv2(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"], hparams["dropout_l"], hparams["heads"],
                        hparams["num_bases"], hparams["negative_slope"],hparams["num_layers"], hparams["linear"]).to(device)
        elif hparams["model"] == "sage":
            model = SAGE(data.x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"],
                            hparams["aggr"], hparams["linear"]).to(device)
    
    data.x, data.y = data.x.float(), data.y.long()
    model.load_state_dict(torch.load("REPO/models/{}_{}.pt".format(hparams["model"], hparams.dataset.name), map_location=device))

    data.x.requires_grad = True
    h, _ = model(data.x, data.edge_index)

    num_nodes = hparams.num_nodes
    nodes_x = np.random.choice(range(data.x.shape[0]), num_nodes)

    influences = np.zeros((num_nodes, data.x.shape[0]))
    for i in range(num_nodes):
        influences[i,:] = influence_distribution(h, nodes_x[i], data).numpy()

    np.save(orig_cwd + "/models/influence/influence_{}_{}".format(hparams.model, hparams.dataset.name))

    # pyg = to_networkx(data, edge_attrs = ["edge_type"], node_attrs = ["ID", "CELEX"])

    # K = hparams["num_layers"]
    # df = pd.DataFrame(np.zeros((K, 2*num_nodes)), columns = ["mean", "std"]*num_nodes)
    # means, stds = [], []
    # for k in range(1, K+1):
    #     m, s = [], []
    #     for n in range(num_nodes):
    #         K_hop = nx.single_source_shortest_path_length(pyg, nodes_x[n], cutoff=K)
    #         k_hop = list(dict(filter(lambda elem: elem[1] == k, K_hop.items())).keys())
    #         m.append(influences[n,k_hop].mean())
    #         s.append(influences[n,k_hop].std())
    #     means.append(np.mean(m))
    #     stds.append(np.mean(s))

    #return model, data


def influence_distribution(embeddings, node_x, data):
    sum_of_grads = torch.autograd.grad(embeddings[node_x], 
                                       data.x, 
                                       torch.ones_like(embeddings[node_x]), 
                                       retain_graph=True)[0]
    abs_grad = sum_of_grads.absolute() # torch.reshape(a.absolute(),[34,-1])
    sum_of_jacobian = abs_grad.sum(axis=1)
    influence_y_on_x = sum_of_jacobian / sum_of_jacobian.sum(dim=0)
    return influence_y_on_x
    