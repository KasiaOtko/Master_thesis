import torch
import torch.nn as nn
import hydra
import wandb
import numpy as np
import logging
from GCN_model import GCN, RGCN

from omegaconf import DictConfig, OmegaConf

from src.data.make_dataset import load_data
from src.models.utils import data_split, log_details_to_wandb, prediction_scores, pyg_data_split, eval_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@hydra.main(config_path="../config", config_name="default_config.yaml")
def train_GCN(config):

    hparams = config.GCN.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    # wandb.init(project="master-thesis", config = hparams, group = "GCN")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
    log_details_to_wandb("GCN", hparams)
    logging.info("Data loaded.")

    epochs = hparams["epochs"]
    num_classes = len(np.unique(data.y))
    model = GCN(hparams["hidden_channels"], data.x.shape[1], )

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(data.edge_index, data.edge_type).argmax(dim=-1)
        train_acc = float((pred[data.train_mask] == data.y[data.train_mask]).float().mean())
        test_acc = float((pred[data.test_mask] == data.y[data.test_max]).float().mean())
    


def train_GAT(config):
    
    hparams = config.ffnn.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "GAT")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
    log_details_to_wandb("GAT", hparams)
    logging.info("Data loaded.")

    model = RGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    out, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        acc = float((out[mask].argmax(-1) == data.y[mask]).sum() / mask.sum())
        accs.append(acc)