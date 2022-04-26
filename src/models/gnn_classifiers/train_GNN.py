from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import wandb
import numpy as np
import logging
from ogb.nodeproppred import Evaluator
from GCN_model import GCN, RGCN
from torch_geometric.loader import NeighborSampler, ClusterData, ClusterLoader
from GAT_model import GAT, RGAT

from omegaconf import OmegaConf
import multiprocessing
from src.data.make_dataset import load_data
from src.models.utils import data_split, log_details_to_wandb, prediction_scores, pyg_data_split, eval_classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_cores = multiprocessing.cpu_count()
logging.basicConfig(
    #filename="logs/ml_classifiers.txt",
    #filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train(data, model, optimizer, scheduler, criterion, hparams):
    model.train()
    if 'ogb' in hparams.dataset.name:
        out = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.edge_index, data.edge_type)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].reshape(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return loss

def train_mini_batch(train_loader, model, optimizer, scheduler, criterion, device):
    model.train()

    total_loss = total_examples = 0
    total_correct = 0
    for batch in train_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        y = batch.y[batch.train_mask].reshape(-1)
        loss = criterion(out[batch.train_mask], y)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        num_examples = batch.train_mask.sum().item()
        total_loss += loss.item() * num_examples
        total_examples += num_examples

        total_correct += out[batch.train_mask].argmax(dim=-1).eq(y).sum().item()
        total_examples += y.size(0)

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def test(data, model, hparams, evaluator = None):

    model.eval()
    if "ogb" in hparams.dataset.name:
        out = model(data.x, data.edge_index)
        y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1, keepdim=True)

        train_score = evaluator.eval({
            'y_true': data.y[data.train_mask],
            'y_pred': y_pred[data.train_mask],
        })['acc']
        valid_score = evaluator.eval({
            'y_true': data.y[data.valid_mask],
            'y_pred': y_pred[data.valid_mask],
        })['acc']

    else:
        out = model(data.x, data.edge_index, data.edge_type)
        y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1, keepdim=True)
        train_score = accuracy_score(data.y[data.train_mask], y_pred[data.train_mask])
        valid_score = accuracy_score(data.y[data.valid_mask], y_pred[data.valid_mask])

    return train_score, valid_score


@torch.no_grad()
def test_mini_batch(data, model, evaluator, subgraph_loader, device):
    model.eval()

    out = model.inference(data.x, subgraph_loader, device)

    y_true = data.y
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[data.train_mask],
        'y_pred': y_pred[data.train_mask]
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[data.valid_mask],
        'y_pred': y_pred[data.valid_mask]
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[data.test_mask],
        'y_pred': y_pred[data.test_mask]
    })['acc']

    return train_acc, valid_acc, test_acc



@hydra.main(config_path="../config", config_name="default_config.yaml")
def train_GCN(config):

    hparams = config.gcn.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "gcn")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))
    
    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
        num_classes = len(np.unique(data[0].y))
        evaluator = Evaluator(name=hparams.dataset.name)
        model = GCN(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"]).to(device)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
        num_classes = len(np.unique(data.y))
        model = RGCN(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"],
                     hparams["num_bases"], hparams["num_layers"]).to(device)
    log_details_to_wandb("gcn", hparams)
    logging.info("Data loaded.")
    data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
    data.x, data.y = data.x.float(), data.y.long()
    data = data.to(device)

    epochs = hparams["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams["scheduler_step_size"], gamma=0.3, verbose=False)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        loss = train(data, model, optimizer, scheduler, criterion, hparams)

        if "ogb" in hparams.dataset.name:
            train_score, valid_score = test(data, model, hparams, evaluator)
        else:
            train_score, valid_score = test(data, model, hparams)

        logging.info("Epoch %d: Loss: %f, Train Accuracy: %f, Valid Accuracy: %f" % (epoch+1, loss, train_score, valid_score))
        wandb.log({"train_score": train_score, "valid_score": valid_score})
    # Final evaluation on a test set
    model.eval()
    if "ogb" in hparams.dataset.name:
        out = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.edge_index, data.edge_type)
    y_pred = F.log_softmax(out).argmax(dim=-1, keepdim=True)
    y_pred = y_pred.detach().cpu().numpy()
    data = data.cpu()
    test_score, f1, recall, precision = eval_classifier(data.y[data.test_mask], y_pred[data.test_mask])
    logging.info("Test Accuracy: %f,\nF1:\nWeighted: %f, Macro: %f,\nRecall:\nWeighted: %f, Macro: %F,\nPrecision:\nWeighted:%f, Macro: %f" % (test_score,
                                                                                        f1[0], f1[1],
                                                                                        recall[0], recall[1],
                                                                                        precision[0], precision[1]))
    wandb.log({"test_score": test_score,
        "test_f1_weighted": f1[0], "test_f1_macro": f1[1], "test_recall_weighted": recall[0],
        "test_recall_macro": recall[1], "test_prec_weighted": precision[0], "test_prec_macro":precision[1]})
    

@hydra.main(config_path="../config", config_name="default_config.yaml")
def train_GAT(config):
    
    hparams = config.gat.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "gat")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
        num_classes = len(np.unique(data[0].y))
        evaluator = Evaluator(name=hparams.dataset.name)
        model = GAT(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["heads"]).to(device)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
        num_classes = len(np.unique(data.y))
        model = RGAT(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"], hparams["heads"],
                        hparams["num_bases"]).to(device)
    log_details_to_wandb("gat", hparams)
    logging.info("Data loaded.")
    data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
    data.x, data.y = data.x.float(), data.y.long()
    data = data.to(device)

    epochs = hparams["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams["scheduler_step_size"], gamma=hparams["scheduler_gamma"], verbose=False)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
    
        loss = train(data, model, optimizer, scheduler, criterion, hparams)

        if "ogb" in hparams.dataset.name:
            train_score, valid_score = test(data, model, hparams, evaluator)
        else:
            train_score, valid_score = test(data, model, hparams)

        logging.info("Epoch %d: Loss: %f, Train Accuracy: %f, Valid Accuracy: %f" % (epoch+1, loss, train_score, valid_score))
        wandb.log({"train_score": train_score, "valid_score": valid_score})
    # Final evaluation on a test set
    model.eval()
    if "ogb" in hparams.dataset.name:
        out = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.edge_index, data.edge_type)
    y_pred = F.log_softmax(out).argmax(dim=-1, keepdim=True)
    y_pred = y_pred.detach().cpu().numpy()
    data = data.cpu()
    test_score, f1, recall, precision = eval_classifier(data.y[data.test_mask], y_pred[data.test_mask])
    logging.info("Test Accuracy: %f,\nF1:\nWeighted: %f, Macro: %f,\nRecall:\nWeighted: %f, Macro: %F,\nPrecision:\nWeighted:%f, Macro: %f" % (test_score,
                                                                                        f1[0], f1[1],
                                                                                        recall[0], recall[1],
                                                                                        precision[0], precision[1]))
    wandb.log({"test_score": test_score,
        "test_f1_weighted": f1[0], "test_f1_macro": f1[1], "test_recall_weighted": recall[0],
        "test_recall_macro": recall[1], "test_prec_weighted": precision[0], "test_prec_macro":precision[1]})
    


@hydra.main(config_path="../config", config_name="default_config.yaml")
def train_GCN_mini_batch(config):

    hparams = config.gcn.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "gcn")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))
    
    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
        num_classes = len(np.unique(data[0].y))
        evaluator = Evaluator(name=hparams.dataset.name)
        model = GCN(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"]).to(device)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
        num_classes = len(np.unique(data.y))
        model = RGCN(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"], hparams["num_bases"]).to(device)
    log_details_to_wandb("gcn", hparams)
    logging.info("Data loaded.")
    data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
    data.x, data.y = data.x.float(), data.y.long()
    
    cluster_data = ClusterData(data, num_parts=15000,
                               recursive=False)

    loader = ClusterLoader(cluster_data, batch_size=32,
                           shuffle=True, num_workers=n_cores)

    subgraph_loader = NeighborSampler(data.edge_index, sizes=[-1],
                                      batch_size=1024, shuffle=False,
                                      num_workers=n_cores)

    epochs = hparams["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams["scheduler_step_size"], gamma=0.3, verbose=False)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        loss = train_mini_batch(loader, model, optimizer, scheduler, criterion, device)

        if "ogb" in hparams.dataset.name:
            train_score, valid_score = test_mini_batch(data, model, evaluator, subgraph_loader, device)
        else:
            train_score, valid_score = test_mini_batch(data, model, hparams)

        logging.info("Epoch %d: Loss: %f, Train Accuracy: %f, Valid Accuracy: %f" % (epoch+1, loss, train_score, valid_score))
        wandb.log({"train_score": train_score, "valid_score": valid_score})
    # Final evaluation on a test set
    model.eval()
    if "ogb" in hparams.dataset.name:
        out = model(data.x, data.edge_index)
    else:
        out = model(data.x, data.edge_index, data.edge_type)
    y_pred = F.log_softmax(out).argmax(dim=-1, keepdim=True)
    test_score, f1, recall, precision = eval_classifier(data.y[data.test_mask], y_pred[data.test_mask])
    logging.info("Test Accuracy: %f,\nF1:\nWeighted: %f, Macro: %f,\nRecall:\nWeighted: %f, Macro: %F,\nPrecision:\nWeighted:%f, Macro: %f" % (test_score,
                                                                                        f1[0], f1[1],
                                                                                        recall[0], recall[1],
                                                                                        precision[0], precision[1]))
    wandb.log({"test_score": test_score,
        "test_f1_weighted": f1[0], "test_f1_macro": f1[1], "test_recall_weighted": recall[0],
        "test_recall_macro": recall[1], "test_prec_weighted": precision[0], "test_prec_macro":precision[1]})


if __name__ == "__main__":
    
    train_GCN()