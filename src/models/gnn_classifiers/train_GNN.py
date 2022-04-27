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
from torch_geometric.loader import NeighborSampler, NeighborLoader, ClusterData, ClusterLoader
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

def train_mini_batch(train_loader, model, optimizer, scheduler, criterion, device, hparams):
    model.train()
    
    total_examples = total_loss = total_correct =  0
    # y_pred = torch.Tensor()
    if "ogb" in hparams.dataset.name:
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            batch_size = batch.batch_size
            out = model(batch.x, batch.edge_index)[:batch_size]
            y = batch.y[:batch_size]
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size
            y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1)
            total_correct += y_pred.eq(y).sum().item()
    else:
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            batch_size = batch.batch_size
            out = model(batch.x, batch.edge_index, batch.edge_type)[:batch_size]
            y = batch.y[:batch_size]
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size
            y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1)
            total_correct += y_pred.eq(y).sum().item()

    full_loss = total_loss / total_examples
    train_score = total_correct / total_examples
    return full_loss, train_score

@torch.no_grad()
def test(data, model, evaluator = None):

    model.eval()
    #if "ogb" in hparams.dataset.name:
    if evaluator:
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
def test_mini_batch(loader, data, model, device, mask, evaluator = None):
    model.eval()
    
    total_examples = total_correct = 0
    y_preds = []
    if evaluator:
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.batch_size,

            out = model(batch.x, batch.edge_index)[:batch_size]
            y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1)
            y = batch.y[:batch_size]

            total_examples += batch_size
            total_correct += int((y_pred == y).sum())

            y_preds.append(y_pred)

        score = evaluator.eval({
                'y_true': data.y[mask],
                'y_pred': y_preds,
            })['acc']

    else:
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.batch_size

            out = model(batch.x, batch.edge_index, batch.edge_type)[:batch_size]
            y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1)
            y = batch.y[:batch_size]

            total_examples += batch_size
            total_correct += int((y_pred == y).sum())

            y_preds.append(y_pred)
        score = total_correct / total_examples

    y_preds = torch.cat(y_preds, dim=0)        

    return score, y_preds


@hydra.main(config_path="../config", config_name="default_config.yaml")
def train_GCN(config):

    hparams = config.gcn.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = hparams["model"])
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))
    
    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
        num_classes = len(np.unique(data[0].y))
        evaluator = Evaluator(name=hparams.dataset.name)
        if hparams["model"] == "gcn":
            model = GCN(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"]).to(device)
        elif hparams["model"] == "gat":
            model = GAT(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["heads"], hparams["num_layers"]).to(device)

    else:
        data = load_data(hparams.dataset.name, orig_cwd)
        num_classes = len(np.unique(data.y))
        if hparams["model"] == "gcn":
            model = RGCN(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"],
                     hparams["num_bases"], hparams["num_layers"]).to(device)
        elif hparams["model"] == "gat":
            model = RGAT(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"], hparams["heads"],
                        hparams["num_bases"], hparams["num_layers"]).to(device)

    log_details_to_wandb(hparams["model"], hparams)
    logging.info("Data loaded.")
    data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
    data.x, data.y = data.x.float(), data.y.long()
    data = data.to(device)

    epochs = hparams["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams["scheduler_step_size"], gamma=0.3, verbose=False)
    criterion = nn.CrossEntropyLoss()

    if not hparams["mini_batch"]:
        for epoch in range(epochs):

            loss, train_score = train(data, model, optimizer, scheduler, criterion, hparams)

            if "ogb" in hparams.dataset.name:
                train_score, valid_score = test(data, model, evaluator)
            else:
                train_score, valid_score = test(data, model)

            logging.info("Epoch %d: Loss: %f, Train Accuracy: %f, Valid Accuracy: %f" % (epoch+1, loss, train_score, valid_score))
            wandb.log({"train_score": train_score, "valid_score": valid_score})
        # Final evaluation on a test set
        model.eval()

        if "ogb" in hparams.dataset.name:
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.edge_index, data.edge_type)
        y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1, keepdim=True)
        y_pred = y_pred.detach().cpu().numpy()[data.test_mask]

    else:
        data.num_nodes = data.x.shape[0]
        data.n_id = torch.arange(data.num_nodes)
        train_loader = NeighborLoader(data, num_neighbors=[10]+[5]*(hparams["num_layers"]-1), shuffle=True,
                              input_nodes=data.train_mask, batch_size = hparams["batch_size"], num_workers = n_cores)
        val_loader = NeighborLoader(data, num_neighbors=[10]+[5]*(hparams["num_layers"]-1),
                            input_nodes=data.valid_mask, batch_size = hparams["batch_size"], num_workers = n_cores)
        test_loader = NeighborLoader(data, num_neighbors=[10]+[5]*(hparams["num_layers"]-1),
                            input_nodes=data.test_mask, batch_size = hparams["batch_size"], num_workers = n_cores)

        for epoch in range(epochs):
            loss, train_score = train_mini_batch(train_loader, model, optimizer, scheduler, criterion, device, hparams)

            if "ogb" in hparams.dataset.name:
                valid_score, _ = test_mini_batch(val_loader, data, model, device, data.valid_mask, evaluator)
            else:
                valid_score, _ = test_mini_batch(val_loader, data, model, device, data.valid_mask)

            logging.info("Epoch %d: Loss: %f, Train Accuracy: %f, Valid Accuracy: %f" % (epoch+1, loss, train_score, valid_score))
            wandb.log({"train_score": train_score, "valid_score": valid_score})

        # Final evaluation on a test set
        if "ogb" in hparams.dataset.name:
            valid_score, y_pred = test_mini_batch(test_loader, data, model, device, data.test_mask, evaluator)
        else:
            valid_score, y_pred = test_mini_batch(test_loader, data, model, device, data.test_mask)        

    y_pred = y_pred.cpu()
    data = data.cpu()
    test_score, f1, recall, precision = eval_classifier(data.y[data.test_mask], y_pred)
    logging.info("Test Accuracy: %f,\nF1:\nWeighted: %f, Macro: %f,\nRecall:\nWeighted: %f, Macro: %F,\nPrecision:\nWeighted:%f, Macro: %f" % (test_score,
                                                                                        f1[0], f1[1],
                                                                                        recall[0], recall[1],
                                                                                        precision[0], precision[1]))
    wandb.log({"test_score": test_score,
        "test_f1_weighted": f1[0], "test_f1_macro": f1[1], "test_recall_weighted": recall[0],
        "test_recall_macro": recall[1], "test_prec_weighted": precision[0], "test_prec_macro":precision[1]})
        

if __name__ == "__main__":
    
    train_GCN()