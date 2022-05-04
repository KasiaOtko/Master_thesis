import csv
import logging
import multiprocessing

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from GAT_model import GATv2
from GCN_model import GCN, RGCN
from GIN_model import GIN
from GraphSAGE_model import SAGE
from ogb.nodeproppred import Evaluator
from omegaconf import OmegaConf
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import subgraph

import wandb
from src.data.make_dataset import load_data
from src.models.utils import (eval_classifier,
                              log_details_to_wandb,
                              pyg_data_split)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_cores = multiprocessing.cpu_count()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def train(data, model, optimizer, scheduler, criterion, hparams):
    model.train()
    if hparams.model == 'rgcn':
        _, out = model(data.x, data.edge_index, data.edge_type)
    else:
        _, out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask].reshape(-1))
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    y_pred = F.log_softmax(out[data.train_mask], dim=-1).argmax(dim=-1)
    train_score = y_pred.eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()

    return loss, train_score


def train_mini_batch(train_loader, model, optimizer, scheduler, criterion, device, hparams):
    model.train()
    
    total_examples = total_loss = total_correct =  0
    if "ogb" in hparams.dataset.name:
        for batch in train_loader:
            optimizer.zero_grad()
            batch = batch.to(device)
            batch_size = batch.batch_size
            _, out = model(batch.x, batch.edge_index)[:batch_size]
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
            if hparams.model != 'rgcn':
                _, out = model(batch.x, batch.edge_index)[:batch_size]
            else:
                _, out = model(batch.x, batch.edge_index, batch.edge_type)[:batch_size]
            y = batch.y[:batch_size].reshape(-1)
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
def test(data, model, hparams, evaluator = None):

    model.eval()
    if evaluator:
        _, out = model(data.x, data.edge_index)
        y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1)

        valid_score = evaluator.eval({
            'y_true': data.y[data.valid_mask],
            'y_pred': y_pred[data.valid_mask],
        })['acc']

    else:
        if hparams.model != 'rgcn':
            _, out = model(data.x, data.edge_index)
        else:
            _, out = model(data.x, data.edge_index, data.edge_type) 
        y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1)[data.valid_mask]
        y = data.y[data.valid_mask]
        valid_score = y_pred.eq(y).sum().item() / data.valid_mask.sum().item()

    return valid_score


@torch.no_grad()
def test_mini_batch(loader, model, device, hparams, evaluator = None):
    model.eval()
    
    total_examples = total_correct = 0
    y_preds, ys = [], []
    if evaluator:
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.batch_size
            _, out = model(batch.x, batch.edge_index)[:batch_size]
            y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1)
            y = batch.y[:batch_size]

            total_examples += batch_size
            total_correct += int((y_pred == y).sum())

            y_preds.append(y_pred)
            ys.append(y)
        
        score = evaluator.eval({
                'y_true': torch.cat(ys, dim=0),
                'y_pred': torch.cat(y_preds, dim=0)
            })['acc']

    else:
        for batch in loader:
            batch = batch.to(device)
            batch_size = batch.batch_size

            if hparams.model != 'rgcn':
                _, out = model(batch.x, batch.edge_index)[:batch_size]
            else:
                _, out = model(batch.x, batch.edge_index, batch.edge_type)[:batch_size]
            y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1)
            y = batch.y[:batch_size]

            total_examples += batch_size
            total_correct += int((y_pred == y).sum())

            y_preds.append(y_pred)
        score = total_correct / total_examples

    y_preds = torch.cat(y_preds, dim=0)        

    return score, y_preds


def to_inductive(data, hparams):
    data = data.clone()
    mask = data.train_mask
    data.x = data.x[mask]
    data.y = data.y[mask]
    data.train_mask = data.train_mask[mask]
    data.test_mask = None
    if hparams.model != 'rgcn':
        data.edge_index, _ = subgraph(mask, data.edge_index, None,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    else:
        data.edge_index, data.edge_type = subgraph(mask, data.edge_index, data.edge_type,
                                  relabel_nodes=True, num_nodes=data.num_nodes)
    data.num_nodes = mask.sum().item()
    return data


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
            model = GCN(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"],
                        hparams["linear"]).to(device)
        elif hparams["model"] == "gat":
            model = GATv2(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["heads"], 
                            hparams["num_layers"], hparams["linear"]).to(device)
        elif hparams["model"] == "sage":
            model = SAGE(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"], hparams["aggr"], hparams["linear"]).to(device)    

    else:
        data = load_data(hparams.dataset.name, orig_cwd)
        num_classes = len(np.unique(data.y))
        if hparams["model"] == "gcn":
            model = GCN(data.x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"], hparams["linear"]).to(device)
        elif hparams["model"] == "rgcn":
            model = RGCN(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"],
                     hparams["num_bases"], hparams["num_layers"], False).to(device)
        elif hparams["model"] == "gat":
            #model = RGAT(data.x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"]).to(device)
            model = GATv2(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"], hparams["heads"],
                         hparams["num_bases"], hparams["num_layers"], hparams["linear"]).to(device)
        elif hparams["model"] == "sage":
            model = SAGE(data.x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"],
                            hparams["aggr"], hparams["linear"]).to(device)
        elif hparams["model"] == "gin":
            model = GIN(data.x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"]).to(device)

    log_details_to_wandb(hparams["model"], hparams)
    logging.info("Data loaded.")
    data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
    data.x, data.y = data.x.float(), data.y.long()

    epochs = hparams["epochs"]
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["lr"], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=hparams["scheduler_step_size"], gamma=0.3, verbose=False)
    criterion = nn.CrossEntropyLoss()

    if not hparams["inference"]: # TRAIN
        if not hparams["mini_batch"]:
            sampler_data = data
            if hparams["inductive"]:
                sampler_data = to_inductive(data, hparams)
            sampler_data = sampler_data.to(device)
            for epoch in range(epochs):

                loss, train_score = train(sampler_data, model, optimizer, scheduler, criterion, hparams)

                if "ogb" in hparams.dataset.name:
                    valid_score = test(data, model, hparams, evaluator)
                else:
                    valid_score = test(data, model, hparams)

                logging.info("Epoch %d: Loss: %f, Train Accuracy: %f, Valid Accuracy: %f" % (epoch+1, loss, train_score, valid_score))
                wandb.log({"train_score": train_score, "valid_score": valid_score})

            torch.save(model.state_dict(), orig_cwd + "/models/{}_{}.pt".format(hparams.model, hparams.dataset.name))
            # Final evaluation on a test set
            model.eval()

            if hparams.model == 'rgcn':
                _, out = model(data.x, data.edge_index, data.edge_type)
            else:
                _, out = model(data.x, data.edge_index)
            y_pred = F.log_softmax(out, dim=-1).argmax(dim=-1, keepdim=True)
            y_pred = y_pred.detach().cpu().numpy()[data.test_mask]

        else:
            data.num_nodes = data.x.shape[0]
            data.n_id = torch.arange(data.num_nodes)
            sampler_data = data
            if hparams["inductive"]:
                sampler_data = to_inductive(data, hparams)
            train_loader = NeighborLoader(sampler_data, num_neighbors=[10]+[5]*(hparams["num_layers"]-1), shuffle=True,
                                input_nodes=None, batch_size = hparams["batch_size"], num_workers = n_cores)
            valid_loader = NeighborLoader(data, num_neighbors=[-1]*(hparams["num_layers"]),
                                input_nodes=data.valid_mask, batch_size = hparams["batch_size"], num_workers = n_cores)
            test_loader = NeighborLoader(data, num_neighbors=[-1]*(hparams["num_layers"]),
                                input_nodes=data.test_mask, batch_size = hparams["batch_size"], num_workers = n_cores)

            for epoch in range(epochs):
                loss, train_score = train_mini_batch(train_loader, model, optimizer, scheduler, criterion, device, hparams)

                if "ogb" in hparams.dataset.name:
                    valid_score, _ = test_mini_batch(valid_loader, model, device, hparams, evaluator)
                else:
                    valid_score, _ = test_mini_batch(valid_loader, model, device, hparams)

                logging.info("Epoch %d: Loss: %f, Train Accuracy: %f, Valid Accuracy: %f" % (epoch+1, loss, train_score, valid_score))
                wandb.log({"train_score": train_score, "valid_score": valid_score})

            torch.save(model.state_dict(), orig_cwd + "/models/{}_{}.pt".format(hparams.model, hparams.dataset.name))
            # Final evaluation on a test set
            if "ogb" in hparams.dataset.name:
                valid_score, y_pred = test_mini_batch(test_loader, data, model, device, data.test_mask, evaluator)
            else:
                valid_score, y_pred = test_mini_batch(test_loader, data, model, device, data.test_mask)        

        test_score, f1, recall, precision = eval_classifier(data.y[data.test_mask], y_pred)
        logging.info("Test Accuracy: %f,\nF1:\nWeighted: %f, Macro: %f,\nRecall:\nWeighted: %f, Macro: %F,\nPrecision:\nWeighted:%f, Macro: %f" % (test_score,
                                                                                            f1[0], f1[1],
                                                                                            recall[0], recall[1],
                                                                                            precision[0], precision[1]))
        wandb.log({"test_score": test_score,
            "test_f1_weighted": f1[0], "test_f1_macro": f1[1], "test_recall_weighted": recall[0],
            "test_recall_macro": recall[1], "test_prec_weighted": precision[0], "test_prec_macro":precision[1]})

    else: # INFERENCE
        test_score_best = 0
        for i in range(10):
            print("Iteration", i)
            if "ogb" in hparams.dataset.name:
                data = load_data(hparams.dataset.name, orig_cwd + config.root)
                num_classes = len(np.unique(data[0].y))
                evaluator = Evaluator(name=hparams.dataset.name)
                if hparams["model"] == "gcn":
                    model = GCN(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"],
                                hparams["linear"]).to(device)
                elif hparams["model"] == "gat":
                    model = GATv2(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["heads"], 
                                    hparams["num_layers"], hparams["linear"]).to(device)
                elif hparams["model"] == "sage":
                    model = SAGE(data[0].x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"], hparams["aggr"], hparams["linear"]).to(device)    

            else:
                data = load_data(hparams.dataset.name, orig_cwd)
                num_classes = len(np.unique(data.y))
                if hparams["model"] == "gcn":
                    model = GCN(data.x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"], hparams["linear"]).to(device)
                if hparams["model"] == "rgcn":
                    model = RGCN(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"],
                             hparams["num_bases"], hparams["num_layers"], False).to(device)
                elif hparams["model"] == "gat":
                    #model = RGAT(data.x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"]).to(device)
                    model = GATv2(data.x.shape[1], hparams["hidden_channels"], num_classes, 2, hparams["dropout_p"], hparams["heads"],
                                hparams["num_bases"], hparams["num_layers"], hparams["linear"]).to(device)
                elif hparams["model"] == "sage":
                    model = SAGE(data.x.shape[1], hparams["hidden_channels"], num_classes, hparams["dropout_p"], hparams["num_layers"],
                                    hparams["aggr"], hparams["linear"]).to(device)

            data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
            data.x, data.y = data.x.float(), data.y.long()

            model.load_state_dict(torch.load(orig_cwd + "/models/{}_{}.pt".format(hparams.model, hparams.dataset.name), map_location=device))
            model.eval()
            if hparams.model == 'rgcn':
                h, out = model(data.x, data.edge_index, data.edge_type)
            else:
                h, out = model(data.x, data.edge_index)
            y_pred_full = F.log_softmax(out, dim=-1).argmax(dim=-1, keepdim=True)
            y_pred_test = y_pred_full.detach().cpu().numpy()[data.test_mask]

            test_score, f1, recall, precision = eval_classifier(data.y[data.test_mask], y_pred_test)

            if test_score > test_score_best:
                test_score_best = test_score
                np.save(orig_cwd + "/models/embeddings/{}_{}".format(hparams.model, hparams.dataset.name), h.detach().cpu().numpy())
                np.save(orig_cwd + "/models/preds/{}_{}".format(hparams.model, hparams.dataset.name), y_pred_full)
                np.save(orig_cwd + "/models/targets/{}_{}".format(hparams.model, hparams.dataset.name), data.y.detach().cpu().numpy())

            wandb.log({"test_score": test_score,
        "test_f1_weighted": f1[0], "test_f1_macro": f1[1], "test_recall_weighted": recall[0],
        "test_recall_macro": recall[1], "test_prec_weighted": precision[0], "test_prec_macro":precision[1]})
            data = ["{}_{}".format(hparams.model, i), hparams.dataset.name, hparams.dataset.random_split,
                    0, 0, test_score,
                    f1[0], f1[1], recall[0], recall[1], precision[0], precision[1]]
            with open(orig_cwd + '/logs/gnns.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)


if __name__ == "__main__":
    
    train_GCN()