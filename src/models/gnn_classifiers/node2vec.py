import csv
import logging
import multiprocessing
import sys
from cgi import test

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score
from torch_geometric.nn import Node2Vec

import wandb
from models.FFNN_model import FFNNClassifier
from src.data.make_dataset import load_data
from src.models.utils import (data_split, draw_learning_curve,
                              log_details_to_wandb, pyg_data_split)

sys.path.append("..")
n_cores = multiprocessing.cpu_count()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_with_neural_net(data, z, hparams, orig_cwd):

    num_classes = len(np.unique(data.y))

    model = FFNNClassifier(z.shape[1], 
                            hparams["num_hidden1"], 
                            hparams["num_hidden2"], 
                            hparams["num_hidden3"], 
                            num_classes, 
                            hparams["dropout_p"])

    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"], betas = (0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    z = z.detach().numpy()

    X_train = torch.from_numpy(z[data.train_mask]).float()
    print("Is cuda?", X_train.is_cuda)
    y_train = data.y[data.train_mask].ravel()
    X_valid = torch.from_numpy(z[data.valid_mask]).float()
    y_valid = data.y[data.valid_mask].ravel()
    X_test = torch.from_numpy(z[data.test_mask]).float()
    y_test = data.y[data.test_mask].ravel()
                                                                    

    batch_size = hparams["batch_size"]
    epochs = hparams["epochs"]
    num_samples_train = X_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = X_valid.shape[0]
    num_batches_valid = num_samples_valid // batch_size
    num_samples_test = X_test.shape[0]
    num_batches_test = num_samples_test // batch_size

    train_acc, train_loss = [], []
    valid_acc, valid_loss = [], []
    test_acc, test_loss = [], []
    cur_loss = 0
    train_losses = []
    valid_losses = []

    get_slice = lambda i, size: range(i * size, (i + 1) * size)

    for epoch in range(epochs):
        # Forward -> Backprob -> Update params
        ## Train
        cur_loss = 0
        train_preds, train_targs = [], []
        model.train()
        for i in range(num_batches_train):  # iterate over each batch
            optimizer.zero_grad()
            slce = get_slice(i, batch_size) # get the batch
            output = model(X_train[slce])     # forward pass
            
            # compute gradients given loss
            y_batch = y_train[slce]
            batch_loss = criterion(output, y_batch) # compute loss
            batch_loss.backward()                        # backward pass
            optimizer.step()                             # update parameters
            
            cur_loss += batch_loss
            
            # Make predictions (evaluate training)
            preds = torch.max(output, 1)[1]
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())

        train_losses.append((cur_loss / num_batches_train).detach().numpy()) # average loss of all batches

        model.eval()        
        ### Evaluate validation
        val_preds, val_targs = [], []
        cur_loss = 0
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = model(X_valid[slce])

            y_batch = y_valid[slce]
            batch_loss = criterion(output, y_batch)

            preds = torch.max(output, 1)[1]
            val_targs += list(y_valid[slce].numpy())
            val_preds += list(preds.data.numpy())
        
            cur_loss += batch_loss
        valid_losses.append((cur_loss / num_batches_valid).detach().numpy())

        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(val_targs, val_preds)
        
        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
        
        # if epoch % 10 == 0:
        logging.info("Epoch %2i : Train loss %f, Valid loss %f, Train acc %f, Valid acc %f" % (
                    epoch+1, train_losses[-1], valid_losses[-1], train_acc_cur, valid_acc_cur))

        wandb.log({"ffnn_train_loss": train_losses[-1].item(), "ffnn_train_acc": train_acc_cur, "ffnn_valid_acc": valid_acc_cur})

    # Evaluate final model on the test set
    # if ~hparams.dataset.random_split:
    #     y_test, X_test = remove_outstanding_classes_from_testset(y_test, X_test)
    # num_batches_test = X_test.shape[0] // batch_size
    test_targs, test_preds = [], []
    cur_loss = 0
    with torch.no_grad():
        for i in range(num_batches_test):
            slce = get_slice(i, batch_size)
            
            output = model(X_test[slce])

            y_batch = y_test[slce]
            batch_loss = criterion(output, y_batch)
            cur_loss += batch_loss

            preds = torch.max(output, 1)[1]
            test_targs += list(y_test[slce].numpy()) # 
            test_preds += list(preds.data.numpy()) # 

    test_acc = accuracy_score(test_targs, test_preds)
    logging.info("Test set evaluation: Loss %f, Accuracy %f" % (cur_loss/num_batches_test, test_acc))
    wandb.log({"train_score": train_acc_cur, "valid_score": valid_acc_cur, "test_score": test_acc})

    draw_learning_curve(epochs, train_losses, valid_losses, "Loss", orig_cwd)
    draw_learning_curve(epochs, train_acc, valid_acc, "Accuracy", orig_cwd)

    return train_acc[-1], valid_acc[-1], test_acc


@hydra.main(config_path="../config", config_name="default_config.yaml")
def run_vode2vec(config):

    hparams = config.node2vec.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "node2vec")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
    log_details_to_wandb("node2vec", hparams)
    logging.info("Data loaded.")

    data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
    print("Tran nodes:", data.train_mask.sum())
    if hparams.inference:
        if "ogb" in hparams.dataset.name:
            data = load_data(hparams.dataset.name, orig_cwd + config.root)
        else:
            data = load_data(hparams.dataset.name, orig_cwd)
        # s = torch.load(orig_cwd + "/models/node2vec_ogbn-products.pt", map_location=device)
        model = Node2Vec(data.edge_index, embedding_dim=hparams["embedding_dim"], 
                    walk_length=hparams["walk_length"],                        # lenght of rw
                    context_size=hparams["context_size"], walks_per_node=hparams["walks_per_node"],
                    num_negative_samples=hparams["num_negative_samples"], 
                    p=hparams["p"], q=hparams["q"],                             # bias parameters
                    sparse=True).to(device)

        model.load_state_dict(torch.load(orig_cwd + "/models/node2vec_ogbn-products.pt", map_location=device))
        model.eval()

        z = model()
        net_hparams = config.ffnn.hyperparameters
        train_score, valid_score, test_score = eval_with_neural_net(data, z, net_hparams, orig_cwd)
        wandb.log({"train_score": train_score, "valid_score": valid_score, "test_score": test_score})

        data = ["n2v", hparams.dataset.name, hparams.dataset.random_split, train_score, valid_score, test_score]
        with open(orig_cwd + '/logs/n2v_{}.csv'.format(hparams.dataset.name), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
    
    else:
        
        model = Node2Vec(data.edge_index, embedding_dim=hparams["embedding_dim"], 
                    walk_length=hparams["walk_length"],                        # lenght of rw
                    context_size=hparams["context_size"], walks_per_node=hparams["walks_per_node"],
                    num_negative_samples=hparams["num_negative_samples"], 
                    p=hparams["p"], q=hparams["q"],                             # bias parameters
                    sparse=True).to(device)

        loader = model.loader(batch_size=hparams["batch_size"], # we generate 128*20 random walks in every batch
                        shuffle=True, num_workers=n_cores)

        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=hparams["lr"])

        epochs = hparams["epochs"]
        train_losses = []

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device)) # compute loss using positive and negative random walks
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            train_losses.append(total_loss / len(loader))

            if epochs+1 % 100 == 0:
                model.eval()
                z = model() # computes embeddings of the model [N, embedding_dim]
                train_score = model.test(z[data.train_mask], data.y[data.train_mask].ravel(),
                                z[data.train_mask], data.y[data.train_mask].ravel(),
                                max_iter=150)

                valid_score = model.test(z[data.train_mask], data.y[data.train_mask].ravel(),
                                z[data.valid_mask], data.y[data.valid_mask].ravel(),
                                max_iter=150) # in the test function, logistic regression is implemented to test the embeddings

                logging.info("Epoch %2i : Train loss %f, Train Accuracy %f, Validation Accuracy %f" % (
                            epoch+1, train_losses[-1], train_score, valid_score))

                wandb.log({"train_loss": train_losses[-1], "train_score": train_score, "valid_score": valid_score})
        
        # test_score = model.test(z[data.train_mask], data.y[data.train_mask].ravel(),
        #                     z[data.test_mask], data.y[data.test_mask].ravel(),
        #                     max_iter=150)
        torch.save(model.state_dict(), orig_cwd + "/models/node2vec_{}.pt".format(hparams.dataset.name))
        net_hparams = config.ffnn.hyperparameters
        train_score, valid_score, test_score = eval_with_neural_net(data, z, net_hparams, orig_cwd)
        wandb.log({"train_score": train_score, "valid_score": valid_score, "test_score": test_score})
    

if __name__ == "__main__":
    
    run_vode2vec()
    # data = ["n2v", "ogbn-products", True, 0.5, 0.55, 0.6]
    # with open('logs/n2v_{}.csv'.format("ogbn-products"), 'a', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(data)
