import argparse
import logging
import sys

import pandas as pd
import numpy as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig, OmegaConf

import wandb
import hydra
from src.data.make_dataset import load_data
from src.models.utils import data_split
from models.FFNN_model import FFNNClassifier

sys.path.append("..")

parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset_name",
    choices=["ogbn-products", "ogbn-arxiv", "EU-judgements"],
    help="Name of the dataset to use (possible: 'ogbn-products', 'ogbn-arxiv', 'EU-judgements')",
)
parser.add_argument(
    "--root", default="data/raw", help="Root directory of a folder storing OGB dataset"
)
parser.add_argument("--model", choices=["lr", "svm", "xgb"], help="Model to tune")

parser.add_argument("--lr_C_range", nargs="*", type=float, default=[0.01,0.1,1,10,100,1000], 
                    help="Regularization strength for Logistic Regression model")

logging.basicConfig(
    filename="logs/ml_classifiers.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# args = parser.parse_args()

@hydra.main(config_path="../config", config_name="default_config.yaml")
def LogReg(config : DictConfig):
    
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.log_reg.hyperparameters
    wandb.config = hparams #.update(args)
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(config["log_reg"]["hyperparameters"]))

    data = load_data("ogbn-arxiv", orig_cwd + "/data/raw")
    logging.info("Data loaded.")

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data)

    if hparams["scale"]:
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    C_range = hparams["C_range"]  # [0.01,0.1,1,10,100, 1000]
    acc_table = pd.DataFrame(
        columns = ["Train_accuracy", "Valid_accuracy", "Test accuracy"],
        index = C_range,
    )
    # acc_table["C_parameter"] = C_range

    for C in C_range:
        lr = LogisticRegression(random_state=hparams["random_state"], C=C, max_iter=hparams["max_iter"]).fit(X_train, y_train)
        y_train_pred = lr.predict(X_train)
        y_valid_pred = lr.predict(X_valid)
        acc_table.loc[C, "Train_accuracy"] = pd.to_numeric(
            accuracy_score(y_train, y_train_pred)
        )
        acc_table.loc[C, "Valid_accuracy"] = pd.to_numeric(
            accuracy_score(y_valid, y_valid_pred)
        )
        logging.info(
            "C = {0}: Train_accuracy = {1}, Validation accuracy = {2}".format(
            C,
            acc_table.loc[C, "Train_accuracy"],
            acc_table.loc[C, "Valid_accuracy"])
        )

    wandb_table = wandb.Table(dataframe=acc_table)
    wandb.log({"acc_table": wandb_table})

    C_final = pd.to_numeric(
        acc_table[~acc_table["Valid_accuracy"].isna()]["Valid_accuracy"]
    ).idxmax()
    lr = LogisticRegression(random_state=hparams["random_state"], C=C_final, max_iter=hparams["max_iter"])
    lr.fit(X_train, y_train)
    y_test_pred = lr.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    wandb.log({"test_accuracy": test_accuracy})
    logging.info("Finish")

    # maybe consider saving the df here

    return acc_table, test_accuracy, lr

@hydra.main(config_path="../config", config_name="default_config.yaml")
def train_Nnet(config):

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.ffnn.hyperparameters
    wandb.config = hparams
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(config["ffnn"]["hyperparameters"]))

    data = load_data(config.ffnn.dataset.name, orig_cwd + "/data/raw")
    logging.info("Data loaded.")

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data, to_tensor=True)
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train).long()
    X_valid = torch.Tensor(X_valid)
    y_valid = torch.Tensor(y_valid).long()

    num_classes = len(np.unique(y_train))

    model = FFNNClassifier(X_train.shape[1], hparams["num_hidden"], len(num_classes))

    optimizer = optim.Adam(model.parameters(), lr=0.01, betas = (0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    # setting hyperparameters and gettings epoch sizes
    batch_size = 500
    num_epochs = 10
    num_samples_train = X_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = X_valid.shape[0]
    num_batches_valid = num_samples_valid // batch_size

    # setting up lists for handling loss/accuracy
    train_acc, train_loss = [], []
    valid_acc, valid_loss = [], []
    test_acc, test_loss = [], []
    cur_loss = 0
    losses = []

    get_slice = lambda i, size: range(i * size, (i + 1) * size)

    for epoch in range(num_epochs):
        # Forward -> Backprob -> Update params
        ## Train
        cur_loss = 0
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
        losses.append(cur_loss / batch_size) # average loss of all batches

        model.eval()
        ### Evaluate training
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            output = model(X_train[slce])
            
            preds = torch.max(output, 1)[1]
            
            train_targs += list(y_train[slce].numpy())
            train_preds += list(preds.data.numpy())
        
        ### Evaluate validation
        val_preds, val_targs = [], []
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            
            output = model(X_valid[slce])
            preds = torch.max(output, 1)[1]
            val_targs += list(y_valid[slce].numpy())
            val_preds += list(preds.data.numpy()) 

        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(val_targs, val_preds)
        
        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
        
        if epoch % 10 == 0:
            print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                    epoch+1, losses[-1], train_acc_cur, valid_acc_cur))

if __name__ == "__main__":
    logging.info("")
    logging.info("Start.")
    # print(args.root)
    # print(args.dataset_name)

    # data = load_data(args.dataset_name, args.root)
    #data = load_data("ogbn-arxiv", "data/raw")
    # logging.info("Data loaded.")
    wandb.init(project="master-thesis")
    # acc_table, final_score, best_model = LogReg()
    train_Nnet()
