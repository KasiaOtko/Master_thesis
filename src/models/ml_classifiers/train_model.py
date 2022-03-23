import argparse
from audioop import mul
import logging
import sys
import multiprocessing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from omegaconf import DictConfig, OmegaConf

import wandb
import hydra
from src.data.make_dataset import load_data
from src.models.utils import data_split, log_details_to_wandb
from models.FFNN_model import FFNNClassifier

sys.path.append("..")
n_cores = multiprocessing.cpu_count()

parser = argparse.ArgumentParser()
# parser.add_argument(
    # "dataset_name",
    # choices=["ogbn-products", "ogbn-arxiv", "EU-judgements"],
    # help="Name of the dataset to use (possible: 'ogbn-products', 'ogbn-arxiv', 'EU-judgements')",
# )
# parser.add_argument(
#     "--root", default="data/raw", help="Root directory of a folder storing OGB dataset"
# )
# parser.add_argument("--model", choices=["lr", "ffnn", "xgb"], help="Model to tune")

# parser.add_argument("--lr_C_range", nargs="*", type=float, default=[0.01,0.1,1,10,100,1000], 
#                     help="Regularization strength for Logistic Regression model")

logging.basicConfig(
    filename="logs/ml_classifiers.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

args = parser.parse_args()

@hydra.main(config_path="../config", config_name="default_config.yaml")
def LogReg(config : DictConfig):
    
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.log_reg.hyperparameters
    wandb.config = hparams #.update(args)
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(config["log_reg"]["hyperparameters"]))

    data = load_data("ogbn-arxiv", orig_cwd + "/data/raw")
    logging.info("Data loaded.")

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data,
                                                                    hparams["scale"],
                                                                    random_split = hparams["random_split"],
                                                                    stratify = hparams["stratify"])

    C_range = hparams["C_range"]
    acc_table = pd.DataFrame(
        columns = ["Train_accuracy", "Valid_accuracy", "Test accuracy"],
        index = C_range,
    )

    for C in C_range:
        lr = LogisticRegression(random_state=hparams["random_state"], C=C, max_iter=hparams["max_iter"], n_jobs = n_cores).fit(X_train, y_train)
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
    wandb.init(project="master-thesis", config = hparams)
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    data = load_data(config.ffnn.dataset.name, orig_cwd + config.root) # "/data/raw"
    # wandb.log({"dataset": config.ffnn.dataset.name, "random_split": config.ffnn.dataset.random_split})
    log_details_to_wandb("ffnn", config)
    logging.info("Data loaded.")

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data, 
                                                                    hparams["scale"], 
                                                                    to_numpy=False, 
                                                                    random_split = config.ffnn.dataset.random_split, 
                                                                    stratify = config.ffnn.dataset.stratify)

    num_classes = len(np.unique(y_train))

    model = FFNNClassifier(X_train.shape[1], hparams["num_hidden1"], hparams["num_hidden2"], hparams["num_hidden3"], num_classes, hparams["dropout_p"])

    optimizer = optim.Adam(model.parameters(), lr=hparams["lr"], betas = (0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    # setting hyperparameters and gettings epoch sizes
    batch_size = hparams["batch_size"]
    epochs = hparams["epochs"]
    num_samples_train = X_train.shape[0]
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = X_valid.shape[0]
    num_batches_valid = num_samples_valid // batch_size
    num_samples_test = X_test.shape[0]
    num_batches_test = num_samples_test // batch_size

    # setting up lists for handling loss/accuracy
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

        train_losses.append((cur_loss / batch_size).detach().numpy()) # average loss of all batches

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
        valid_losses.append((cur_loss / batch_size).detach().numpy())

        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(val_targs, val_preds)
        
        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)
        
        # if epoch % 10 == 0:
        logging.info("Epoch %2i : Train loss %f, Valid loss %f, Train acc %f, Valid acc %f" % (
                    epoch+1, train_losses[-1], valid_losses[-1], train_acc_cur, valid_acc_cur))

        wandb.log({"ffnn_train_loss": train_losses[-1].item(), "ffnn_train_acc": train_acc_cur, "ffnn_valid_acc": valid_acc_cur})

    # Evaluate final model on the test set
    test_targs, test_preds = [], []
    cur_loss = 0
    for i in range(num_batches_test):
        slce = get_slice(i, batch_size)
        
        output = model(X_test[slce])

        y_batch = y_test[slce]
        batch_loss = criterion(output, y_batch)
        cur_loss += batch_loss

        preds = torch.max(output, 1)[1]
        test_targs += list(y_test[slce].numpy())
        test_preds += list(preds.data.numpy())
    test_acc = accuracy_score(test_targs, test_preds)
    logging.info("Test set evaluation: Loss %f, Accuracy %f" % (cur_loss/batch_size, test_acc))
    wandb.log({"Test accuracy": test_acc})

    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), train_acc, 'r', range(epochs), valid_acc, 'b')
    plt.legend(['Train Accucary','Validation Accuracy'])
    plt.xlabel('Epochs'), plt.ylabel('Accuracy');
    plt.savefig(f"{orig_cwd}/reports/figures/FFNN_accuracy_curve.png")
    wandb.log({"accuracy_curve": wandb.Image(fig)})

    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), train_losses, 'r', range(epochs), valid_losses, 'b')
    plt.legend(['Train Loss','Validation Loss'])
    plt.xlabel('Epochs'), plt.ylabel('Loss');
    plt.savefig(f"{orig_cwd}/reports/figures/FFNN_loss_curve.png")
    wandb.log({"loss_curve": wandb.Image(fig)})

@hydra.main(config_path="../config", config_name="default_config.yaml")
def RandomForest(config):

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.rf.hyperparameters
    wandb.config = hparams
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(config["rf"]["hyperparameters"]))

    data = load_data(config.rf.dataset.name, orig_cwd + "/data/raw")
    wandb.log({"dataset": config.rf.dataset.name})
    logging.info("Data loaded.")

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data, 
                                                                    hparams["scale"], 
                                                                    to_numpy=False, 
                                                                    random_split = True, 
                                                                    stratify = config.rf.dataset.stratify)

    n_estimators = [int(x) for x in np.linspace(start = 500, stop = 2000, num = 4)]
    max_features = ['sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(50, 110, num = 6)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    bootstrap = [True, False]

    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
 
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, verbose=2, random_state=0,
                                n_jobs = n_cores)

    rf_random.fit(X_train, y_train)
    

if __name__ == "__main__":

    
    train_Nnet()
