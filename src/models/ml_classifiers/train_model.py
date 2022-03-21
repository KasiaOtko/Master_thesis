import argparse
import logging
import sys

import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from omegaconf import DictConfig, OmegaConf

import wandb
import hydra
from src.data.make_dataset import load_data

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


def data_split(data):

    split_idx = data.get_idx_split()

    data = data[0]

    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f"{key}_mask"] = mask

    X_train = data.x[data["train_mask"]].numpy()
    y_train = data.y[data["train_mask"]].numpy().ravel()
    X_valid = data.x[data["valid_mask"]].numpy()
    y_valid = data.y[data["valid_mask"]].numpy().ravel()
    X_test = data.x[data["test_mask"]].numpy()
    y_test = data.y[data["test_mask"]].numpy().ravel()

    return X_train, y_train, X_valid, y_valid, X_test, y_test

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

if __name__ == "__main__":
    logging.info("")
    logging.info("Start.")
    # print(args.root)
    # print(args.dataset_name)

    # data = load_data(args.dataset_name, args.root)
    #data = load_data("ogbn-arxiv", "data/raw")
    # logging.info("Data loaded.")
    wandb.init(project="master-thesis")
    acc_table, final_score, best_model = LogReg()
