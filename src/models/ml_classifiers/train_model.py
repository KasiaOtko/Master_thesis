import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import logging

from src.data.make_dataset import load_data

import wandb
wandb.init(project="master-thesis")


parser = argparse.ArgumentParser()
parser.add_argument("dataset_name", choices=['ogbn-products', 'ogbn-arxiv', 'EU-judgements'],
                    help = "Name of the dataset to use (possible: 'ogbn-products', 'ogbn-arxiv', 'EU-judgements')")
parser.add_argument("--root", default = 'data/raw',
                    help="Root directory of a folder storing OGB dataset")
parser.add_argument("--model", choices = ["lr", "rf", "xgb"],
                    help="Model to tune")

logging.basicConfig(filename="logs/ml_classifiers/logfile.txt",
                    filemode='a',
                    level=logging.INFO, 
                    format='%(asctime)s %(levelname)-8s %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')                 

#args = parser.parse_args()

def data_split(data):

    split_idx = data.get_idx_split()

    data = data[0]

    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        data[f'{key}_mask'] = mask

    X_train = data.x[data['train_mask']].numpy()
    y_train = data.y[data['train_mask']].numpy().ravel()
    X_valid = data.x[data['valid_mask']].numpy()
    y_valid = data.y[data['valid_mask']].numpy().ravel()
    X_test = data.x[data['test_mask']].numpy()
    y_test = data.y[data['test_mask']].numpy().ravel()

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def LogReg(data, scale = True):

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data)

    if scale:
        scaler = StandardScaler()
        scaler = scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

    C_range = [10, 100] #[0.01,0.1,1,10,100, 100]
    acc_table = pd.DataFrame(columns = ['Train_accuracy', 'Valid_accuracy', 'Test accuracy'], index = [0.001,0.01,0.1,1,10,100])
    #acc_table["C_parameter"] = C_range

    for C in C_range:
        lr = LogisticRegression(random_state=0, C = C, max_iter = 300).fit(X_train, y_train)
        y_train_pred = lr.predict(X_train)
        y_valid_pred = lr.predict(X_valid)
        acc_table.loc[C, "Train_accuracy"] = accuracy_score(y_train, y_train_pred)
        acc_table.loc[C, "Valid_accuracy"] = accuracy_score(y_valid, y_valid_pred)

    wandb_table = wandb.Table(dataframe=acc_table)
    wandb.log({'acc_table': wandb_table})

    C_final = pd.to_numeric(acc_table[~acc_table["Valid_accuracy"].isna()]["Valid_accuracy"]).idxmax()
    lr = LogisticRegression(random_state=0, C = int(C_final), max_iter = 300).fit(X_train, y_train)
    y_test_pred = lr.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    wandb.log({'test_accuracy': test_accuracy})

    return acc_table, test_accuracy, lr

if __name__ == "__main__":

    #print(args.root)
    #print(args.dataset_name)
    
    #data = load_data(args.dataset_name, args.root)
    data = load_data('ogbn-arxiv', 'data/raw')
    acc_table, final_score, best_model = LogReg(data)

    

    



