import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import wandb

def data_split(data, dataset, scale = False, to_numpy = False, random_split = False, stratify = False):
    
    if "ogb" in dataset:

        split_idx = data.get_idx_split()
        data = data[0]
        
        if random_split: # according to the i.i.d. assumption
            if stratify:
                X_train, X_test, y_train, y_test = train_test_split(data.x, data.y, test_size=0.2, random_state=0, stratify = data.y)
                X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify = y_train) # 0.25 x 0.8 = 0.2
            else:
                X_train, X_test, y_train, y_test = train_test_split(data.x, data.y, test_size=0.2, random_state=0)
                X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0.2

        else: # Use the splits provided by OGB
        
            for key, idx in split_idx.items():
                mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                mask[idx] = True
                data[f"{key}_mask"] = mask

            X_train = data.x[data["train_mask"]]    
            y_train = data.y[data["train_mask"]]
            X_valid = data.x[data["valid_mask"]]
            y_valid = data.y[data["valid_mask"]]
            X_test = data.x[data["test_mask"]]
            y_test = data.y[data["test_mask"]]

    else:

        # if random_split:

        X_train = data.x[data["train_mask"]].float()
        y_train = data.y[data["train_mask"]]
        X_valid = data.x[data["valid_mask"]].float()
        y_valid = data.y[data["valid_mask"]]
        X_test = data.x[data["test_mask"]].float()
        y_test = data.y[data["test_mask"]]
        

            # if stratify:
                # X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=0, stratify = data[1])
                # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify = y_train) # 0.25 x 0.8 = 0.2
            # else:
                # X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=0)
                # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0.2

    if scale:
            x_mean, x_std = X_train.mean(), X_train.std()
            X_train = (X_train - x_mean)/x_std
            X_valid = (X_valid - x_mean)/x_std
            X_test = (X_test - x_mean)/x_std

    if to_numpy:
        X_train = X_train.numpy()
        y_train = y_train.numpy().ravel()
        X_valid = X_valid.numpy()
        y_valid = y_valid.numpy().ravel()
        X_test = X_test.numpy()
        y_test = y_test.numpy().ravel()

    else:
        y_train = y_train.long().reshape(-1)
        y_valid = y_valid.long().reshape(-1)
        y_test = y_test.long().reshape(-1)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def log_details_to_wandb(model, hparams):
    wandb.log({"model": model,
               "dataset": hparams.dataset.name, 
               "random_split": hparams.dataset.random_split})

def prediction_scores(model, X_train, y_train, X_valid, y_valid, X_test = None, y_test = None):

    train_pred = model.predict(X_train)
    valid_pred = model.predict(X_valid)
    
    train_score = accuracy_score(train_pred, y_train)
    valid_score = accuracy_score(valid_pred, y_valid)

    if X_test is not None:
        test_pred = model.predict(X_test)
        test_score = accuracy_score(test_pred, y_test)
        return train_score, valid_score, test_score
    else:
        return train_score, valid_score


def remove_outstanding_classes_from_testset(y_test, X_test):

    outstanding_classes = [42, 43, 44, 45, 46]

    mask = sum(y_test==i for i in outstanding_classes)
    mask = np.invert(np.array(mask, dtype = bool))
    mask_idx = mask.reshape(-1).nonzero()[0].reshape(-1)
 
    return y_test[mask_idx], X_test[mask_idx, :]# , y_pred[mask_idx]