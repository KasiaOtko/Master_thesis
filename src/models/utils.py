import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import wandb

def pyg_random_split(data, train_ratio = 0.8, valid_ratio = 0.25):

    train_ratio = train_ratio
    valid_ratio = valid_ratio
    num_nodes = data.x.shape[0]
    num_train = int(num_nodes * train_ratio)
    num_valid = int(num_nodes * train_ratio * valid_ratio)
    idx = [i for i in range(num_nodes)]

    np.random.shuffle(idx)

    train_mask = torch.full_like(data.y, False, dtype=bool)
    train_mask[idx[:num_train-num_valid]] = True

    valid_mask = torch.full_like(data.y, False, dtype=bool)
    valid_mask[idx[num_train-num_valid:num_train]] = True

    test_mask = torch.full_like(data.y, False, dtype=bool)
    test_mask[idx[num_train:]] = True

    assert torch.sum(train_mask) + torch.sum(valid_mask) + torch.sum(test_mask) == num_nodes, "Wrong proprtions of split."
    assert len(set(train_mask.nonzero()).intersection(valid_mask.nonzero())) == 0, "Wrong train-valid split."
    assert len(set(train_mask.nonzero()).intersection(test_mask.nonzero())) == 0, "Wrong train-test split."
    assert len(set(valid_mask.nonzero()).intersection(test_mask.nonzero())) == 0, "Wrong valid-test split."

    data.train_mask = train_mask.reshape(-1)
    data.valid_mask = valid_mask.reshape(-1)
    data.test_mask = test_mask.reshape(-1)

    return data

def pyg_data_split(data, dataset, random_split):

    if "ogb" in dataset:
    
        split_idx = data.get_idx_split()
        data = data[0]
    
        if random_split:
            if "products" in dataset:
                data = pyg_random_split(data, 0.4, 0.25)
            elif "arxiv" in dataset:
                data = pyg_random_split(data, 0.71, 0.24)

        else:
            for key, idx in split_idx.items():
                mask = torch.zeros(data.num_nodes, dtype=torch.bool)
                mask[idx] = True
                data[f"{key}_mask"] = mask
    
    else:

        if random_split:
            data = pyg_random_split(data)

    return data


def eval_classifier(test_targs, test_preds):
    
    test_acc = accuracy_score(test_targs, test_preds)
    test_f1_weighted = f1_score(test_targs, test_preds, average = 'weighted')
    test_f1_macro = f1_score(test_targs, test_preds, average = 'macro')
    test_recall_weighted = recall_score(test_targs, test_preds, average = 'weighted')
    test_recall_macro = recall_score(test_targs, test_preds, average = 'macro')
    test_prec_weighted = precision_score(test_targs, test_preds, average = 'weighted')
    test_prec_macro = precision_score(test_targs, test_preds, average = 'macro')

    return test_acc, (test_f1_weighted, test_f1_macro), (test_recall_weighted, test_recall_macro), (test_prec_weighted, test_prec_macro)


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

        # X_train = data.x[data["train_mask"]].float()
        # y_train = data.y[data["train_mask"]]
        # X_valid = data.x[data["valid_mask"]].float()
        # y_valid = data.y[data["valid_mask"]]
        # X_test = data.x[data["test_mask"]].float()
        # y_test = data.y[data["test_mask"]]
        

            if stratify:
                X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=0, stratify = data[1])
                X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify = y_train) # 0.25 x 0.8 = 0.2
            else:
                X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.2, random_state=0)
                X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=0) # 0.25 x 0.8 = 0.2

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

    # else:
    #     y_train = y_train.long().reshape(-1)
    #     y_valid = y_valid.long().reshape(-1)
    #     y_test = y_test.long().reshape(-1)

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


def draw_learning_curve(epochs, train_losses, valid_losses, metric, root):

    fig = plt.figure(figsize=(8, 5))
    plt.plot(range(epochs), train_losses, 'r', range(epochs), valid_losses, 'b')
    plt.legend(['Train {}'.format(metric),'Validation {}'.format(metric)])
    plt.xlabel('Epochs'), plt.ylabel(f"{metric}");
    plt.savefig(f"{root}/reports/figures/FFNN_{metric}_curve.png")
    wandb.log({f"{metric}_curve": wandb.Image(fig)})

    # fig = plt.figure(figsize=(8, 5))
    # plt.plot(range(epochs), train_losses, 'r', range(epochs), valid_losses, 'b')
    # plt.legend(['Train {}'.format(metric), 'Validation {}'.format(metric)])
    # plt.xlabel('Epochs'), plt.ylabel('Loss');
    # plt.savefig(f"{orig_cwd}/reports/figures/FFNN_{metric}_curve.png")
    # wandb.log({"loss_curve": wandb.Image(fig)})