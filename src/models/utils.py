import torch
from sklearn.preprocessing import StandardScaler

def data_split(data, scale = True, to_numpy = False):
    
    split_idx = data.get_idx_split()

    data = data[0]

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