import torch

def data_split(data, to_tensor = False):
    
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

    if to_tensor:
        X_train = torch.Tensor(X_train)
        y_train = torch.Tensor(y_train).long()
        X_valid = torch.Tensor(X_valid)
        y_valid = torch.Tensor(y_valid).long()
        X_test = torch.Tensor(X_test)
        y_test = torch.Tensor(y_test).long()

    return X_train, y_train, X_valid, y_valid, X_test, y_test