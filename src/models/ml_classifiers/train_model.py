import csv
import logging
import multiprocessing
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearnex import patch_sklearn

import wandb
from models.FFNN_model import FFNNClassifier
from src.data.make_dataset import load_data
from src.models.utils import (data_split, eval_classifier,
                              log_details_to_wandb, prediction_scores,
                              pyg_data_split)

sys.path.append("..")
n_cores = multiprocessing.cpu_count()

logging.basicConfig(
    filename="logs/ml_classifiers.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@hydra.main(config_path="../config", config_name="default_config.yaml")
def Run_log_reg(config : DictConfig):
    
    hparams = config.log_reg.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "log_reg")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
    log_details_to_wandb("log_reg", hparams)
    logging.info("Data loaded.")

    # X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data,
    #                                                                 dataset = hparams.dataset.name,
    #                                                                 scale = hparams["scale"],
    #                                                                 to_numpy = False,
    #                                                                 random_split = hparams.dataset.random_split,
    #                                                                 stratify = hparams.dataset.stratify)
    data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
    X_train = data.x[data["train_mask"]].numpy()
    y_train = data.y[data["train_mask"]].numpy().ravel()
    X_valid = data.x[data["valid_mask"]].numpy()
    y_valid = data.y[data["valid_mask"]].numpy().ravel()
    X_test = data.x[data["test_mask"]].numpy()
    y_test = data.y[data["test_mask"]].numpy().ravel()

    # C_range = hparams["C_range"]
    # acc_table = pd.DataFrame(
    #     columns = ["Train_accuracy", "Valid_accuracy", "Test accuracy"],
    #     index = C_range,
    # )

    # for C in C_range:
    #     lr = LogisticRegression(random_state=hparams["random_state"], C=C, max_iter=hparams["max_iter"], n_jobs = n_cores).fit(X_train, y_train)
    #     y_train_pred = lr.predict(X_train)
    #     y_valid_pred = lr.predict(X_valid)
    #     acc_table.loc[C, "Train_accuracy"] = pd.to_numeric(
    #         accuracy_score(y_train, y_train_pred)
    #     )
    #     acc_table.loc[C, "Valid_accuracy"] = pd.to_numeric(
    #         accuracy_score(y_valid, y_valid_pred)
    #     )
    #     logging.info(
    #         "C = {0}: Train_accuracy = {1}, Validation accuracy = {2}".format(
    #         C,
    #         acc_table.loc[C, "Train_accuracy"],
    #         acc_table.loc[C, "Valid_accuracy"])
    #     )

    # wandb_table = wandb.Table(dataframe=acc_table)
    # wandb.log({"acc_table": wandb_table})

    # C_final = pd.to_numeric(
    #     acc_table[~acc_table["Valid_accuracy"].isna()]["Valid_accuracy"]
    # ).idxmax()

    lr = LogisticRegression(random_state=hparams["random_state"], C=hparams["C_range"], max_iter=hparams["max_iter"])
    lr.fit(X_train, y_train)
    train_score, valid_score, test_score = prediction_scores(lr, X_train, y_train, X_valid, y_valid, X_test, y_test)

    if hparams.inference:
        test_score_best = 0
        for i in range(10):
            print("Iteration", i)
            if "ogb" in hparams.dataset.name:
                data = load_data(hparams.dataset.name, orig_cwd + config.root)
            else:
                data = load_data(hparams.dataset.name, orig_cwd)

            data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
            X_test = data.x[data["test_mask"]].numpy()
            y_test = data.y[data["test_mask"]].numpy().ravel()
            test_preds=lr.predict(X_test)
            
            test_score, f1, recall, precision = eval_classifier(y_test, test_preds)

            if test_score > test_score_best:
                np.save(orig_cwd + "/models/preds_{}".format(hparams.dataset.name), test_preds)

            wandb.log({"test_score": test_score,
                    "test_f1_weighted": f1[0], "test_f1_macro": f1[1], "test_recall_weighted": recall[0],
                    "test_recall_macro": recall[1], "test_prec_weighted": precision[0], "test_prec_macro":precision[1]})

            stats = ["logreg_{}".format(i), hparams.dataset.name, hparams.dataset.random_split,
                    train_score, valid_score, test_score,
                    f1[0], f1[1], recall[0], recall[1], precision[0], precision[1]]
            with open(orig_cwd+'/logs/inference_local.csv', 'a', newline='') as f: # .format(hparams.dataset.name)
                writer = csv.writer(f)
                writer.writerow(stats)

    # train_score, valid_score, test_score = prediction_scores(lr, X_train, y_train, X_valid, y_valid, X_test, y_test)
    # wandb.log({"train_score": train_score, "valid_score": valid_score, "test_score": test_score})
    # logging.info("Train score %f, Validation score %f, Test score %f" % (train_score, valid_score, test_score))


@hydra.main(config_path="../config", config_name="default_config.yaml")
def train_Nnet(config):

    hparams = config.ffnn.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "ffnn")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
    log_details_to_wandb("ffnn", hparams)
    logging.info("Data loaded.")

    # X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data,
    #                                                                 dataset = hparams.dataset.name,
    #                                                                 scale=hparams.dataset.scale,
    #                                                                 to_numpy=False,
    #                                                                 random_split=hparams.dataset.random_split,
    #                                                                 stratify=hparams.dataset.stratify)
    data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
    X_train = data.x[data["train_mask"]].float()
    y_train = data.y[data["train_mask"]].long().ravel()
    X_valid = data.x[data["valid_mask"]].float()
    y_valid = data.y[data["valid_mask"]].long().ravel()
    X_test = data.x[data["test_mask"]].float()
    y_test = data.y[data["test_mask"]].long().ravel()
    # if "EU" in hparams.dataset.name:
    #     X_train = torch.from_numpy(X_train).float()
    #     X_valid = torch.from_numpy(X_valid).float()
    #     X_test = torch.from_numpy(X_test).float()
    #     y_train = torch.from_numpy(y_train).long()
    #     y_valid = torch.from_numpy(y_valid).long()
    #     y_test = torch.from_numpy(y_test).long()

    num_classes = 47#max(len(np.unique(y_train)), len(np.unique(y_valid)), len(np.unique(y_test)))

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

    if hparams.inference:
        test_score_best = 0

        for i in range(10):
            print("Iteration", i)
            if "ogb" in hparams.dataset.name:
                data = load_data(hparams.dataset.name, orig_cwd + config.root)
            else:
                data = load_data(hparams.dataset.name, orig_cwd)
            data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
            X_test = data.x[data["test_mask"]].float()
            y_test = data.y[data["test_mask"]].long().ravel()
            num_samples_test = X_test.shape[0]
            num_batches_test = num_samples_test // batch_size

            model.load_state_dict(torch.load(orig_cwd + "/models/ffnn_{}.pt".format(hparams.dataset.name), map_location='cpu'))
            model.eval()

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
                    test_targs += list(y_test[slce].numpy())
                    test_preds += list(preds.data.numpy())

            test_score, f1, recall, precision = eval_classifier(test_targs, test_preds)
            if test_score > test_score_best:
                np.save(orig_cwd + "/models/ffnn_preds_{}".format(hparams.dataset.name), test_preds)

            wandb.log({"test_score": test_score,
        "test_f1_weighted": f1[0], "test_f1_macro": f1[1], "test_recall_weighted": recall[0],
        "test_recall_macro": recall[1], "test_prec_weighted": precision[0], "test_prec_macro":precision[1]})

            stats = ["ffnn_{}".format(i), hparams.dataset.name, hparams.dataset.random_split,
                    0, 0, test_score,
                    f1[0], f1[1], recall[0], recall[1], precision[0], precision[1]]
            with open(orig_cwd+'/logs/inference_local.csv', 'a', newline='') as f: # .format(hparams.dataset.name)
                writer = csv.writer(f)
                writer.writerow(stats)
    else:
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
                batch_loss = criterion(output, y_batch)     # compute loss
                batch_loss.backward()                       # backward pass
                optimizer.step()                            # update parameters
                
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

        torch.save(model.state_dict(), orig_cwd + "/models/ffnn_{}.pt".format(hparams.dataset.name))
        # Evaluate final model on the test set
        # if ~hparams.dataset.random_split:
        #     y_test, X_test = remove_outstanding_classes_from_testset(y_test, X_test)
        # num_batches_test = X_test.shape[0] // batch_size
        test_targs, test_preds = [], []
        cur_loss = 0
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
def Run_random_forest(config):

    hparams = config.rf.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "rf")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(config["rf"]["hyperparameters"]))

    data = load_data(hparams.dataset.name, orig_cwd + "/data/raw")
    wandb.log({"dataset": hparams.dataset.name})
    logging.info("Data loaded.")

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data, 
                                                                    hparams["scale"], 
                                                                    to_numpy=False, 
                                                                    random_split = True, 
                                                                    stratify = hparams.dataset.stratify)

    # X_train, y_train, X_valid, y_valid = X_train[:100], y_train[:100], X_valid[:100], y_valid[:100] 

    rf = RandomForestClassifier(n_estimators = hparams['n_estimators'], criterion = hparams['criterion'], 
                                max_depth = hparams['max_depth'], min_samples_split = hparams['min_samples_split'], 
                                min_samples_leaf=hparams['min_samples_leaf'], max_features=hparams['max_features'], 
                                bootstrap = hparams['bootstrap'],
                                n_jobs = 8, verbose = 1, random_state = 0)
    #rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, verbose=2, random_state=0,
                                #n_jobs = n_cores, cv = 3, return_train_score = True)

    #rf_random.fit(X_train, y_train)
    
    rf.fit(X_train, y_train)

    train_pred = rf.predict(X_train)
    valid_pred = rf.predict(X_valid)
    train_score = accuracy_score(train_pred, y_train)
    valid_score = accuracy_score(valid_pred, y_valid)
    wandb.log({"train_score": train_score, "valid_score": valid_score})
    logging.info("Train score %f, Validation score %f" % (train_score, valid_score))


@hydra.main(config_path="../config", config_name="default_config.yaml")
def run_XGBoost(config):

    hparams = config.xgb.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "xgb")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(config["xgb"]["hyperparameters"]))

    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
    wandb.log({"dataset": hparams.dataset.name})
    logging.info("Data loaded.")

    # X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data,
    #                                                                 dataset = hparams.dataset.name,
    #                                                                 scale = hparams["scale"], 
    #                                                                 to_numpy=False, 
    #                                                                 random_split = True, 
    #                                                                 stratify = hparams.dataset.stratify)

    xgb.config_context(verbosity = 3)
    if hparams.inference:
        test_score_best = 0
        for i in range(10):
            print("Iteration", i)
            if "ogb" in hparams.dataset.name:
                data = load_data(hparams.dataset.name, orig_cwd + config.root)
            else:
                data = load_data(hparams.dataset.name, orig_cwd)
            data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
            X_test = data.x[data["test_mask"]].numpy()
            y_test = data.y[data["test_mask"]].numpy().ravel()
            model = xgb.XGBClassifier(n_estimators = hparams["num_round"], max_depth = hparams["max_depth"],
                                learning_rate = hparams["eta"], objective = hparams["objective"],
                                tree_method = hparams["tree_method"], n_jobs = n_cores, gamma = hparams["gamma"],
                                min_child_weight  = hparams["min_child_weight"],
                                subsample = hparams["subsample"], colsample_bytree = hparams["colsample_bytree"])
            model.load_model(orig_cwd + "/models/xgb_{}.txt".format(hparams.dataset.name))

            test_preds = model.predict(X_test)

            test_score, f1, recall, precision = eval_classifier(y_test, test_preds)
            if test_score > test_score_best:
                np.save(orig_cwd + "/models/xgb_preds_{}".format(hparams.dataset.name), test_preds)

            wandb.log({"test_score": test_score,
        "test_f1_weighted": f1[0], "test_f1_macro": f1[1], "test_recall_weighted": recall[0],
        "test_recall_macro": recall[1], "test_prec_weighted": precision[0], "test_prec_macro":precision[1]})

            stats = ["xgb_{}".format(i), hparams.dataset.name, hparams.dataset.random_split,
                    0, 0, test_score,
                    f1[0], f1[1], recall[0], recall[1], precision[0], precision[1]]
            with open(orig_cwd+'/logs/inference.csv', 'a', newline='') as f: # .format(hparams.dataset.name)
                writer = csv.writer(f)
                writer.writerow(stats)

    else:
        data = pyg_data_split(data, hparams.dataset.name, hparams.dataset.random_split)
        X_train = data.x[data["train_mask"]].numpy()
        y_train = data.y[data["train_mask"]].numpy().ravel()
        X_valid = data.x[data["valid_mask"]].numpy()
        y_valid = data.y[data["valid_mask"]].numpy().ravel()
        X_test = data.x[data["test_mask"]].numpy()
        y_test = data.y[data["test_mask"]].numpy().ravel()
        evalset = [(X_train, y_train), (X_valid, y_valid)]
        model = xgb.XGBClassifier(n_estimators = hparams["num_round"], max_depth = hparams["max_depth"],
                                learning_rate = hparams["eta"], objective = hparams["objective"],
                                tree_method = hparams["tree_method"], n_jobs = n_cores, gamma = hparams["gamma"],
                                min_child_weight  = hparams["min_child_weight"],
                                subsample = hparams["subsample"], colsample_bytree = hparams["colsample_bytree"])

        model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric=['mlogloss', 'merror'], eval_set=evalset, verbose = True)

        results = model.evals_result()

        fig = plt.figure(figsize = (8, 5))
        plt.plot(results['validation_0']['mlogloss'], label='train')
        plt.plot(results['validation_1']['mlogloss'], label='validation')
        plt.title("XGB Log Loss")
        plt.legend()
        plt.savefig(f"{orig_cwd}/reports/figures/xgb_loss_curve.png")
        wandb.log({"loss_curve": wandb.Image(fig)})

        fig = plt.figure(figsize = (8, 5))
        plt.plot(1-np.array(results['validation_0']['merror']), label='train')
        plt.plot(1-np.array(results['validation_1']['merror']), label='validation')
        plt.title("XGB Accuracy score")
        plt.legend()
        plt.savefig(f"{orig_cwd}/reports/figures/xgb_accuracy_curve.png")
        wandb.log({"accuracy_curve": wandb.Image(fig)})

        train_score, valid_score, test_score = prediction_scores(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
        wandb.log({"train_score": train_score, "valid_score": valid_score, "test_score": test_score})
        logging.info("Train score %f, Validation score %f, Test score %f" % (train_score, valid_score, test_score))

        model.save_model(orig_cwd + "/models/xgb_{}.txt".format(hparams.dataset.name))
    

@hydra.main(config_path="../config", config_name="default_config.yaml")
def run_SVM(config):
    
    hparams = config.svm.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "svm")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(config["svm"]["hyperparameters"]))

    data = load_data(hparams.dataset.name, orig_cwd + "/data/raw")
    wandb.log({"dataset": hparams.dataset.name})
    logging.info("Data loaded.")

    patch_sklearn()
    
    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data, 
                                                                    hparams["scale"], 
                                                                    to_numpy=False, 
                                                                    random_split = True, 
                                                                    stratify = hparams.dataset.stratify)

    model = SVC(C = 100, kernel = "rbf", gamma = "scale", verbose = True, decision_function_shape = 'ovo')
    model.fit(X_train, y_train)
    logging.info("SVM trained. Making predictions.")

    train_score, valid_score, test_score = prediction_scores(model, X_train, y_train, X_valid, y_valid, X_test, y_test)
    wandb.log({"train_score": train_score, "valid_score": valid_score, "test_score": test_score})
    logging.info("Train score %f, Validation score %f, Test score %f" % (train_score, valid_score, test_score))


if __name__ == "__main__":

    train_Nnet()
