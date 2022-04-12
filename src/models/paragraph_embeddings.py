import numpy as np
import pandas as pd
import multiprocessing
import sys
import logging
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score
from src.data.make_dataset import load_data
from src.models.utils import data_split
from models.FFNN_model import FFNNClassifier
import wandb
import hydra
import torch
import torch.optim as optim
import torch.nn as nn
from omegaconf import OmegaConf

sys.path.append("..")
n_cores = multiprocessing.cpu_count()

logging.basicConfig(
    filename="logs/ml_classifiers.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def train_NNet(X_train, y_train, X_valid, y_valid, X_test, y_test, hparams):

    X_train = torch.from_numpy(X_train).float()
    X_valid = torch.from_numpy(X_valid).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_valid = torch.from_numpy(y_valid).long()
    y_test = torch.from_numpy(y_test).long()

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
    logging.info("Test set evaluation: Loss %f, Accuracy %f" % (cur_loss/num_batches_test, test_acc))
        # wandb.log({"Test accuracy": test_acc})

    return train_acc[-1], valid_acc[-1], test_acc


def tokenize(sentences):
    tokenized_doc = []
    for d in sentences.TEXT:
        tokenized_doc.append(word_tokenize(d.lower()))
    return tokenized_doc


def tag_data(tokenized_doc):
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
    return tagged_data


def prepare_sentences(links, cases):
        temp = links[['CELEX_FROM', 'FROM_ID', 'text_from_clean', 'CELEX_TO', 'TO_ID', 'text_to_clean']].copy()
        ID = [col for col in temp.columns if "ID" in col]
        text = [col for col in temp.columns if "text" in col]
        celex = [col for col in temp.columns if 'CELEX' in col]
        temp = pd.lreshape(temp, {'ID': ID,'TEXT': text, 'CELEX': celex})
        temp = temp.drop_duplicates()

        temp = temp.merge(cases[["CELEX", "Category_encoded"]], how = "left", left_on = "CELEX", right_on = "CELEX")

        sentences = temp.set_index("ID")

        return sentences


def infer_embeddings(model, tagged_data, temp, norm = True):
    
    assert len(tagged_data) == len(temp), "Sizes of tagged data and dataframe do not match."
    
    embeddings = np.zeros((len(tagged_data), model.vector_size), dtype = 'object')

    for i in range(len(tagged_data)):
        embeddings[i,:] = model.infer_vector(tagged_data[i].words)
        # Normalize vector
        if norm:
            embeddings[i,:] = embeddings[i,:]/np.linalg.norm(embeddings[i,:])
        
    return embeddings


def generate_embeddings(sentences, hparams):

    tokenized_doc = tokenize(sentences)
    tagged_data = tag_data(tokenized_doc)
    model = Doc2Vec(tagged_data,
                              dm=hparams['dm'], vector_size=hparams['vector_size'], window=hparams['window'],
                              min_count = hparams["min_count"], epochs = hparams['epochs'],
                              **hparams.common_params)
    logging.info("Model trained.")

    X = infer_embeddings(model, tagged_data, sentences, norm = True)

    X = X.astype(float)

    return X

@hydra.main(config_path="config", config_name="default_config.yaml")
def classify(config):
    
    hparams = config.doc2vec.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    wandb.init(project="master-thesis", config = hparams, group = "doc2vec")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    links, cases = load_data(hparams.dataset.name, orig_cwd)

    sentences = prepare_sentences(links, cases)

    X = generate_embeddings(sentences, hparams)
    logging.info("Embeddings generated.")

    y = sentences.Category_encoded.values

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split((X, y),
                                                                    dataset = hparams.dataset.name,
                                                                    scale = hparams["scale"],
                                                                    random_split = hparams["random_split"],
                                                                    stratify = hparams["stratify"])




    # Evaluate embeddings - use them to classify the paragraphs
    # lr = LogisticRegression(random_state=0, max_iter = 2000, C = 100, n_jobs = n_cores)
    # lr.fit(X_train, y_train)
    # logging.info("Classifier fitted.")
    # train_preds = lr.predict(X_train)
    # val_preds = lr.predict(X_valid)
    # test_preds = lr.predict(X_test)

    # train_score = accuracy_score(train_preds, y_train)
    # valid_score = accuracy_score(val_preds, y_valid)
    # test_score = accuracy_score(test_preds, y_test)
    net_hparams = config.ffnn.hyperparameters
    train_score, valid_score, test_score = train_NNet(X_train, y_train, X_valid, y_valid, X_test, y_test, net_hparams)

    wandb.log({"train_score": train_score, "valid_score": valid_score, "test_score": test_score})
    logging.info("Train score %f, Validation score %f, Test score %f" % (train_score, valid_score, test_score))


if __name__ == "__main__":
    
    classify()