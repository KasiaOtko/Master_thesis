import torch
from torch_geometric.nn import Node2Vec
from src.data.make_dataset import load_data
from src.models.utils import data_split, log_details_to_wandb, prediction_scores
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import multiprocessing
import sys
import logging

sys.path.append("..")
n_cores = multiprocessing.cpu_count()

logging.basicConfig(
    filename="logs/ml_classifiers.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

@hydra.main(config_path="../config", config_name="default_config.yaml")
def run_vode2vec(config):

    hparams = config.ffnn.hyperparameters
    print(f"configuration: \n {OmegaConf.to_yaml(hparams)}")
    #wandb.init(project="master-thesis", config = hparams, group = "node2vec")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    if "ogb" in hparams.dataset.name:
        data = load_data(hparams.dataset.name, orig_cwd + config.root)
    else:
        data = load_data(hparams.dataset.name, orig_cwd)
    log_details_to_wandb("node2vec", hparams)
    logging.info("Data loaded.")

    X_train, y_train, X_valid, y_valid, X_test, y_test = data_split(data,
                                                                    dataset = hparams.dataset.name,
                                                                    scale = hparams["scale"],
                                                                    to_numpy = True,
                                                                    random_split = hparams.dataset.random_split,
                                                                    stratify = hparams.dataset.stratify)
    model = Node2Vec(data.edge_index, embedding_dim=64, 
                 walk_length=20,                        # lenght of rw
                 context_size=10, walks_per_node=20,
                 num_negative_samples=1, 
                 p=500, q=0.5,                             # bias parameters
                 sparse=True).to(device)

    loader = model.loader(batch_size=128, # we generate 128*20 random walks in every batch
                      shuffle=True, num_workers=4)

    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    batch_size = hparams["batch_size"]
    epochs = hparams["epochs"]
    train_losses = []
    train_acc, valid_acc = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device)) # compute loss using positive and negative random walks
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(loader))

        model.eval()
        z = model() # computes embeddings of the model [N, embedding_dim]
        train_score = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.train_mask], data.y[data.train_mask],
                        max_iter=150)

        valid_score = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.valid_mask], data.y[data.valid_mask],
                        max_iter=150) # in the test function, logistic regression is implemented to test the embeddings

        logging.info("Epoch %2i : Train loss %f, Train Accuracy %f, Validation Accuracy %f" % (
                    epoch+1, train_losses[-1], train_score, valid_score))

        #wandb.log({"train_loss": train_losses[-1], "train_score": train_score, "valid_score": valid_score})
    
    test_score = model.test(z[data.train_mask], data.y[data.train_mask],
                        z[data.test_mask], data.y[data.test_mask],
                        max_iter=150)
    #wandb.log({"test_score": test_score})
