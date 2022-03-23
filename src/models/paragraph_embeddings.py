import numpy as np
import pandas as pd
import multiprocessing
import sys
import logging
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from src.data.make_dataset import load_data
from src.models.utils import data_split, log_details_to_wandb
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append("..")
n_cores = multiprocessing.cpu_count()

logging.basicConfig(
    filename="logs/ml_classifiers.txt",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

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
        embeddings[i,:] = model.infer_vector(tagged_data[i].words, epochs = common_params['epochs'])
        # Normalize vector
        if norm:
            embeddings[i,:] = embeddings[i,:]/np.linalg.norm(embeddings[i,:])
        
    return embeddings


def generate_embeddings(sentences, hparam, common_params):

    tokenized_doc = tokenize(sentences)
    tagged_data = tag_data(tokenized_doc)
    model = Doc2Vec(tagged_data,
                              dm=hparam['dm'], vector_size=hparam['vector_size'], window=hparam['window'],
                              **common_params)

    X = infer_embeddings(model, tagged_data, sentences, norm = True)

    return X

@hydra.main(config_path="../config", config_name="default_config.yaml")
def classify(config):

    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    hparams = config.doc2vec.hyperparameters
    wandb.init(project="master-thesis", config = hparams, group = "ffnn")
    orig_cwd = hydra.utils.get_original_cwd()
    logging.info("Configuration: {0}".format(hparams))

    links, cases = load_data(hparams.dataset.name, orig_cwd + config.root)

    sentences = prepare_sentences(links, cases)

    X = generate_embeddings(sentences, hparams, hparams.common_params)