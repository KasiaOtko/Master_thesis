import numpy as np
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from src.data.make_dataset import load_data
from src.models.utils import data_split, log_details_to_wandb

def tokenize(sentences):
    tokenized_doc = []
    for d in sentences.TEXT:
        tokenized_doc.append(word_tokenize(d.lower()))
    return tokenized_doc

def tag_data(tokenized_doc):
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
    return tagged_data


def infer_embeddings(model, tagged_data, temp, norm = True):
    
    assert len(tagged_data) == len(temp), "Sizes of tagged data and dataframe do not match."
    
    embeddings = np.zeros((len(tagged_data), model.vector_size), dtype = 'object')

    for i in range(len(tagged_data)):
        embeddings[i,:] = model.infer_vector(tagged_data[i].words, epochs = common_params['epochs'])
        # Normalize vector
        if norm:
            embeddings[i,:] = embeddings[i,:]/np.linalg.norm(embeddings[i,:])
        
    return embeddings


def generate_embeddings(sentences, param, common_params):

    tokenized_doc = tokenize(sentences)
    tagged_data = tag_data(tokenized_doc)
    model = Doc2Vec(tagged_data,
                              dm=param['dm'], vector_size=param['vector_size'], window=param['window'],
                              **common_params)

    X = infer_embeddings(model, tagged_data, sentences, norm = True)

    return X


def classify():

    data = load_data(hparams.dataset.name, orig_cwd + config.root)

    X = generate_embeddings(sentences, param, common_params)