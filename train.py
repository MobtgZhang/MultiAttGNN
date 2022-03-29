import pickle
import os

import numpy as np
import tensorflow as tf

from src.adjacency import preprocess_add_apdding
from config import get_tf_args

def main():
    FLAGS = get_tf_args()
    # load embeddings and dataset 
    result_data_dir = os.path.join(FLAGS.result_dir,FLAGS.dataset)
    load_chars_dict_file = os.path.join(result_data_dir,"chars_dict.pkl")     
    load_words_dict_file = os.path.join(result_data_dir,"words_dict.pkl") 
    load_embeddings_file = os.path.join(result_data_dir,"embedding.pkl")
    with open(load_chars_dict_file,mode="rb") as rfp:
        chars_dict = pickle.load(rfp)
    with open(load_words_dict_file,mode="rb") as rfp:
        words_dict = pickle.load(rfp)
    with open(load_embeddings_file,mode="rb") as rfp:
        embedding_data = pickle.load(rfp)
        char_embeddings = embedding_data["char-embeddings"]
        word_embeddings = embedding_data["word-embeddings"]
    load_graph_file = os.path.join(result_data_dir,"train_graph.pkl")
    load_validate_graph_file = os.path.join(result_data_dir,"validate_graph.pkl")
    with open(load_graph_file,mode="rb") as rfp:
        train_graph_data = pickle.load(rfp)
        
    with open(load_validate_graph_file,mode="rb") as rfp:
        validate_graph_data = pickle.load(rfp)

    # Some preprocessing
    print('loading training set')
    preprocess_add_apdding(train_graph_data)
    print(train_graph_data.keys())
    exit()
    train_feature = preprocess_features(train_feature)
    print('loading validation set')
    val_adj, val_mask = preprocess_adj(val_adj)
    val_feature = preprocess_features(val_feature)

if __name__ == "__main__":
    main()
