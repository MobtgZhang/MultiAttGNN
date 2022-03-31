import pickle
import os

import numpy as np
import tensorflow as tf

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

    if FLAGS.model == 'gnn':
        # support = [preprocess_adj(adj)]
        # num_supports = 1
        model_func = GNN
    elif FLAGS.model == 'gcn_cheby': # not used
        # support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GNN
    elif FLAGS.model == 'dense': # not used
        # support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

if __name__ == "__main__":
    main()
