import os
import pickle
import random
from tqdm import tqdm

import jieba
import scipy.sparse as ssp
import numpy as np
import pandas as pd

from src.data import Dictionary
# load_stop_words
def build_stop_words(load_stop_words_file,save_stop_words_file):
    words_dict = Dictionary()
    with open(load_stop_words_file,mode="r",encoding="gbk") as rfp:
        for line in tqdm(rfp.readlines()):
            word = line.strip()
            words_dict.add(word)
    words_dict.save(save_stop_words_file)
def build_embeddings(save_embedding_file,load_embedding_file,words_dict,chars_dict):
    word_embeddings = {}
    char_embeddings = {}
    with open(load_embedding_file,mode="r",encoding="utf-8") as rfp:
        word_nums,embedding_dim = rfp.readline().strip().split()
        word_nums = int(word_nums)
        embedding_dim = int(embedding_dim)
        for line in tqdm(rfp.readlines(),desc="loading embeddings"):
            data = line.split()
            word = str(data[0])
            if word in words_dict:
                word_embeddings[word] = list(map(float,data[1:]))
            if word in chars_dict:
                char_embeddings[word] = list(map(float,data[1:]))
    for word in tqdm(words_dict,"fixing embeddings"):
        if word not in word_embeddings:
            word_embeddings[word] = [random.random() for _ in range(embedding_dim)]
    for word in tqdm(chars_dict,"fixing embeddings"):
        if word not in word_embeddings:
            char_embeddings[word] = [random.random() for _ in range(embedding_dim)]
    data_dict = {
        "embedding-dim":embedding_dim,
        "word-embeddings":word_embeddings,
        "char-embeddings":char_embeddings
    }
    with open(save_embedding_file,mode="wb") as wfp:
        pickle.dump(data_dict,wfp)

def build_dictionary(result_dir,data_dict_dir,sent_name):
    words_dict = Dictionary()
    chars_dict = Dictionary()
    
    for file_name in os.listdir(data_dict_dir):
        path_name_file = os.path.join(data_dict_dir,file_name)
        dataset = pd.read_csv(path_name_file)
        all_raw_data = list(dataset[sent_name].map(jieba.cut))
        for idx in tqdm(range(len(all_raw_data)),desc='processing file:%s'%path_name_file):
            for word in all_raw_data[idx]:
                words_dict.add(word)
            for char in "".join(all_raw_data[idx]):
                chars_dict.add(char)
    save_chars_dict_file = os.path.join(result_dir,"chars_dict.pkl")
    save_words_dict_file = os.path.join(result_dir,"words_dict.pkl")
    chars_dict.save(save_chars_dict_file)
    words_dict.save(save_words_dict_file)
def build_dataset(dataset_file,save_dataset_file,sent_name):
    dataset = pd.read_csv(dataset_file)
    all_data = list(map(list,list(dataset[sent_name].map(jieba.cut))))
    labels = dataset.drop(labels='id',axis=1).drop(labels=sent_name,axis=1).to_numpy()
    idx = dataset['id'].to_numpy()
    labels = labels+2
    all_process_data = {
        "idx":idx,
        "content":all_data,
        "labels":labels
    }
    with open(save_dataset_file,mode="wb") as wfp:
        pickle.dump(all_process_data,wfp)

# build graph function
def build_graph(word_dict,word_embeddings,save_dataset_file,save_graph_file,window_size=3,weighted_graph=True,truncate=True,max_trunc_len=350):
    x_adj = []
    x_feature = []
    doc_len_list = []

    with open(save_dataset_file,mode="rb") as rfp:
        raw_dataset = pickle.load(rfp)
    all_dataset = raw_dataset['content']
    dataset_len = len(all_dataset)
    for i in tqdm(range(dataset_len),desc="building graph"):
        doc_words = all_dataset[i]
        if truncate:
            doc_words = doc_words[:max_trunc_len]
        doc_len = len(doc_words)

        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)

        doc_len_list.append(doc_nodes)

        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_dict[word_p]
                    word_q = window[q]
                    word_q_id = word_dict[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
    
        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[word_dict[p]])
            col.append(doc_word_id_map[word_dict[q]])
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = ssp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
    
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(word_embeddings[k])

        x_adj.append(adj)
        x_feature.append(features)
    graph_data = {
        "x_adj":x_adj,
        "x_feature":x_feature,
        "doc_len_list":doc_len_list,
        "weighted_graph":weighted_graph,
        "window_size":window_size,
        "truncate":truncate,
        "max_trunc_len":max_trunc_len
    }
    with open(save_graph_file,mode="wb") as wfp:
        pickle.dump(graph_data,wfp)
def load_dataset(result_data_dir):
    load_train_file = os.path.join(result_data_dir,"train.pkl")
    load_validate_file = os.path.join(result_data_dir,"validate.pkl")
    load_train_graph_file = os.path.join(result_data_dir,"train_graph.pkl")
    load_validate_graph_file = os.path.join(result_data_dir,"validate_graph.pkl")
    train_adj = []
    train_embed = []
    train_content = []
    train_y = []
    val_adj = []
    val_embed = []
    val_y = []
    val_content = []

    # The train dataset loading
    with open(load_train_file,mode="rb") as rfp:
        train_dataset = pickle.load(rfp)
    with open(load_train_graph_file,mode="rb") as rfp:
        train_graph_data = pickle.load(rfp)
    train_length = len(train_graph_data["x_adj"])
    for idx in range(train_length):
        train_adj.append(train_graph_data["x_adj"][idx])
        train_embed.append(train_graph_data["x_feature"][idx])
        train_y.append(train_dataset["labels"][idx])
        train_content.append(train_dataset["content"][idx])

    # The validate dataset loading
    with open(load_validate_file,mode="rb") as rfp:
        validate_dataset = pickle.load(rfp)
    with open(load_validate_graph_file,mode="rb") as rfp:
        validate_graph_data = pickle.load(rfp)
    validate_length = len(validate_graph_data["x_adj"])
    for idx in range(validate_length):
        val_adj.append(validate_graph_data["x_adj"][idx])
        val_embed.append(validate_graph_data["x_feature"][idx])
        val_y.append(validate_dataset["labels"][idx])
        val_content.append(validate_dataset["content"][idx])
    return (train_adj, train_embed,train_content,train_y),(val_adj, val_embed,val_content,val_y)
