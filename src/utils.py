import os
import pickle
import jieba
import random

from tqdm import tqdm
import numpy as np
import scipy.sparse as ssp
import pandas as pd

from src.data import Dictionary

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

def build_dictionary(result_dir):
    words_dict = Dictionary()
    chars_dict = Dictionary()
    save_train_dataset_file = os.path.join(result_dir,"train.pkl")
    save_validate_dataset_file = os.path.join(result_dir,"validate.pkl")
    listed_dirs = [save_train_dataset_file,save_validate_dataset_file]
    for file_name in listed_dirs:
        with open(save_train_dataset_file,mode="rb") as rfp:
            all_data = pickle.load(rfp)
        for idx in tqdm(range(len(all_data["content"])),desc='processing file:%s'%file_name):
            for word in all_data["content"][idx]:
                words_dict.add(word)
            for char in "".join(all_data["content"][idx]):
                chars_dict.add(char)
    save_chars_dict_file = os.path.join(result_dir,"chars_dict.pkl")
    save_words_dict_file = os.path.join(result_dir,"words_dict.pkl")
    chars_dict.save(save_chars_dict_file)
    words_dict.save(save_words_dict_file)
def build_embeddings(result_dir,load_embedding_file,save_embedding_file):
    save_words_dataset_file = os.path.join(result_dir,"chars_dict.pkl")
    save_chars_dataset_file = os.path.join(result_dir,"words_dict.pkl")

    chars_dict = Dictionary.load(save_chars_dataset_file)
    words_dict = Dictionary.load(save_words_dataset_file)
    char_embeddings = {}
    word_embeddings = {}
    with open(load_embedding_file,mode="r",encoding="utf-8") as rfp:
        word_nums,embedding_dim = rfp.readline().strip().split()
        word_nums = int(word_nums)
        embedding_dim = int(embedding_dim)
        for line in tqdm(rfp.readlines(),desc="loading embeddings"):
            data = line.split()
            word = str(data[0])
            if word in words_dict:
                word_embeddings[word] = np.array(list(map(float,data[1:])),dtype=np.float)
            if word in chars_dict:
                char_embeddings[word] = np.array(list(map(float,data[1:])),dtype=np.float)
    for word in tqdm(words_dict,"fixing word embeddings"):
        if word not in word_embeddings:
            word_embeddings[word] = np.random.uniform(-0.01,0.01,embedding_dim)
    for word in tqdm(chars_dict,"fixing char embeddings"):
        if word not in word_embeddings:
            char_embeddings[word] = np.random.uniform(-0.01,0.01,embedding_dim)
    data_dict = {
        "embedding-dim":embedding_dim,
        "word-embeddings":word_embeddings,
        "char-embeddings":char_embeddings
    }
    with open(save_embedding_file,mode="wb") as wfp:
        pickle.dump(data_dict,wfp)
def build_stop_words(load_stop_words_file,save_stop_words_file):
    words_dict = Dictionary()
    with open(load_stop_words_file,mode="r",encoding="gbk") as rfp:
        for line in tqdm(rfp.readlines(),desc='processing stop words'):
            word = line.strip()
            words_dict.add(word)
    words_dict.save(save_stop_words_file)
# build graph function
def build_graph(words_dict,chars_dict,save_dataset_file,save_graph_file,
            save_stopwords_file=None,window_size=3,weighted_graph=True):
    x_adj = []
    doc_len_list = []

    with open(save_dataset_file,mode="rb") as rfp:
        raw_dataset = pickle.load(rfp)
    if save_stopwords_file is not None:
        with open(save_dataset_file,mode="rb") as rfp:
            stop_words_dict = pickle.load(rfp)
        stop_words_flag = True
    else:
        stop_words_flag = False
    all_dataset = raw_dataset['content']
    dataset_len = len(all_dataset)
    x_words_features = []
    x_chars_features = []
    x_masks = []
    for i in tqdm(range(dataset_len),desc="building graph"):
        doc_words = all_dataset[i]
        
        doc_len = len(doc_words)
        doc_len_list.append(doc_len)
        doc_word2id_map = {}
        doc_id2word = list(set(doc_words))
        for j in range(len(doc_id2word)):
            doc_word2id_map[doc_id2word[j]] = j
        # sliding windows
        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)

        word_pair_count = {}
        for win in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = win[p]
                    word_p_id = doc_word2id_map[word_p]
                    word_q = win[q]
                    word_q_id = doc_word2id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    value = 0.0
                    if word_p not in words_dict.label_token:
                        if stop_words_flag and word_p not in stop_words_dict:
                            value += 0.5
                    if word_q not in words_dict.label_token:
                        if stop_words_flag and word_p not in stop_words_dict:
                            value += 0.5
                    
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += value
                    else:
                        word_pair_count[word_pair_key] = value
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += value
                    else:
                        word_pair_count[word_pair_key] = value
    
        row = []
        col = []
        weight = []
        words_features = []
        chars_features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(p)
            col.append(q)
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = ssp.csr_matrix((weight, (row, col)), shape=(doc_len,doc_len))
    
        for word in doc_words:
            words_features.append(words_dict[word])
        for char in "".join(doc_words):
            chars_features.append(chars_dict[char])
        x_adj.append(adj)
        x_words_features.append(words_features)
        x_chars_features.append(chars_features)
    words_len = [len(item) for item in x_words_features]
    chars_len = [len(item) for item in x_chars_features]
    average_words_len = np.mean(words_len)
    average_chars_len = np.mean(chars_len)
    max_words_len = max(words_len)
    max_chars_len = max(chars_len)
    graph_data = {
        "x_adj":x_adj,
        "x_words_features":x_words_features,
        "x_chars_features":x_chars_features,
        "x_masks":x_masks,
        "labels":raw_dataset["labels"],
        "average_words_len":average_words_len,
        "average_chars_len":average_chars_len,
        "max_words_len":max_words_len,
        "max_chars_len":max_chars_len,
    }
    with open(save_graph_file,mode="wb") as wfp:
        pickle.dump(graph_data,wfp)





