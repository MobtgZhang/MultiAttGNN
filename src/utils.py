import os
import pickle
import math
import json

import jieba
from janome.tokenizer import Tokenizer

from tqdm import tqdm
import numpy as np
import scipy.sparse as ssp
import pandas as pd
import sklearn

from .data import Dictionary

def build_aichallenger_dataset(dataset_dir,reusult_dir):
    load_train_file = os.path.join(dataset_dir,"sentiment_analysis_trainingset.csv")
    load_validate_file = os.path.join(dataset_dir,"sentiment_analysis_validationset.csv")
    train_dataset = pd.read_csv(load_train_file)
    dataset = pd.read_csv(load_validate_file)
    dataset = sklearn.utils.shuffle(dataset)
    val_len = len(dataset)//2
    val_dataset = dataset.iloc[:val_len,:]
    test_dataset = dataset.iloc[val_len:,:]
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,val_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc='processing aichallenger dataset'):
        save_dataset_file = os.path.join(reusult_dir,"%s_dataset.pkl"%name)
        if os.path.exists(save_dataset_file):
            continue
        all_data = list(map(list,list(tp_dataset['content'].map(jieba.cut))))
        labels = tp_dataset.drop(labels='id',axis=1).drop(labels='content',axis=1).to_numpy()
        idx = tp_dataset['id'].to_numpy()
        labels = labels+2
        all_process_data = {
            "idx":idx,
            "content":all_data,
            "labels":labels
        }
        with open(save_dataset_file,mode="wb") as wfp:
            pickle.dump(all_process_data,wfp)
def build_wrime_dataset(load_dataset_file,result_data_dir,label_name='writer',wrime_name="v1"):
    dataset = pd.read_csv(load_dataset_file,sep='\t')
    columns_list = []
    for label in dataset.columns.values:
        if label_name in label.lower():
            columns_list.append(label)
    train_dataset = dataset[dataset['Train/Dev/Test']=='train']
    dev_dataset = dataset[dataset['Train/Dev/Test']=='dev']
    test_dataset = dataset[dataset['Train/Dev/Test']=='test']
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,dev_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc="processing WRIME-v2 dataset"):
        token_tp = Tokenizer()
        content = list(tp_dataset["Sentence"].map(lambda x:[item.surface for item in token_tp.tokenize(x)]))
        index = tp_dataset.index.to_numpy()
        labels = tp_dataset[columns_list].to_numpy()
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels
        }
        save_file_name = os.path.join(result_data_dir,'%s_dataset.pkl'%name)
        with open(save_file_name,mode='wb') as wfp:
            pickle.dump(all_data,wfp)
def build_weibo4moods(load_data_file,result_data_dir):
    train_per = 0.64
    val_per = 0.82
    dataset = pd.read_csv(load_data_file)
    dataset = sklearn.utils.shuffle(dataset)
    train_len = int(train_per*len(dataset))
    val_len = int(val_per*len(dataset))
    train_dataset = dataset.iloc[:train_len,:]
    validate_dataset = dataset.iloc[train_len:val_len,:]
    test_dataset = dataset.iloc[val_len:,:]
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,validate_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc='processing weibo4moods dataset'):
        save_file_name = os.path.join(result_data_dir,'%s_dataset.pkl'%name)
        content = list(map(list,list(tp_dataset['review'].map(jieba.cut))))
        index = tp_dataset.index.to_numpy()
        labels = tp_dataset['label'].to_numpy()
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels
        }
        with open(save_file_name,mode='wb') as wfp:
            pickle.dump(all_data,wfp)
def build_student(load_data_file,result_data_dir):
    train_per = 0.64
    val_per = 0.82
    dataset = pd.read_excel(load_data_file)
    dataset = sklearn.utils.shuffle(dataset)
    train_len = int(train_per*len(dataset))
    val_len = int(val_per*len(dataset))
    train_dataset = dataset.iloc[:train_len,:]
    validate_dataset = dataset.iloc[train_len:val_len,:]
    test_dataset = dataset.iloc[val_len:,:]
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,validate_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc='processing student dataset'):
        save_file_name = os.path.join(result_data_dir,'%s_dataset.pkl'%name)
        content = list(map(list,list(tp_dataset['content'].map(jieba.cut))))
        index = tp_dataset['总序号'].to_numpy()
        labels = tp_dataset['态度标签'].to_numpy()-1
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels
        }
        with open(save_file_name,mode='wb') as wfp:
            pickle.dump(all_data,wfp)
        
def build_clue2020emotions_dataset(dataset_dir,result_data_dir):
    names_list = ['train','validate','test']
    for name in tqdm(names_list,desc='processing student dataset'):
        load_filename = os.path.join(dataset_dir,"%s.txt"%name)
        save_file_name = os.path.join(result_data_dir,"%s_dataset.pkl"%name)
        dataset = []
        with open(load_filename,mode="r",encoding="utf-8") as rfp:
            for line in rfp:
                data_dict = json.loads(line)
                dataset.append(data_dict)
            dataset = pd.DataFrame(dataset)
            content = list(map(list,list(dataset['content'].map(jieba.cut))))
            index = dataset['id'].to_numpy()
            labels = dataset['label']
            all_data = {
                "idx":index,
                "content":content,
                "labels":labels
            }
            with open(save_file_name,mode='wb') as wfp:
                pickle.dump(all_data,wfp)
def build_goemotions_dataset(dataset_dir,result_data_dir):
    train_per = 0.64
    val_per = 0.82
    columns_list = ['admiration',
       'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
       'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
       'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
       'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
       'remorse', 'sadness', 'surprise', 'neutral']
    all_dataset = [pd.read_csv(os.path.join(dataset_dir,"goemotions_%s.csv"%str(idx))) for idx in range(1,4)]
    all_dataset = pd.concat(all_dataset)
    all_dataset = sklearn.utils.shuffle(all_dataset)
    
    train_len = int(train_per*len(all_dataset))
    val_len = int(val_per*len(all_dataset))
    train_dataset = all_dataset.iloc[:train_len,:]
    validate_dataset = all_dataset.iloc[train_len:val_len,:]
    test_dataset = all_dataset.iloc[val_len:,:]
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,validate_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc='processing student dataset'):
        save_file_name = os.path.join(result_data_dir,'%s_dataset.pkl'%name)
        content = list(map(list,list(tp_dataset['text'].map(lambda x:x.split(' ')))))
        index = tp_dataset['id']
        labels = tp_dataset[columns_list]
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels
        }
        with open(save_file_name,mode='wb') as wfp:
            pickle.dump(all_data,wfp)
def build_SST5_dataset(dataset_dir,result_data_dir):
    load_train_file = os.path.join(dataset_dir,"train.txt")
    load_valid_file = os.path.join(dataset_dir,"dev.txt")
    load_test_file = os.path.join(dataset_dir,"test.txt")
    columns = ["label","sentence"]
    train_dataset = pd.read_csv(load_train_file,sep='\t',header=None)
    train_dataset.columns = columns
    validate_dataset = pd.read_csv(load_valid_file,sep='\t',header=None)
    validate_dataset.columns = columns
    test_dataset = pd.read_csv(load_test_file,sep='\t',header=None)
    test_dataset.columns = columns
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,validate_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc='processing student dataset'):
        save_file_name = os.path.join(result_data_dir,'%s_dataset.pkl'%name)
        content = list(map(list,list(tp_dataset['sentence'].map(lambda x:x.split()))))
        index = tp_dataset.index.to_numpy()
        labels = tp_dataset["label"].to_numpy()
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels
        }
        with open(save_file_name,mode='wb') as wfp:
            pickle.dump(all_data,wfp)

def build_dictionary(result_dir,stop_words=None):
    words_dict = Dictionary()
    chars_dict = Dictionary()
    listed_names= ['train','validate']
    for name in listed_names:
        save_file_name = os.path.join(result_dir,"%s_dataset.pkl"%name)
        with open(save_file_name,mode="rb") as rfp:
            all_data = pickle.load(rfp)
        for idx in tqdm(range(len(all_data["content"])),desc='words processing file:%s'%save_file_name):
            for word in all_data["content"][idx]:
                words_dict.add(word)
            for char in "".join(all_data["content"][idx]):
                chars_dict.add(char)
    save_chars_dict_file = os.path.join(result_dir,"chars_dict.pkl")
    save_words_dict_file = os.path.join(result_dir,"words_dict.pkl")
    words_dict.add_stopwords(stop_words)
    chars_dict.save(save_chars_dict_file)
    words_dict.save(save_words_dict_file)
def build_embeddings(result_dir,load_embedding_file,save_embedding_file,head=True):
    save_words_dataset_file = os.path.join(result_dir,"chars_dict.pkl")
    save_chars_dataset_file = os.path.join(result_dir,"words_dict.pkl")

    chars_dict = Dictionary.load(save_chars_dataset_file)
    words_dict = Dictionary.load(save_words_dataset_file)
    char_embeddings = {}
    word_embeddings = {}
    with open(load_embedding_file,mode="r",encoding="utf-8") as rfp:
        if head:
            _,embedding_dim = rfp.readline().strip().split()
            embedding_dim = int(embedding_dim)
        else:
            pass
        word_nums=0
        for line in tqdm(rfp.readlines(),desc="loading embeddings"):
            data = line.split()
            word = str(data[0])
            vecs = list(map(float,data[1:]))
            word_nums += 1
            if word in words_dict:
                word_embeddings[word] = np.array(vecs,dtype=np.float)
                embedding_dim = len(vecs)
            if word in chars_dict:
                char_embeddings[word] = np.array(vecs,dtype=np.float)
                embedding_dim = len(vecs)
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
def build_stop_words(load_stop_words_file,save_stop_words_file,lang='zh'):
    words_dict = set()
    encoding = 'utf-8'
    with open(load_stop_words_file,mode="r",encoding=encoding) as rfp:
        words_dict = set(json.load(rfp))
    with open(save_stop_words_file,mode="wb") as wfp:
        pickle.dump(words_dict,wfp)
# build graph function
def build_graph(words_dict,chars_dict,save_dataset_file,save_graph_file,window_size=3,weighted_graph=True):
    with open(save_dataset_file,mode="rb") as rfp:
        raw_dataset = pickle.load(rfp)
    all_dataset = raw_dataset['content']
    
    doc_words_features = []
    doc_chars_features = []
    doc_mask_list = []
    doc_len_list = []
    doc_adj = []
    words_len_list = [len(item) for item in all_dataset]
    average_words_len = np.mean(words_len_list)
    chars_len_list = [len("".join(item)) for item in all_dataset]
    average_chars_len = np.mean(chars_len_list)
    seq_words_len = math.ceil(average_words_len)
    seq_chars_len = math.ceil(average_chars_len)
    max_words_len = max(words_len_list)
    max_chars_len = max(chars_len_list)

    dataset_len = len(all_dataset)
    for i in tqdm(range(dataset_len),desc="building graph"):
        doc_words = all_dataset[i]
        doc_chars = list("".join(doc_words))
        if len(doc_words)>seq_words_len:
            doc_words = doc_words[:seq_words_len]
        else:
            pad_len = seq_words_len - len(doc_words)# padding for each epoch
            doc_words += [words_dict.pad]*pad_len
        if len(doc_chars)>seq_chars_len:
            doc_chars = doc_chars[:seq_chars_len]
        else:
            pad_len = seq_chars_len - len(doc_chars)
            doc_chars += [words_dict.pad]*pad_len
        doc_words = [words_dict.start] + doc_words + [words_dict.end]
        doc_chars = [words_dict.start] + doc_chars + [words_dict.end]
        ################################################################
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
                    
                    value = words_dict.get_value(word_p) + words_dict.get_value(word_q)

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
        #################################################################################
        mask = []
        for word in doc_words:
            words_features.append(words_dict[word])
        mask = [0 if word in words_dict.ind2token or word in words_dict.stopwords else 1 for word in doc_words]
        mask = np.array(mask,dtype=np.int64)
        for char in doc_chars:
            chars_features.append(chars_dict[char])
        doc_words_features.append(np.array(words_features,dtype=np.int64))
        doc_chars_features.append(np.array(chars_features,dtype=np.int64))
        doc_adj.append(adj)
        doc_mask_list.append(mask)
    words_len = [len(item) for item in doc_words_features]
    chars_len = [len(item) for item in doc_chars_features]
    average_words_len = np.mean(words_len)
    average_chars_len = np.mean(chars_len)
    max_words_len = max(words_len)
    max_chars_len = max(chars_len)
    graph_data = {
        "doc_adj":doc_adj,
        "doc_words_features":np.vstack(doc_words_features),
        "doc_chars_features":np.vstack(doc_chars_features),
        "doc_masks":np.vstack(doc_mask_list),
        "labels":raw_dataset["labels"],
        "average_words_len":average_words_len,
        "average_chars_len":average_chars_len,
        "max_words_len":max_words_len,
        "max_chars_len":max_chars_len,
    }
    with open(save_graph_file,mode="wb") as wfp:
        pickle.dump(graph_data,wfp)

