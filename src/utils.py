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

from .dictionary import Dictionary

def build_aichallenger_dataset(dataset_dir,reusult_dir,save_stop_words_file):
    load_train_file = os.path.join(dataset_dir,"sentiment_analysis_trainingset.csv")
    load_validate_file = os.path.join(dataset_dir,"sentiment_analysis_validationset.csv")
    train_dataset = pd.read_csv(load_train_file)
    dataset = pd.read_csv(load_validate_file)
    dataset = sklearn.utils.shuffle(dataset)
    val_len = len(dataset)//2
    val_dataset = dataset.iloc[:val_len,:]
    test_dataset = dataset.iloc[val_len:,:]
    with open(save_stop_words_file,mode="rb") as rfp:
        stopword_dict = pickle.load(rfp)
        def tp_fun(sent):
            mask = []
            for word in sent:
                if word in stopword_dict:
                    mask += [0]*len(word)
                else:
                    mask += [1]*len(word) 
            return mask
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,val_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc='processing aichallenger dataset'):
        save_dataset_file = os.path.join(reusult_dir,"%s_dataset.pkl"%name)
        if os.path.exists(save_dataset_file):
            continue
        content = list(map(list,list(tp_dataset['content'].map(jieba.cut))))
        masks = list(map(lambda sent:tp_fun(sent),content))
        labels = tp_dataset.drop(labels='id',axis=1).drop(labels='content',axis=1).to_numpy()
        idx = tp_dataset['id'].to_numpy()
        labels = labels+2
        all_process_data = {
            "idx":idx,
            "content":content,
            "labels":labels,
            "masks":masks
        }
        with open(save_dataset_file,mode="wb") as wfp:
            pickle.dump(all_process_data,wfp)
def build_wrime_dataset(load_dataset_file,result_data_dir,save_stop_words_file,label_name='writer'):
    dataset = pd.read_csv(load_dataset_file,sep='\t')
    columns_list = []
    for label in dataset.columns.values:
        if label_name in label.lower():
            columns_list.append(label)
    train_dataset = dataset[dataset['Train/Dev/Test']=='train']
    dev_dataset = dataset[dataset['Train/Dev/Test']=='dev']
    test_dataset = dataset[dataset['Train/Dev/Test']=='test']
    with open(save_stop_words_file,mode="rb") as rfp:
        stopword_dict = pickle.load(rfp)
        def tp_fun(sent):
            mask = []
            for word in sent:
                if word in stopword_dict:
                    mask += [0]*len(word)
                else:
                    mask += [1]*len(word) 
            return mask
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,dev_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc="processing WRIME-v2 dataset"):
        token_tp = Tokenizer()
        content = list(tp_dataset["Sentence"].map(lambda x:[item.surface for item in token_tp.tokenize(x)]))
        masks = list(map(lambda sent:tp_fun(sent),content))
        index = tp_dataset.index.to_numpy()
        labels = tp_dataset[columns_list].to_numpy()
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels,
            "masks":masks
        }
        save_file_name = os.path.join(result_data_dir,'%s_dataset.pkl'%name)
        with open(save_file_name,mode='wb') as wfp:
            pickle.dump(all_data,wfp)
def build_weibo4moods(load_data_file,result_data_dir,save_stop_words_file):
    train_per = 0.64
    val_per = 0.82
    dataset = pd.read_csv(load_data_file)
    dataset = sklearn.utils.shuffle(dataset)
    train_len = int(train_per*len(dataset))
    val_len = int(val_per*len(dataset))
    train_dataset = dataset.iloc[:train_len,:]
    validate_dataset = dataset.iloc[train_len:val_len,:]
    test_dataset = dataset.iloc[val_len:,:]
    with open(save_stop_words_file,mode="rb") as rfp:
        stopword_dict = pickle.load(rfp)
        def tp_fun(sent):
            mask = []
            for word in sent:
                if word in stopword_dict:
                    mask += [0]*len(word)
                else:
                    mask += [1]*len(word) 
            return mask
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,validate_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc='processing weibo4moods dataset'):
        save_file_name = os.path.join(result_data_dir,'%s_dataset.pkl'%name)
        content = list(map(list,list(tp_dataset['review'].map(jieba.cut))))
        masks = list(map(lambda sent:tp_fun(sent),content))
        index = tp_dataset.index.to_numpy()
        labels = tp_dataset['label'].to_numpy()
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels,
            "masks":masks
        }
        with open(save_file_name,mode='wb') as wfp:
            pickle.dump(all_data,wfp)
def build_student(load_data_file,result_data_dir,save_stop_words_file):
    train_per = 0.64
    val_per = 0.82
    dataset = pd.read_excel(load_data_file)
    dataset = sklearn.utils.shuffle(dataset)
    train_len = int(train_per*len(dataset))
    val_len = int(val_per*len(dataset))
    train_dataset = dataset.iloc[:train_len,:]
    validate_dataset = dataset.iloc[train_len:val_len,:]
    test_dataset = dataset.iloc[val_len:,:]
    with open(save_stop_words_file,mode="rb") as rfp:
        stopword_dict = pickle.load(rfp)
        def tp_fun(sent):
            mask = []
            for word in sent:
                if word in stopword_dict:
                    mask += [0]*len(word)
                else:
                    mask += [1]*len(word) 
            return mask
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,validate_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc='processing student dataset'):
        save_file_name = os.path.join(result_data_dir,'%s_dataset.pkl'%name)
        content = list(map(list,list(tp_dataset['content'].map(jieba.cut))))
        masks = list(map(lambda sent:tp_fun(sent),content))
        index = tp_dataset['总序号'].to_numpy()
        labels = tp_dataset['态度标签'].to_numpy()-1
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels,
            "masks":masks
        }
        with open(save_file_name,mode='wb') as wfp:
            pickle.dump(all_data,wfp)
        
def build_clue2020emotions_dataset(dataset_dir,result_data_dir,save_stop_words_file):
    names_list = ['train','validate','test']
    with open(save_stop_words_file,mode="rb") as rfp:
        stopword_dict = pickle.load(rfp)
        def tp_fun(sent):
            mask = []
            for word in sent:
                if word in stopword_dict:
                    mask += [0]*len(word)
                else:
                    mask += [1]*len(word) 
            return mask
    label_list = ['like', 'fear', 'disgust', 'anger', 'surprise', 'sadness', 'happiness']
    label_dict = {label_list[idx]:idx for idx in range(len(label_list))}
    all_dict = {
        "idx2label":label_list,
        "label2idx":label_dict
    }
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
            masks = list(map(lambda sent:tp_fun(sent),content))
            index = dataset['id'].to_numpy()
            labels = dataset['label'].map(lambda x:label_dict[x]).to_numpy()[:,np.newaxis]
            label_dict.update(data_dict)
            all_data = {
                "idx":index,
                "content":content,
                "labels":labels,
                "masks":masks
            }
            with open(save_file_name,mode='wb') as wfp:
                pickle.dump(all_data,wfp)
    save_labels_file = os.path.join(result_data_dir,"labels.pkl")
    with open(save_labels_file,mode='wb') as wfp:
        pickle.dump(all_dict,wfp)
def build_goemotions_dataset(dataset_dir,result_data_dir,save_stop_words_file):
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
    with open(save_stop_words_file,mode="rb") as rfp:
        stopword_dict = pickle.load(rfp)
        def tp_fun(sent):
            mask = []
            for word in sent:
                if word in stopword_dict:
                    mask += [0]*len(word)
                else:
                    mask += [1]*len(word) 
            return mask
    names_list = ['train','validate','test']
    dataset_list = [train_dataset,validate_dataset,test_dataset]
    for name,tp_dataset in tqdm(zip(names_list,dataset_list),desc='processing student dataset'):
        save_file_name = os.path.join(result_data_dir,'%s_dataset.pkl'%name)
        content = list(map(list,list(tp_dataset['text'].map(lambda x:x.split(' ')))))
        masks = list(map(lambda sent:tp_fun(sent),content))
        index = tp_dataset['id'].to_numpy()
        labels = tp_dataset[columns_list].to_numpy()
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels,
            "masks":masks
        }
        with open(save_file_name,mode='wb') as wfp:
            pickle.dump(all_data,wfp)
def build_SST5_dataset(dataset_dir,result_data_dir,save_stop_words_file):
    load_train_file = os.path.join(dataset_dir,"train.txt")
    load_valid_file = os.path.join(dataset_dir,"dev.txt")
    load_test_file = os.path.join(dataset_dir,"test.txt")
    with open(save_stop_words_file,mode="rb") as rfp:
        stopword_dict = pickle.load(rfp)
        def tp_fun(sent):
            mask = []
            for word in sent:
                if word in stopword_dict:
                    mask += [0]*len(word)
                else:
                    mask += [1]*len(word) 
            return mask
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
        masks = list(map(lambda sent:tp_fun(sent),content))
        index = tp_dataset.index.to_numpy()
        labels = tp_dataset["label"].to_numpy()
        all_data = {
            "idx":index,
            "content":content,
            "labels":labels,
            "masks":masks
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
    print("words dictionary",len(words_dict))
    print("chars dictionary",len(chars_dict))
    save_chars_dict_file = os.path.join(result_dir,"chars_dict.pkl")
    save_words_dict_file = os.path.join(result_dir,"words_dict.pkl")
    words_dict.add_stopwords(stop_words)
    chars_dict.save(save_chars_dict_file)
    words_dict.save(save_words_dict_file)
def build_embeddings(result_dir,load_embedding_file,save_embedding_file,head=True):
    save_chars_dataset_file = os.path.join(result_dir,"chars_dict.pkl")
    save_words_dataset_file = os.path.join(result_dir,"words_dict.pkl")

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
            w = str(data[0])
            vecs = list(map(float,data[1:]))
            word_nums += 1
            if w in words_dict:
                word_embeddings[w] = np.array(vecs,dtype=np.float)
                embedding_dim = len(vecs)
            if w in chars_dict:
                char_embeddings[w] = np.array(vecs,dtype=np.float)
                embedding_dim = len(vecs)
    for w in tqdm(words_dict,"fixing word embeddings"):
        if w not in word_embeddings:
            word_embeddings[w] = np.random.uniform(-0.01,0.01,embedding_dim)
    for c in tqdm(chars_dict,"fixing char embeddings"):
        if c not in char_embeddings:
            char_embeddings[c] = np.random.uniform(-0.01,0.01,embedding_dim)
    print("words:",len(word_embeddings),"words dictionary:",len(words_dict))
    print("chars",len(char_embeddings),"chars dictionary:",len(chars_dict))
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
    graph_data = []
    dataset_len = len(all_dataset)
    doc_len_list = []
    for idx in tqdm(range(dataset_len),desc="building graph"):
        doc_words = all_dataset[idx]
        doc_chars = list("".join(doc_words))
        doc_nodes = list(set(doc_words))

        # doc_words = [words_dict.start] + doc_words + [words_dict.end]
        # doc_chars = [words_dict.start] + doc_chars + [words_dict.end]
        ################################################################
        doc_len = len(doc_nodes)
        doc_len_list.append(doc_len)

        doc_word2id_map = {}
        for j in range(len(doc_nodes)):
            doc_word2id_map[doc_nodes[j]] = j
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
            for p in range(1, len(win)):
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

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(p)
            col.append(q)
            weight.append(word_pair_count[key] if weighted_graph else 1.)
        adj = ssp.csr_matrix((weight, (row, col)), shape=(doc_len,doc_len))
        #################################################################################
        if len(doc_chars)!=len(raw_dataset["masks"][idx]):
            print(idx)
        tp_dict = {
            "idx":raw_dataset["idx"][idx],
            "adj":adj,
            "words":doc_nodes,
            "chars":doc_chars,
            "chars-mask":raw_dataset["masks"][idx],
            "labels":raw_dataset["labels"][idx],
        }
        graph_data.append(tp_dict)
    with open(save_graph_file,mode="wb") as wfp:
        pickle.dump(graph_data,wfp)

def construct_feed_dict(item,placeholders):
    """
        Construct feed dictionary.
        item: idx,(adj,adj_words,adj_mask),(chars,chars_mask),labels
    """
    feed_dict = dict()
    feed_dict.update({placeholders['adj']: item[1][0]})
    feed_dict.update({placeholders['adj-words']: item[1][1]})
    feed_dict.update({placeholders['adj-mask']: item[1][2]})
    feed_dict.update({placeholders['chars']: item[2][0]})
    feed_dict.update({placeholders['chars-mask']: item[2][1]})
    feed_dict.update({placeholders['labels']:item[3]})
    feed_dict.update({placeholders['num_features_nonzero']:item[1][1].shape[1]})
    return feed_dict
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
def batchfy(batch):
    batch_size = len(batch)
    idx = [ex["idx"] for ex in batch]
    adj = [ex["adj"] for ex in batch]
    adj_words = [ex["words"] for ex in batch]
    max_words_len = max([len(ex["words"]) for ex in batch])
    max_chars_len = max([len(ex["chars"]) for ex in batch])
    # initialize the tensor
    adj_tokens = np.zeros((batch_size,max_words_len),dtype=np.int64)
    adj_mask = np.zeros((batch_size,max_words_len),dtype=np.int64)
    chars = np.zeros((batch_size,max_chars_len),dtype=np.int64)
    chars_mask = np.zeros((batch_size,max_chars_len),dtype=np.int64)
    labels = np.array([ex["labels"] for ex in batch],dtype=np.int64)
    # add value
    for i in range(batch_size):
        adj_normalized = normalize_adj(adj[i]) # no self-loop
        chars_len = len(batch[i]["chars"])
        words_len = len(batch[i]["words"])
        pad = max_words_len - words_len # padding for each epoch
        adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
        adj[i] = adj_normalized
        adj_tokens[i][:words_len] = np.array(batch[i]["words"],dtype=np.int64)
        adj_mask[i][:words_len] = np.ones((words_len,),dtype=np.int64)
        chars[i][:chars_len] = np.array(batch[i]["chars"],dtype=np.int64)
        chars_mask[i][:chars_len] = np.array(batch[i]["chars-mask"],dtype=np.int64)
    adj = np.array(adj,dtype=np.float64)
    return idx,(adj,adj_words,adj_mask),(chars,chars_mask),labels
def load_embeddings(result_data_dir):
    load_embeddings_file = os.path.join(result_data_dir,"embedding.pkl")
    load_chars_file = os.path.join(result_data_dir,"chars_dict.pkl")
    load_words_file = os.path.join(result_data_dir,"words_dict.pkl")
    words_dict = Dictionary.load(load_words_file)
    chars_dict = Dictionary.load(load_chars_file)
    with open(load_embeddings_file,mode="rb") as rfp:
        embedding_data = pickle.load(rfp)
        embedding_dim = embedding_data["embedding-dim"]
        num_chars = len(embedding_data["char-embeddings"])
        num_words = len(embedding_data["word-embeddings"])
        print(embedding_data.keys(),(len(embedding_data["char-embeddings"]),len(chars_dict)),
                (len(embedding_data["word-embeddings"]),len(words_dict)))
        char_embeddings = np.zeros((num_chars,embedding_dim),dtype=np.float64)
        word_embeddings = np.zeros((num_words,embedding_dim),dtype=np.float64)
        for word in tqdm(embedding_data["word-embeddings"],desc='word embedding'):
            word_embeddings[words_dict[word]] = embedding_data["word-embeddings"][word]
            
        for char in tqdm(embedding_data["char-embeddings"],desc='char embedding'):
            char_embeddings[chars_dict[word]] = embedding_data["char-embeddings"][char]
    return word_embeddings,char_embeddings
        


