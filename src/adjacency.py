import math
import numpy as np
from tqdm import tqdm

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
def preprocess_add_apdding(graph_data):
    """
        "x_adj":x_adj,
        "x_words_features":x_words_features,
        "x_chars_features":x_chars_features,
        "x_masks":x_masks,
        "labels":raw_dataset["labels"]
    """
    adj_list = graph_data["x_adj"]
    x_words_features = graph_data["x_words_features"]
    x_chars_features = graph_data["x_chars_features"]
    seq_words_len = math.ceil(graph_data["average_words_len"])
    seq_chars_len = math.ceil(graph_data["average_chars_len"])
    data_len = len(adj_list)
    mask_list = []  # mask for padding

    for i in tqdm(range(data_len)):
        adj_normalized = normalize_adj(adj_list[i]) # no self-loop
        x_words = x_words_features[i]
        x_chars = x_chars_features[i]
        if adj_normalized.shape[0]>seq_words_len:
            adj_normalized = adj_normalized[:seq_words_len,:seq_words_len]
            real_words_len = seq_words_len
            x_words = x_words[:seq_words_len]
        else:
            pad_len = seq_words_len - adj_normalized.shape[0]# padding for each epoch
            real_words_len = adj_normalized.shape[0]
            adj_normalized = np.pad(adj_normalized, ((0,pad_len),(0,pad_len)), mode='constant')
            x_words +=[0]*pad_len
        if len(x_chars)>seq_chars_len:
            x_chars = x_chars[:seq_chars_len]
        else:
            pad_len = seq_chars_len - len(x_chars)
            x_chars += [0]*pad_len
        mask = np.zeros((seq_words_len,))
        mask[:real_words_len] = 1.
        mask_list.append(mask)
        adj_list[i] = adj_normalized
        print(len(x_words))
        x_words_features[i] = np.array(x_words,dtype=np.int64)
        x_chars_features[i] = np.array(x_chars,dtype=np.int64)
        if i>500:
            break
    graph_data["x_adj"] = adj_list
    graph_data["masks"] = np.vstack(mask_list)
    print(graph_data["masks"].shape)
    graph_data["x_words_features"] = np.vstack(x_words_features)
    print(graph_data["x_chars_features"].shape)
    graph_data["x_chars_features"] = np.vstack(x_chars_features)
    print(graph_data["x_chars_features"].shape)


