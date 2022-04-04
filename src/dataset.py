import copy
import pickle
import numpy as np

class Dataset(object):
    """An abstract class representing a Dataset.

    All other datasets should subclass it. All subclasses should override
    ``__len__``, that provides the size of the dataset, and ``__getitem__``,
    supporting integer indexing in range from 0 to len(self) exclusive.
    """
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
class RegularDataset(Dataset):
    def __init__(self,load_graph_file,words_dict,chars_dict):
        super(RegularDataset,self).__init__()
        with open(load_graph_file,mode="rb") as rfp:
            self.graph_data = pickle.load(rfp)
        self.words_dict = words_dict
        self.chars_dict = chars_dict
    def __getitem__(self, index):
        # graph_item = self.graph_data[index]
        graph_item = copy.deepcopy(self.graph_data[index])
        graph_item["words"] = [self.words_dict[w] for w in graph_item["words"]]
        graph_item["chars"] = [self.chars_dict[w] for w in graph_item["chars"]]
        return graph_item
    def __len__(self):
        return len(self.graph_data)

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
    max_words_len = max([len(ex["words"]) for ex in batch])
    max_chars_len = max([len(ex["chars"]) for ex in batch])
    # initialize the tensor
    adj_words = np.zeros((batch_size,max_words_len),dtype=np.int64)
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
        adj_words[i][:words_len] = np.array(batch[i]["words"],dtype=np.int64)
        adj_mask[i][:words_len] = np.ones((words_len,),dtype=np.int64)
        chars[i][:chars_len] = np.array(batch[i]["chars"],dtype=np.int64)
        chars_mask[i][:chars_len] = np.array(batch[i]["chars-mask"],dtype=np.int64)
    adj = np.array(adj,dtype=np.float64)
    return idx,(adj,adj_words,adj_mask),(chars,chars_mask),labels