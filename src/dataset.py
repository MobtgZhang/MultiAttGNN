import os
import pickle
from .dictionary import Dictionary

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
        graph_item = self.graph_data[index]
        graph_item["words"] = [self.words_dict[w] for w in graph_item["words"]]
        graph_item["chars"] = [self.chars_dict[w] for w in graph_item["chars"]]
        return graph_item
    def __len__(self):
        return len(self.graph_data)


