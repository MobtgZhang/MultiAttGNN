import pickle
import json

class Dictionary:
    def __init__(self):
        self.name = 'default'
        self.ind2token = ['<PAD>','<START>','<END>','<UNK>',]
        self.token2ind = {'<PAD>':0,'<START>':1,'<END>':2,'<UNK>':3}
        self.token2val = {'<PAD>':1.0,'<START>':0.5,'<END>':0.5,'<UNK>':2.0}
        self.stopwords = set()
        self.start_index = 0
        self.end_index = len(self.ind2token)
    @property
    def pad(self):
        return self.ind2token[0]
    @property
    def start(self):
        return self.ind2token[1]
    @property
    def end(self):
        return self.ind2token[2]
    @property
    def unk(self):
        return self.ind2token[3]
    def add_stopwords(self,word):
        if word == None:
            return
        if type(word)==str:
            self.token2val[word] = 1.25
        elif type(word)==set:
            for w in word:
                self.token2val[w] = 1.25
        else:
            raise TypeError("Unknown model type %s"%str(type(word)))
    def __iter__(self):
        return self
    def __next__(self):
        if self.start_index < self.end_index:
            ret = self.ind2token[self.start_index]
            self.start_index += 1
            return ret
        else:
            raise StopIteration
    def __getitem__(self,item):
        if type(item) == str:
            return self.token2ind.get(item,self.token2ind[self.unk])
        elif type(item) == int:
            word = self.ind2token[item]
            return word
        else:
            raise IndexError()
    def add(self,word):
        if word not in self.token2ind:
            self.token2ind[word] = len(self.ind2token)
            self.ind2token.append(word)
            self.end_index = len(self.ind2token)
            if word in self.stopwords:
                self.token2val[word] = 1.25
            else:
                self.token2val[word] = 1.5
    def get_value(self,word):
        return self.token2val.get(word,self.token2val[self.unk])
    def save(self,save_file):
        with open(save_file,"wb") as wfp:
            data = {
                "ind2token":self.ind2token,
                "token2ind":self.token2ind,
                "token2val":self.token2val
            }
            pickle.dump(data,wfp)
    @staticmethod
    def load(load_file):
        tp_dict = Dictionary()
        with open(load_file,"rb") as rfp:
            data = pickle.load(rfp)
            tp_dict.token2ind = data["token2ind"]
            tp_dict.ind2token = data["ind2token"]
            tp_dict.token2val = data["token2val"]
            tp_dict.end_index = len(tp_dict.ind2token)
        return tp_dict
    def __contains__(self,word):
        assert type(word) == str
        return word in self.token2ind
    def __len__(self):
        return len(self.token2ind)
    def __repr__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))
    def __str__(self) -> str:
        return '{}(num_keys={})'.format(
            self.__class__.__name__,len(self.token2ind))



            
class SentimentDataset:
    def __init__(self,x_features,x_adj,y_labels):
        pass
    def __getitem__(self,idx):
        pass
    def __len__(self):
        pass

