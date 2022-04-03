import tensorflow as tf

from .base import Model
from .layers import Embedding
# classic model
class CNNModel(Model):
    r"""
    The implementation for the TextCNN model.
    paper title:Convolutional Neural Networks for Sentence Classification.
    paper web site:https://arxiv.org/abs/1408.5882
    """
    def __init__(self,in_dim,placeholders,vocab_nums=None,embedding_dim=None,**kwargs):
        super(CNNModel,self).__init__(**kwargs)
        self.words_embedding = Embedding(placeholders,vocab_nums,embedding_dim)
        self.chars_embedding = Embedding(placeholders,vocab_nums,embedding_dim)
    def load_embeddings(self,embedding = None):
        self.embedding.load_pretrain(embedding)
class BiLSTMModel(Model):
    r"""
    The implementation for the TextRNN model.
    paper title:A Bi-LSTM-RNN Model for Relation Classification Using Low-Cost Sequence Features.
    paper web site:https://arxiv.org/abs/1605.05101

    paper title: Recurrent Neural Network for Text Classification with Multi-Task Learning.
    paper web site:https://arxiv.org/abs/1605.05101
    """
    def __init__(self, **kwargs):
        super(BiLSTMModel,self).__init__(**kwargs)
class BiLSTMAttention(Model):
    r"""
    The implementation for the BiLSTM-Attention model.
    refer paper title: Feed-Forward Networks with Attention Can Solve Some Long-Term Memory Problems
    paper web site: https://arxiv.org/abs/1512.08756
    
    refer paper title: Neural Machine Translation by Jointly Learning to Align and Translate
    paper web site: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, **kwargs):
        super(BiLSTMAttention,self).__init__(**kwargs)
class RCNN(Model):
    r"""
    The implementation for the TextRCNN model.
    paper title: Recurrent Convolutional Neural Networks for Text Classification
    paper web site: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552
    """
    def __init__(self, **kwargs):
        super(RCNN,self).__init__(**kwargs)
class DPCNN(Model):
    r"""
    The implementation for the DPCNN model.
    paper title: Deep Pyramid Convolutional Neural Networks for Text Categorization
    paper web site: https://aclanthology.org/P17-1052.pdf
    """
    def __init__(self, **kwargs):
        super(DPCNN,self).__init__(**kwargs)
class HANNet(Model):
    r"""
    The implementation for the HANNet model.
    paper title: Hierarchical Attention Networks for Document Classification
    paper web site: https://aclanthology.org/N16-1174/
    """
    def __init__(self, **kwargs):
        super(HANNet,self).__init__(**kwargs)
class CapsuleNet(Model):
    r"""
    The implementation for the CapsuleNet model.
    paper title: Investigating Capsule Networks with Dynamic Routing for Text Classification
    paper web site: https://arxiv.org/abs/1804.00538
    """
    def __init__(self, **kwargs):
        super(CapsuleNet,self).__init__(**kwargs)
class TextGCN(Model):
    r"""
    The implementation for the TextGCN model.
    paper titile: Text Level Graph Neural Network for Text Classification
    paper web site: http://arxiv.org/abs/1910.02356
    """
    def __init__(self, **kwargs):
        super(TextGCN,self).__init__(**kwargs)
class DCCNN(Model):
    r"""
    The implementation for the DCCNN model.
    paper title: Densely Connected CNN with Multi-scale Feature Attention for Text Classification
    paper web site: https://www.ijcai.org/proceedings/2018/0621.pdf
    """
class GNNING(Model):
    r"""
    The implementation for the GNNING model.
    paper title: Every Document Owns Its Structure: Inductive Text Classification via Graph Neural Networks
    paper web site: http://arxiv.org/abs/2004.13826v2
    """
    def __init__(self, **kwargs):
        super(GNNING,self).__init__(**kwargs)
# our model for the project
class MultiAttGNN(Model):
    r"""
    

    """
    def __init__(self, **kwargs):
        super(MultiAttGNN,self).__init__(**kwargs)
