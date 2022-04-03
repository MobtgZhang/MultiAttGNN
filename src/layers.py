import numpy as np
import tensorflow as tf

from .base import Layer
from .inits import glorot,zeros,init_vars
from .functional import sparse_dropout,dot
from .functional import gru_unit
class Embedding(Layer):
    def __init__(self,placeholders,vocab_nums,embedding_dim,embedding=None,dropout=False,sparse_inputs=False,
                    featureless=False,**kwargs):
        super(Embedding,self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        # variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        with tf.compat.v1.varibale_scope(self.name+'_vars'):
            if embedding is None:
                if vocab_nums is not None and embedding_dim is not None:
                    self.vars['embedding'] = glorot([vocab_nums,embedding_dim],name='embedding')
                else:
                    self.vars['embedding'] = None
            else:
                self.vars['embedding'] = init_vars(embedding,"embedding")
    def _call(self,inputs):
        x_embed = tf.nn.embedding_lookup(self.vars['embedding'],inputs)
        if self.sparse_inputs:
            x_embed = sparse_dropout(x_embed,self.dropout,self.num_features_nonzero)
        else:
            x_embed = tf.nn.dropout(x_embed,rate=self.dropout)
        return x_embed
    def load_pretrain(self,embedding):
        if type(embedding) == np.ndarray:
            pass
        elif type(embedding) == list:
            embedding = np.array(embedding)
        else:
            raise ValueError("Unknown type %s"%str(type(embedding)))
        with tf.compat.v1.varibale_scope(self.name+'_vars'):
            self.vars['embedding'] = init_vars(embedding,name='embedding')                    
class Linear(Layer):
    def __init__(self,in_dim,out_dim,placeholders,dropout=False,sparse_inputs=False,
                act=tf.nn.relu,bias=False,featureless=False,**kwargs):
        super(Linear,self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        with tf.compat.v1.varibale_scope(self.name+'_vars'):
            self.vars['weights'] = glorot([in_dim,out_dim],name="weight")
            if self.bias:
                self.vars['bias'] = zeros([out_dim],name="bias")
        if self.logging:
            self._log_vars()
    def _call(self,inputs):
        x = inputs

        if self.sparse_inputs:
            x = sparse_dropout(x,self.dropout,self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x,rate=self.dropout)

        output = dot(x,self.vars['weights'],sparse=self.sparse_inputs)
        # bias 
        if self.bias:
            output += self.vars['bias']
        if self.act is not None:
            return output
        else:
            return self.act(output)

class GraphGNN(Layer):
    """
    This is the graph layer for the model.
    """
    def __init__(self,in_dim,out_dim,placeholders,dropout=False,
            sparse_inputs=False,act=tf.nn.relu,bias=False,
            featureless=False,steps=2,**kwargs):
        super(GraphGNN,self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        
        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.mask = placeholders['mask']
        self.steps = steps

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        with tf.compat.v1.variable_scope(self.name+'_vars'):
            self.vars['weight_encode'] = glorot([in_dim,out_dim],name='weight_encode')
            self.vars['weights_z0'] = glorot([out_dim,out_dim],name='weights_z0')
            self.vars['weights_z1'] = glorot([out_dim,out_dim],name='weights_z1')
            self.vars['weights_r0'] = glorot([out_dim,out_dim],name='weights_r0')
            self.vars['weights_r1'] = glorot([out_dim,out_dim],name='weights_r1')
            self.vars['weights_h0'] = glorot([out_dim,out_dim],name='weights_h0')
            self.vars['weights_h1'] = glorot([out_dim,out_dim],name='weights_h1')

            self.vars['bias_encode'] = glorot([in_dim,out_dim],name='bias_encode')
            self.vars['bias_z0'] = glorot([out_dim,out_dim],name='bias_z0')
            self.vars['bias_z1'] = glorot([out_dim,out_dim],name='bias_z1')
            self.vars['bias_r0'] = glorot([out_dim,out_dim],name='bias_r0')
            self.vars['bias_r1'] = glorot([out_dim,out_dim],name='bias_r1')
            self.vars['bias_h0'] = glorot([out_dim,out_dim],name='bias_h0')
            self.vars['bias_h1'] = glorot([out_dim,out_dim],name='bias_h1')
        

        if self.logging:
            self._log_vars()
    def _call(self, inputs):
        x = inputs

        if self.sparse_inputs:
            x = sparse_dropout(x,self.dropout,self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x,rate=self.dropout)

        # encode inputs
        x = dot(x,self.vars['weight_encode'],self.sparse_inputs) + self.vars['bias_encode']

        output = self.mask * self.act(x)
        for _ in range(self.steps):
            output = gru_unit(self.support, output, self.vars, self.act,
                              self.mask,self.dropout, self.sparse_inputs)
        return output

class Readout(Layer):
    def __init__(self,in_dim,out_dim,placeholders,dropout=0.0,
            sparse_inputs=False,act=tf.nn.relu,bias=False,**kwargs):
        super(Readout,self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.mask = placeholders['mask']

        with tf.compat.v1.variable_scope(self.name+'_vars'):
            self.vars['weight_att'] = glorot([in_dim,1],name='weight_att')
            self.vars['weight_emb'] = glorot([in_dim,in_dim],name='weight_emb')
            
            
            self.vars['bias_att'] = zeros([1],name='bias_att')
            self.vars['bias_emb'] = zeros([in_dim],name='bias_emb')
        if self.logging:
            self._log_vars()
    def _call(self, inputs):
        x = inputs

        # soft attention

        att = tf.sigmoid(dot(x,self.vars['weight_att'])+self.vars['bias_att'])
        emb = self.act(dot(x,self.vars['weight_emb'])+self.vars['bias_emb'])

        N = tf.reduce_sum(self.mask,axis=1)
        M = (self.mask-1)*1e9
        
        # graph summation
        g = self.mask * att * emb  

        g = tf.reduce_sum(g,axis=1)/N + tf.reduce_max(g+M,axis=1)
        g = tf.nn.dropout(g,rate = self.dropout)

        return g
class Softmax(Layer):
    def __init__(self,in_dim,n_class,placeholders,dropout=0.0,
                sparse_inputs=False,**kwargs):
        super(Softmax,self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        self.sparse_inputs = sparse_inputs
        self.vars['weight_mlp'] = glorot([in_dim,n_class],name='weight_mlp')
        self.vars['bias_mlp'] = glorot([n_class],name='bias_mlp')
        if self.logging:
            self._log_vars()
    def _call(self, inputs):
        x = inputs
        # classification
        output = tf.matmul(x, self.vars['weights_mlp']) + self.vars['bias_mlp']
        return output

class RelAttention(Layer):
    def __init__(self, **kwargs):
        super(RelAttention,self).__init__(**kwargs)



