import numpy as np
import tensorflow as tf

from .base import Layer
from .inits import glorot,zeros,init_vars,ones,normal,constant
from .functional import sparse_dropout,dot
from .functional import gru_unit
class Embedding(Layer):
    def __init__(self,placeholders,vocab_nums=None,embedding_dim=None,embedding=None,dropout=True,sparse_inputs=False,
                    featureless=False,**kwargs):
        super(Embedding,self).__init__(**kwargs)
        self.inputs = placeholders['adj-words']
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        # variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']
        
        with tf.compat.v1.variable_scope(self.name+'_vars'):
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
        with tf.compat.v1.variable_scope(self.name+'_vars'):
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

class GGNNLayer(Layer):
    """
    This is the graph layer for the model.
    """
    def __init__(self,in_dim,out_dim,placeholders,dropout=False,
            sparse_inputs=False,act=tf.nn.relu,featureless=False,steps=2,**kwargs):
        super(GGNNLayer,self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        
        self.act = act
        self.support = placeholders['adj']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.mask =  tf.expand_dims(placeholders['adj-mask'], -1)
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
        
            self.vars['bias_encode'] = zeros([out_dim],name='bias_encode')
            self.vars['bias_z0'] = zeros([out_dim],name='bias_z0')
            self.vars['bias_z1'] = zeros([out_dim],name='bias_z1')
            self.vars['bias_r0'] = zeros([out_dim],name='bias_r0')
            self.vars['bias_r1'] = zeros([out_dim],name='bias_r1')
            self.vars['bias_h0'] = zeros([out_dim],name='bias_h0')
            self.vars['bias_h1'] = zeros([out_dim],name='bias_h1')
        

        if self.logging:
            self._log_vars()
    def _call(self, inputs):
        x = inputs

        if self.sparse_inputs:
            x = sparse_dropout(x,self.dropout,self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x,rate=self.dropout)

        # encode inputs
        x = self.mask * self.act(x)
        output = dot(x,self.vars['weight_encode'],self.sparse_inputs) + self.vars['bias_encode']
        for _ in range(self.steps):
            output = gru_unit(self.support, output, self.vars, self.act,
                              self.mask,self.dropout, self.sparse_inputs)
        return output

class ReadoutLayer(Layer):
    def __init__(self,in_dim,out_dim,placeholders,dropout=False,
            sparse_inputs=False,act=tf.nn.relu,**kwargs):
        super(ReadoutLayer,self).__init__(**kwargs)
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.mask = tf.expand_dims(placeholders['adj-mask'], -1)

        with tf.compat.v1.variable_scope(self.name+'_vars'):
            self.vars['weight_att'] = glorot([in_dim,1],name='weight_att')
            self.vars['weight_emb'] = glorot([in_dim,in_dim],name='weight_emb')
            self.vars['weight_mlp'] = glorot([in_dim,out_dim],name='weight_mlp')
            
            
            self.vars['bias_att'] = zeros([1],name='bias_att')
            self.vars['bias_emb'] = zeros([in_dim],name='bias_emb')
            self.vars['bias_mlp'] = zeros([out_dim],name='bias_mlp')
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

        # classification
        out = tf.matmul(g, self.vars['weight_mlp']) + self.vars['bias_mlp']
        return out
class RCNNLayer(Layer):
    def __init__(self,in_dim,hid_dim,out_dim,rnn_type,placeholders,dropout=True,**kwargs):
        super(RCNNLayer,self).__init__(**kwargs)
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.rnn_type = rnn_type
        self.context_dim = self.in_dim + self.hid_dim*2 
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        def _get_cell(rnn_type,rnn_dim):
            if rnn_type == "vanilla":
                return tf.nn.rnn_cell.BasicRNNCell(rnn_dim)
            elif rnn_type == "lstm":
                return tf.nn.rnn_cell.BasicLSTMCell(rnn_dim)
            elif rnn_type == "gru":
                return tf.nn.rnn_cell.GRUCell(rnn_dim)
            else:
                raise TypeError("Unknown type %s"%rnn_type)
        with tf.compat.v1.variable_scope(self.name+'_vars'):
            fw_cell = _get_cell(rnn_type,hid_dim)
            self.fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=1-self.dropout)
            bw_cell = _get_cell(rnn_type,hid_dim)
            self.bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=1-self.dropout)        
            self.vars["fc-weight"] = glorot(shape=(self.context_dim,self.hid_dim),name="fc-weight")
            self.vars["fc-bias"] = glorot(shape=(self.hid_dim),name="fc-bias")
            self.vars['weight_mlp'] = glorot([hid_dim,out_dim],name='weight_mlp')
            self.vars['bias_mlp'] = zeros([out_dim],name='bias_mlp')
    def _call(self, inputs):
        (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.fw_cell,
                                                                             cell_bw=self.bw_cell,
                                                                             inputs=inputs,dtype=tf.float32)
        shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
        c_left = tf.concat([tf.zeros(shape), output_fw[:, :-1]], axis=1, name="context_left")
        c_right = tf.concat([output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")
        last = tf.concat([c_left, inputs, c_right], axis=2, name="last")
        fc_lin = tf.nn.relu(dot(last,self.vars["fc-weight"])+self.vars["fc-bias"])
        fc_pool = tf.reduce_max(fc_lin, axis=1)
        out = dot(fc_pool,self.vars["weight_mlp"])+self.vars["bias_mlp"]
        return out

class CNNLayer(Layer):
    def __init__(self,in_dim,num_filters,placeholders,filter_sizes=(2,3,4),dropout=False,
            sparse_inputs=False,**kwargs):
        super(CNNLayer,self).__init__(**kwargs)
        self.filter_sizes = filter_sizes
        self.in_dim = in_dim
        self.num_filters = num_filters
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
        # Convolution and maxpooling layer
        self.pool_weights = []
        for idx, filter_size in enumerate(self.filter_sizes):
            with tf.compat.v1.variable_scope(self.name+'_vars'):
                filter_shape = [filter_size, self.in_dim,1,self.num_filters]
                weight_name = "weight-%d-convolution"%idx
                bias_name = "bias-%d-convolution"%idx
                self.vars[weight_name] = normal(filter_shape,mean=0.0,stddev=0.1,name=weight_name)
                self.vars[bias_name] = constant([self.num_filters],0.1,name=bias_name)
        # Full connected layer 
        num_filters_total = self.num_filters * len(self.filter_sizes)
        with tf.compat.v1.variable_scope(self.name+'_vars'):
            self.vars["weight-mlp"] = glorot(shape=[num_filters_total,self.num_classes],name="weight-mlp")
            self.vars["bias-mlp"] = constant(shape=[self.n_class],value=0.1,name="bias-mlp")
    def _call(self, inputs):
        seq_length = inputs.shape[1]
        x_inputs = tf.expand_dims(inputs,-1)
        pooled_outputs = []
        # Convolution and maxpooling layer
        for idx,filter_size in enumerate(self.filter_sizes):
            weight_name = "weight-%d-convolution"%idx
            bias_name = "bias-%d-convolution"%idx
            conv = tf.nn.conv2d(
                    x_inputs,self.vars[weight_name],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
            hidden = tf.nn.relu(tf.nn.bias_add(conv, self.vars[bias_name]), name="relu")
            pooled = tf.nn.max_pool(
                    hidden,
                    ksize=[1, seq_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool"
                )
            pooled_outputs.append(pooled)
        # concat the layer with every tensor
        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        h_drop = tf.nn.dropout(h_pool_flat,rate = self.dropout)

        # Full connect
        out = tf.nn.xw_plus_b(h_drop, self.vars["weight-mlp"],self.vars["bias-mlp"],name="scores")
        return out

class MultiAttLayer(Layer):
    def __init__(self,**kwargs):
        super(MultiAttLayer,self).__init__(**kwargs)
class SFULayer(Layer):
    def __init__(self,in_dim,fusion_dim,**kwargs):
        super(SFULayer,self).__init__(**kwargs)
        self.in_dim = in_dim
        self.fusion_dim = fusion_dim
        with tf.compat.v1.variable_scope(self.name+'_vars'):
            self.vars['weight-r'] = glorot(shape=(in_dim+fusion_dim,in_dim),name='weight-r')
            self.vars['weight-g'] = glorot(shape=(in_dim+fusion_dim,in_dim),name='weight-g')

            self.vars['bias-r'] = constant(shape=[in_dim],value=0.1,name='bais-r')
            self.vars['bias-g'] = constant(shape=[in_dim],value=0.1,name='bais-g')
    def _call(self,inputs,fusions):
        rf = tf.concat([inputs,fusions],2)
        r = tf.nn.tanh(dot(rf,vars['weight-r'])+self.vars['bias-r'])
        g = tf.nn.sigmoid(dot(rf,vars['weight-g'])+self.vars['bias-g'])
        o = g*r + (1-g)*inputs
        return o
class MatchLayer(Layer):
    def __init__(self,in_dim,label_size,**kwargs):
        super(MatchLayer,self).__init__(**kwargs)
        self.label_size = label_size
        self.in_dim = in_dim
    def _call(self, inputs):
        return super()._call(inputs)


class DPCNNLayer(Layer):
    def __init__(self,in_dim,num_filters,kernel_size,n_class,placeholders,dropout=True,**kwargs):
        super(DPCNNLayer,self).__init__(**kwargs)
        self.num_filters = num_filters
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.n_class = n_class
        self.keep_prob = 1-dropout
        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0
    def _call(self, inputs):
        x_inputs = tf.expand_dims(inputs, axis=-1)  # [batch-size,seq_len,embedding_dim,1]
        with tf.name_scope(self.name+"_embedding_vars"):
            # region_embedding  # [batch,seq-3+1,1,250]
            region_embedding = tf.layers.conv2d(x_inputs, self.num_filters,[self.kernel_size, self.in_dim])
            pre_activation = tf.nn.relu(region_embedding, name='preactivation')
        with tf.name_scope(self.name+"_conv3_0_vars"):
            conv3 = tf.layers.conv2d(pre_activation, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)
        with tf.name_scope(self.name+"_conv3_1_vars"):
            conv3 = tf.layers.conv2d(conv3, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        # resdul
        conv3 = conv3 + region_embedding
        with tf.name_scope(self.name+"_pool1_vars"):
            pool = tf.pad(conv3, paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
            pool = tf.nn.max_pool(pool, [1, 3, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

        with tf.name_scope(self.name+"_conv3_2_vars"):
            conv3 = tf.layers.conv2d(pool, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        with tf.name_scope(self.name+"_conv3_3_vars"):
            conv3 = tf.layers.conv2d(conv3, self.num_filters, self.kernel_size,
                                     padding="same", activation=tf.nn.relu)
            conv3 = tf.layers.batch_normalization(conv3)

        # resdul
        conv3 = conv3 + pool
        seq_length = conv3.shape[1]
        pool_size = int((seq_length - 3 + 1)/2)
        conv3 = tf.layers.max_pooling1d(tf.squeeze(conv3, [2]), pool_size, 1)
        conv3 = tf.squeeze(conv3, [1]) # [batch,250]
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        with tf.name_scope(self.name+"_out_vars"):
            # classify
            logits = tf.layers.dense(conv3, self.n_class, name='fc2')
        return logits
class RelAttention(Layer):
    def __init__(self, **kwargs):
        super(RelAttention,self).__init__(**kwargs)


class MultiInputLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
