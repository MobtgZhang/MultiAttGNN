import tensorflow as tf

from  .base import Layer
from .functional import sparse_dropout,dot
from .inits import glorot,zeros,ones

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
            self.vars['weights_Wz'] = glorot([out_dim,out_dim],name='weights_Wz')
            self.vars['weights_Uz'] = glorot([out_dim,out_dim],name='weights_Uz')
            self.vars['weights_Wr'] = glorot([out_dim,out_dim],name='weights_Wr')
            self.vars['weights_Ur'] = glorot([out_dim,out_dim],name='weights_Ur')
            self.vars['weights_Wh'] = glorot([out_dim,out_dim],name='weights_Wh')
            self.vars['weights_Uh'] = glorot([out_dim,out_dim],name='weights_Uh')
        
            self.vars['bias-z'] = zeros([out_dim],name='bias_encode')
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
        x = dot(x,self.vars['weight_encode'],self.sparse_inputs) + self.vars['bias_encode']
        
        output = self.mask * self.act(x)
        for _ in range(self.steps):
            output = gru_unit(output,self.support,self.vars, self.act,
                              self.mask,self.dropout, self.sparse_inputs)
        return output
def gru_unit(inputs,support,vars,act,mask,dropout,sparse_inputs):
    support = tf.nn.dropout(support, rate=dropout)
    val_a = tf.matmul(support,inputs) + vars['bias-val-a']
    val_z = act(dot(a,vars['weights_Wz'],sparse_inputs)+dot(inputs,vars['weights_Uz'],sparse_inputs)+vars['bias-z'])
    val_r = act(dot(a,vars['weights_Wr'],sparse_inputs)+dot(inputs,vars['weights_Ur'],sparse_inputs)+vars['bias-r'])
    val_h = tf.tanh(dot(val_a,vars['weights_Wh'])+dot(val_r*inputs,vars["weights_Uh"])+vars['bias-h'])
    val_h = val_h*mask
    val_h = (1-val_z)*inputs + val_z*val_h
    return val_h
