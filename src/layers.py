import tensorflow as tf


# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    def __init__(self,**kwargs) -> None:
        allowed_kwargs = {'name','logging'}
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))

        self.name = name
        self.vars = {}
        logging = kwargs.get('logging')
        self.logging = logging
        self.sparse_inputs = False
    def _call(self,inputs):
        return inputs
    def __call__(self,inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.compat.v1.summary.histogram(self.name + '/inputs',inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.compat.v1.summary.histogram(self.name+'/outputs',outputs)
            return outputs
    def _log_vars(self,):
        for var in self.vars:
            tf.compat.v1.summary.histogram(self.name+'/vars'+var,self.vars[var])
class Dense(Layer):
    def __init__(self,input_dim,output_dim,placeholders,dropout=0.0,sparse_inputs=False,
                act=tf.nn.relu,bias=False,featureless=False,**kwargs):
        super(Dense,self).__init__(**kwargs)
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
            self.vars['weights'] = None
            