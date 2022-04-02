import os
import tensorflow as tf

from .functional import get_layer_uid

class Model(object):
    def __init__(self,**kwargs):
        allowed_kwargs = {'name','logging'}
        for arg in kwargs.keys():
            assert arg in allowed_kwargs,'Invaild keyword argument:%s'%str(arg)
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name
        logging = kwargs.get('logging',False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}
        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None
        self.embeddings = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
    
    def build(self):
        with tf.compat.v1.variable_scope(self.name):
            self._build()
        # Build sequential layer model
        self.activations = [self.inputs]
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.embeddings = self.activations[-2]
        self.ouptuts = self.activations[-1]

        # store model variables for the easy access
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name for var in variables}

        # build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)
    def _build(self):
        raise NotImplementedError
    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError
    def save(self,save_file_name,sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided!")
        saver = tf.compat.v1.train.Saver(self.vars)
        save_filename = os.path.join(save_file_name,"%s.ckpt"%self.name)
        saver.save(save_filename)
    def load(self,load_file_name,sess=None):
        saver = tf.compat.v1.train.Saver(self.vars)
        save_filename = os.path.join(load_file_name,"%s.ckpt"%self.name)
        saver.restore(sess,save_filename)
    
class Layer(object):
    def __init__(self,**kwargs):
        allowed_kwargs = {'name','logging'}
        for arg in kwargs.keys():
            assert arg in allowed_kwargs,'Invalid keyword argument: ' + arg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging',False)
        self.logging = logging
        self.sparse_inputs = False
    def _call(self,inputs):
        return inputs
    def __call__(self,inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.compat.v1.summary.histogram(self.name+'/inputs',inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.compat.v1.summary.histogram(self.name+'/outputs',outputs)
            return outputs
    def _log_vars(self):
        for var in self.vars:
            tf.compat.v1.summary.histogram(self.name+'/vars/' + var,self.vars[var])
