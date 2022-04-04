import tensorflow as tf
import numpy as np

def glorot(shape,name=None):
    """
    Glorot & Bengio (AISTATS 2010) init.
    """
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random.uniform(shape,minval=-init_range,maxval=init_range,dtype=tf.float32)
    return tf.compat.v1.Variable(initial,name=name)
def init_vars(embedding,name=None):
    if type(embedding)==np.ndarray:
        pass
    elif type(embedding)==list:
        embedding = np.array(embedding)
    else:
        raise ValueError("Unknown type: %s"%str(type(embedding)))
    return tf.compat.v1.Variable(embedding,name=name,dtype=tf.float32)
def zeros(shape,name=None):
    """
    zeros tensors
    """
    initial = tf.zeros(shape,dtype=tf.float32)
    return tf.compat.v1.Variable(initial)
def ones(shape, name=None):
    """
    ones tensors
    """
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.compat.v1.Variable(initial, name=name)
def uniform(shape, scale=0.05, name=None):
    """
    uniform tensors
    """
    initial = tf.random.uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.compat.v1.Variable(initial, name=name)

