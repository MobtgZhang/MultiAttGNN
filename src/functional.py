import tensorflow as tf

_LAYER_UIDS = {}
def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]
def sparse_dropout(x, rate, noise_shape):
    keep_prob = 1-rate
    """
    Dropout for sparse tensors.
    """
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)
def sparse_dense_matmul_batch(sp_a, b):

    def map_function(x):
        i, dense_slice = x[0], x[1]
        sparse_slice = tf.sparse.reshape(tf.sparse.slice(
            sp_a, [i, 0, 0], [1, sp_a.dense_shape[1], sp_a.dense_shape[2]]),
            [sp_a.dense_shape[1], sp_a.dense_shape[2]])
        mult_slice = tf.sparse.matmul(sparse_slice, dense_slice)
        return mult_slice

    elems = (tf.range(0, sp_a.dense_shape[0], delta=1, dtype=tf.int64), b)
    return tf.map_fn(map_function, elems, dtype=tf.float32, back_prop=True)
def dot(x, y, sparse=False):
    """
	Wrapper for 3D tf.matmul (sparse vs dense).
    """
    if sparse:
        res = sparse_dense_matmul_batch(x, y)
    else:
        res = tf.einsum('bij,jk->bik', x, y) # tf.matmul(x, y)
    return res
def gru_unit(support, x, var, act, mask, dropout, sparse_inputs=False):
    """GRU unit with 3D tensor inputs."""
    # message passing
    support = tf.nn.dropout(support, rate=dropout) # optional
    a = tf.matmul(support, x)

    # update gate        
    z0 = dot(a, var['weights_z0'], sparse_inputs) + var['bias_z0']
    z1 = dot(x, var['weights_z1'], sparse_inputs) + var['bias_z1'] 
    z = tf.sigmoid(z0 + z1)
    
    # reset gate
    r0 = dot(a, var['weights_r0'], sparse_inputs) + var['bias_r0']
    r1 = dot(x, var['weights_r1'], sparse_inputs) + var['bias_r1']
    r = tf.sigmoid(r0 + r1)

    # update embeddings    
    h0 = dot(a, var['weights_h0'], sparse_inputs) + var['bias_h0']
    h1 = dot(r*x, var['weights_h1'], sparse_inputs) + var['bias_h1']
    h = act(mask * (h0 + h1))
    
    return h*z + x*(1-z)

