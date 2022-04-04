import numpy as np
import tensorflow as tf

def get_tf_args():
    flags = tf.app.flags
    FLAGS = tf.app.flags.FLAGS 
    flags.DEFINE_string('result_dir','./result','Dataset string')
    flags.DEFINE_string('data_dir','./data','Dataset string')
    flags.DEFINE_string('dataset','CLUE2020Emotions','Dataset string')
    flags.DEFINE_integer('seed',1234,'Model training seed')
    flags.DEFINE_integer('seq_len',384,'The length of the content.')
    # Set random seed
    np.random.seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)

    # Settings
    flags.DEFINE_string('model', 'gnning', 'Model string.') 
    flags.DEFINE_float('learning_rate', 0.05, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('batch_size', 4096, 'Size of batches per epoch.') 
    flags.DEFINE_integer('hid_dim', 100, 'Dimension of input.') 
    flags.DEFINE_integer('in_dim', 100, 'Dimension of input.') 
    flags.DEFINE_integer('embedding_dim', 200, 'Dimension of embedding layer.')
    flags.DEFINE_integer('steps', 2, 'Number of graph layers.')
    flags.DEFINE_float('dropout', 0.2, 'Dropout rate .')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.') # 5e-4
    flags.DEFINE_integer('num_filters',5,'The CNN model for num_filters')
    flags.DEFINE_string('filter_size','2,3,4','The CNN model for filter_size')
    return FLAGS


