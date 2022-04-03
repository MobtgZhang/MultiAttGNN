import pickle
import os
import time

import numpy as np
import tensorflow as tf

from config import get_tf_args
from src.utils import construct_feed_dict
import src.metrics as metrics
def evaluate(sess,model,features, support, mask, labels, placeholders):
    s_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, mask, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels], feed_dict=feed_dict_val)
    e_test = time.time()
    t_sect = e_test - s_test
    return outs_val,t_sect
def main():
    FLAGS = get_tf_args()
    # load embeddings and dataset 
    result_data_dir = os.path.join(FLAGS.result_dir,FLAGS.dataset)
    load_chars_dict_file = os.path.join(result_data_dir,"chars_dict.pkl")     
    load_words_dict_file = os.path.join(result_data_dir,"words_dict.pkl") 
    load_embeddings_file = os.path.join(result_data_dir,"embedding.pkl")
    with open(load_embeddings_file,mode="rb") as rfp:
        embedding_data = pickle.load(rfp)
        char_embeddings = embedding_data["char-embeddings"]
        word_embeddings = embedding_data["word-embeddings"]
    load_graph_file = os.path.join(result_data_dir,"train_graph.pkl")
    load_validate_graph_file = os.path.join(result_data_dir,"validate_graph.pkl")
    with open(load_graph_file,mode="rb") as rfp:
        train_graph_data = pickle.load(rfp)
    with open(load_validate_graph_file,mode="rb") as rfp:
        validate_graph_data = pickle.load(rfp)
    
    
    if FLAGS.model == 'gnn':
        # support = [preprocess_adj(adj)]
        # num_supports = 1
        model = GNN()
    elif FLAGS.model == 'gcn_cheby': # not used
        # support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model = GNN()
    elif FLAGS.model == 'dense': # not used
        # support = [preprocess_adj(adj)]
        num_supports = 1
        model = MLP()
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    # the defination of the placeholders
    placeholders = {
        'support': tf.compat.v1.placeholder(tf.float32, shape=(None, None, None)),
        'features': tf.compat.v1.placeholder(tf.float32, shape=(None, None, FLAGS.input_dim)),
        'mask': tf.compat.v1.placeholder(tf.float32, shape=(None, None, 1)),
        'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, train_y.shape[1])),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
    }
    # Initialize session
    sess = tf.compat.v1.Session()

    # merged = tf.summary.merge_all()
    # writer = tf.summary.FileWriter('logs/', sess.graph)

    # Init variables
    sess.run(tf.compat.v1.global_variables_initializer())

    cost_val = []
    best_val = 0
    best_epoch = 0
    best_acc = 0
    best_cost = 0
    test_doc_embeddings = None
    preds = None
    labels = None

    print('train start...')

    dataset_len = len(train_y)
    for epoch in range(FLAGS.eopchs):
        start_time = time.time()

        # training step for the model
        indices = np.arange(0,dataset_len)
        np.random.shuffle(indices)
        for idx in range(0,dataset_len):
            start_id = idx
            end_id = start_id + FLAGS.batch_size
            batch_idx = indices[start_id:end_id]
            feed_dict = construct_feed_dict(train_graph_data["words"][batch_idx],train_graph_data["chars"][batch_idx],
                                            train_graph_data["words_mask"][batch_idx],train_graph_data["chars_mask"][batch_idx],
                                            train_graph_data["x_adj"][batch_idx],train_graph_data["labels"][batch_idx],placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            train_loss += outs[1]*len(idx)
            train_acc += outs[2]*len(idx)
        train_loss /= len(dataset_len)
        train_acc /= len(dataset_len)

        # validation
        # Validation
        val_cost, val_acc, val_duration, _, _, _ = evaluate(sess,model,val_feature, val_adj, val_mask, val_y, placeholders)
        cost_val.append(val_cost)
        
        # Test
        test_cost, test_acc, test_duration, embeddings, pred, labels = evaluate(sess,model,test_feature, test_adj, test_mask, test_y, placeholders)
        
        if val_acc >= best_val:
            best_val = val_acc
            best_epoch = epoch
            best_acc = test_acc
            best_cost = test_cost
            preds = pred
        # Print results
        end_time = time.time()
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
            "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_cost),
            "val_acc=", "{:.5f}".format(val_acc), "test_acc=", "{:.5f}".format(test_acc), 
            "time=", "{:.5f}".format(end_time- start_time))
    print("Optimization Finished!")

    # Best results
    print('Best epoch:', best_epoch)
    print("Test set results:", "cost=", "{:.5f}".format(best_cost),"accuracy=", "{:.5f}".format(best_acc))

    print("Test Precision, Recall and F1-Score...")
    print(metrics.classification_report(labels, preds, digits=4))
    print("Macro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(labels, preds, average='macro'))
    print("Micro average Test Precision, Recall and F1-Score...")
    print(metrics.precision_recall_fscore_support(labels, preds, average='micro'))
if __name__ == "__main__":
    main()
