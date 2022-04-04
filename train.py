import os
import time

from sklearn import metrics
import tensorflow as tf

from config import get_tf_args
from src.utils import load_embeddings
from src.dataloader import DataLoader
from src.dataset import RegularDataset,batchfy,construct_feed_dict
from src.dictionary import Dictionary

from src.model import GNNING,CNNModel

def evaluate(sess,model,features, support, mask, labels, placeholders):
    s_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, mask, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels], feed_dict=feed_dict_val)
    e_test = time.time()
    t_sect = e_test - s_test
    return outs_val,t_sect
def get_n_class(dataset_name):
    
    if dataset_name == 'aichallenger':
        n_class = 14
    elif dataset_name == 'goemotions':
        n_class =  5
    elif dataset_name == 'wrime':
        n_class =  10
    elif dataset_name == 'clue2020emotions':
        n_class = 5
    else:
        raise ValueError('Invalid argument for model: ' + str(dataset_name))
    return n_class
def main():
    FLAGS = get_tf_args()
    # load dictionary
    result_data_dir = os.path.join(FLAGS.result_dir,FLAGS.dataset)
    load_chars_dict_file = os.path.join(result_data_dir,"chars_dict.pkl")     
    load_words_dict_file = os.path.join(result_data_dir,"words_dict.pkl")
    words_dict = Dictionary.load(load_words_dict_file)
    chars_dict = Dictionary.load(load_chars_dict_file)
    # load dataset
    load_train_graph_file = os.path.join(result_data_dir,"train_graph.pkl")
    train_dataset = RegularDataset(load_train_graph_file,words_dict,chars_dict)
    train_dataloader = DataLoader(train_dataset,FLAGS.batch_size,shuffle=True,collate_fn=batchfy)
    
    load_validate_graph_file = os.path.join(result_data_dir,"validate_graph.pkl")
    validate_dataset = RegularDataset(load_validate_graph_file,words_dict,chars_dict)
    validate_dataloader = DataLoader(validate_dataset,FLAGS.batch_size,shuffle=True,collate_fn=batchfy)
    
    load_test_graph_file = os.path.join(result_data_dir,"test_graph.pkl")
    test_dataset = RegularDataset(load_test_graph_file,words_dict,chars_dict)
    test_dataloader = DataLoader(test_dataset,FLAGS.batch_size,shuffle=True,collate_fn=batchfy)
    # dataset preparing
    label_length = test_dataset[0]["labels"].shape[0]
    # the defination of the placeholders
    placeholders = {
        'adj': tf.compat.v1.placeholder(tf.float32, shape=(None, None, None)),
        'adj-words': tf.compat.v1.placeholder(tf.int64, shape=(None,None)),
        'chars': tf.compat.v1.placeholder(tf.int64, shape=(None, None)),
        'adj-mask': tf.compat.v1.placeholder(tf.float32, shape=(None, None)),
        'chars-mask': tf.compat.v1.placeholder(tf.float32, shape=(None, None)),
        'labels': tf.compat.v1.placeholder(tf.int64, shape=(None,label_length)),
        'dropout': tf.compat.v1.placeholder_with_default(FLAGS.dropout,shape=()),
        'num_features_nonzero': tf.compat.v1.placeholder(tf.float32), # helper variable for sparse dropout
    }
    print(placeholders)
    n_class = get_n_class(FLAGS.dataset.lower())
    if FLAGS.dataset.lower() in ['aichallenger','goemotions','wrime']:
        if FLAGS.model == 'gnning':
            model = GNN()
        elif FLAGS.model == 'cnn':
            pass
        elif FLAGS.model == 'cnn':
            pass
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    elif FLAGS.dataset.lower() in ['clue2020emotions','weibo4moods','sst-5','student']:
        if FLAGS.model == 'gnning':
            embedding,_ = load_embeddings(result_data_dir)
            model = GNNING(placeholders,FLAGS.hid_dim,n_class,
                            FLAGS.learning_rate,FLAGS.weight_decay,embedding)
        elif FLAGS.model == 'cnn':
            embedding,_ = load_embeddings(result_data_dir)
            filter_size = [int(v) for v in FLAGS.filter_size.split(',')]
            model = CNNModel(n_class,FLAGS.num_filters,placeholders,embedding,filter_size,
                    FLAGS.learning_rate,FLAGS.weight_decay)
        else:
            raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
    else:
        raise ValueError("Unknown dataset:%s"%str(FLAGS.dataset))
    
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
    preds = None
    labels = None

    print('train start...')

    for epoch in range(FLAGS.epochs):
        start_time = time.time()

        for item in train_dataloader:
            feed_dict = construct_feed_dict(item,placeholders)
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            train_loss += outs[1]*FLAGS.batch_size
            train_acc += outs[2]*FLAGS.batch_size
        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)

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
