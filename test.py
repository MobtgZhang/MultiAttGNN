import numpy as np
import tensorflow as tf
def accuracy(preds,labels):
    correct_prediction = tf.equal(labels, tf.argmax(preds, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)
def f1_score(preds,labels,average='binary'):
    """
    (batch_size,n_class,label_size) = preds.shape
    #labels = tf.repeat(tf.expand_dims(labels,1),n_class,axis=1)
    #correct_prediction = tf.equal(labels, tf.argmax(preds, 1))
    labels = tf.repeat(tf.expand_dims(labels,1),n_class,axis=1)
    expand_labels = tf.expand_dims(tf.expand_dims(tf.arange(n_class),axis=0),axis=2)
    expand_labels = tf.repeat(expand_labels,label_size,axis=2)
    expand_labels = tf.repeat(expand_labels,batch_size,axis=0)
    result = tf.cast(tf.equal(preds,expand_labels),dtype=tf.float32)
    result = tf.reduce_sum(result,axis=2)
    """
    (batch_size,n_class,label_size) = preds.shape
    preds = tf.argmax(preds, 1)
    for idx in range(n_class):
        val1 = tf.equal(labels,idx)
        val2 = tf.equal(preds,idx)
        val3 = tf.equal(val1,val2)
def make_preds(batch_size,label_size,n_class):
    preds = np.random.randint(0,n_class,size=(batch_size,label_size))
    preds = preds[:,np.newaxis,:].repeat(n_class,axis=1)
    min_preds = np.arange(n_class)[np.newaxis,:,np.newaxis]
    min_preds = min_preds.repeat(label_size,axis=2).repeat(batch_size,axis=0)
    preds = np.array(preds==min_preds,dtype=np.int64)
    return preds
def equal_preds(labels,n_class):
    batch_size,label_size = labels.shape
    preds = labels[:,np.newaxis,:].repeat(n_class,axis=1)
    min_preds = np.arange(n_class)[np.newaxis,:,np.newaxis]
    min_preds = min_preds.repeat(label_size,axis=2).repeat(batch_size,axis=0)
    preds = np.array(preds==min_preds,dtype=np.int64)
    return preds
def softmax(preds,labels):
    preds = tf.nn.softmax(preds,axis=1)
    labels = tf.expand_dims(labels,1)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds,labels=labels,dim=1)
    return tf.reduce_mean(loss)
def main():
    batch_size = 20
    label_size = 14
    n_class = 4
    labels = np.random.randint(0,n_class,size=(batch_size,label_size))
    preds = np.random.rand(batch_size,n_class,label_size)
    #preds = make_preds(batch_size,label_size,n_class)
    #acc = accuracy(preds,labels)
    #print(preds.shape)
    #value = f1_score(preds,labels)
    value = softmax(preds,labels)
    with tf.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        out_acc = sess.run(value)
        print(out_acc)
if __name__ == "__main__":
    main()

