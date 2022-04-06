import tensorflow as tf


def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    return tf.reduce_mean(loss)

def accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = tf.equal(labels,tf.argmax(preds,1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)



def pr_re_f1_multilabel(cm, pos_indices):
    num_classes = cm.shape[0]
    neg_indices = [i for i in range(num_classes) if i not in pos_indices]
    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, neg_indices] = 0 # 将负样本预测正确的位置清零零
    diag_sum = tf.reduce_sum(tf.diag_part(cm * cm_mask)) # 正样本预测正确的数量

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[:, neg_indices] = 0 # 将负样本对应的列清零
    tot_pred = tf.reduce_sum(cm * cm_mask) # 所有被预测为正的样本数量

    cm_mask = np.ones([num_classes, num_classes])
    cm_mask[neg_indices, :] = 0 # 将负样本对应的行清零
    tot_gold = tf.reduce_sum(cm * cm_mask) # 所有正样本的数量

    pr = safe_div(diag_sum, tot_pred)
    re = safe_div(diag_sum, tot_gold)
    f1 = safe_div(2. * pr * re, pr + re)
    
    return pr, re, f1

def f1_score_multi(preds,labels):
    pass


