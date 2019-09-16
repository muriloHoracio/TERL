#! /usr/bin/env python
import sys
import numpy as np
import tensorflow as tf
from get_cnn import seqCNN    
import data_handler as dh


def classify(num_classes, num_filters, pool_size, vocab_size, region_size, max_len, x_test, y_test, test_len, batch_size):
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('model/model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('model'))
            cnn = seqCNN(
                graph,
                num_classes,
                num_filters,
                pool_size,
                vocab_size,
                region_size,
                max_len)

            pre_x = tf.placeholder(tf.uint8,[None, max_len])
            pre_y = tf.placeholder(tf.uint8,[None])
            one_hot_x = tf.one_hot(pre_x,vocab_size)
            one_hot_y = tf.one_hot(pre_y,num_classes)


            accuracies = []
            def eval_step(x_batch, y_batch):
                feed_dict = {
                    cnn.x_input: x_batch,
                    cnn.y_input: y_batch,
                    cnn.dropout_param: 0.5
                }
                predictions = sess.run(cnn.predictions,feed_dict)
                return predictions

            def evaluate(test_len, batch_size, x_test, y_test):
                predictions = np.array([],dtype=np.uint8)
                for i in range(0, val_len, batch_size):
                    x_batch = x_test[i:i + batch_size]
                    y_batch = y_test[i:i + batch_size]
                    pre_xo = sess.run(one_hot_x,feed_dict={pre_x:x_batch})
                    x_batch = pre_xo.reshape(x_batch.shape[0],-1,1,1)
                    y_batch = sess.run(one_hot_y,feed_dict={pre_y:y_batch})
                    predictions = np.concatenate([predictions, eval_step(x_batch,y_batch)])
                m = metrics.Metric(y_test,predictions)
                accuracies.append([epoch, m.accuracy_M, m.accuracy_m, m.accuracy])
                return predictions

            #TEST
            predictions = evaluate(test_len, batch_size, x_test, y_test)
    return y_test, np.array(predictions,dtype=np.uint8), accuracies


test_ds = [
            "LTR_test.fa"
            ,"LINE_test.fa"
            ,"SINE_test.fa"
            ,"DNA_test.fa"
            ]


db = dh.DataHandler([], test_ds)

num_filters = [64, 32]
region_size = 60
pool_size = 60
train_batch_size = 64
val_batch_size = 64
batch_size = 64
num_epochs = 30
dropout_param = 0.5

if (db.max_len - region_size + 1) % pool_size != 0:
    while (db.max_len - region_size + 1) % pool_size != 0: pool_size += 1
    print("Pooling kernel should be a factor of the max sequence length\nPooling kernel changed to: ",str(pool_size))

labels, predictions, accuracies = classify(db.num_classes, num_filters, pool_size, db.vocab_size, region_size, db.max_len, db.x_test, db.y_test, bd.test_size, batch_size)
 
m = metrics.Metric(labels, predictions, class_names=db.classes)
m.print_report()
m.save_confusion_matrix(output_file='out/60_epochs.png')
m.save_learning_curve(accuracies,acc=0,output_file='out/60_epochs.png')
