import sys
import os
import numpy as np
import tensorflow as tf
from cnn_model import CNN_model 
import data_handler as dh
import metrics

import time
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'


def train_evaluate(x_train, y_train, x_test, y_test, vocab_size, max_len, classes, conv_layers, pool_layers, fc_layers, train_batch_size, test_batch_size, num_epochs, dropout_param, output_file='Models/'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')):
    num_classes = len(classes)
    train_length = len(y_train)
    test_length = len(y_test)
    num_it_pe = int(train_length/train_batch_size) #number of iterations per epoch
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = CNN_model( 
                    num_classes,
                    conv_layers,
                    pool_layers,
				    fc_layers,
                    vocab_size,
                    max_len
                )
            cnn.print_layers_shape()

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3) #learning rate
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            pre_x = tf.placeholder(tf.uint8,[None, max_len])
            pre_y = tf.placeholder(tf.uint8,[None])
            one_hot_x = tf.one_hot(pre_x, vocab_size, dtype=tf.float32)
            one_hot_y = tf.one_hot(pre_y, num_classes, dtype=tf.float32)

            accuracies = []
            training_time = 0
            test_times = []
            best_result = [0, []]
            saver = tf.train.Saver(tf.global_variables())
            sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.x_input: x_batch,
                    cnn.y_input: y_batch,
                    cnn.dropout_param: dropout_param
                }
                _, step = sess.run([train_op, global_step], feed_dict)

            def eval_step(x_batch, y_batch):
                feed_dict = {
                    cnn.x_input: x_batch,
                    cnn.y_input: y_batch,
                    cnn.dropout_param: dropout_param
                }
                predictions = sess.run(cnn.predictions, feed_dict)
                return predictions

            def evaluate(epoch, test_len, batch_size, x_test, y_test):
                predictions = np.array([], dtype=np.uint8)
                for i in range(0, test_len, batch_size):
                    x_batch = x_test[i : i + batch_size]
                    y_batch = y_test[i : i + batch_size]
                    pre_xo = sess.run(one_hot_x, feed_dict={pre_x:x_batch})
                    x_batch = pre_xo.reshape(x_batch.shape[0], max_len, vocab_size, 1)
                    y_batch = sess.run(one_hot_y, feed_dict={pre_y:y_batch})
                    predictions = np.concatenate([predictions, eval_step(x_batch, y_batch)])
                m = metrics.Metric(y_test, predictions, classes=classes)
                accuracies.append([epoch, m.accuracy_M, m.accuracy_m, m.accuracy])
                return predictions

            #TRAIN
            training_time = time.time()
            for epoch in range(num_epochs):
                for batch in range(0, train_length, train_batch_size):
                    x_batch = x_train[batch : batch + train_batch_size]
                    y_batch = y_train[batch : batch + train_batch_size]
                    pre_xo = sess.run(one_hot_x, feed_dict={pre_x:x_batch})
                    x_batch = pre_xo.reshape(x_batch.shape[0], max_len, vocab_size, 1)
                    y_batch = sess.run(one_hot_y, feed_dict={pre_y:y_batch})
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                shuffle_indices = np.random.permutation(range(train_length))
                x_train = x_train[shuffle_indices]
                y_train = y_train[shuffle_indices]
                test_times.append(time.time())
                predictions = evaluate(epoch, test_length, test_batch_size, x_test, y_test)
                test_times[-1] = time.time() - test_times[-1]
                if accuracies[-1][1] > best_result[0]: best_result = [accuracies[-1][1], np.copy(predictions)]
                time_str = datetime.datetime.now().isoformat()
                print(time_str+': '+str(accuracies[-1]))
            training_time = time.time() - training_time
            saver.save(sess, output_file)

            #TEST
            test_times.append(time.time())
            predictions = evaluate(epoch, test_length, test_batch_size, x_test, y_test)
            test_times[-1] = time.time() - test_times[-1]
    return y_test, np.array(predictions, dtype=np.uint8), accuracies, best_result, training_time, test_times

"""
train_ds = [
		'Datasets/Round4/DS5/Train/Copia.fa',
		'Datasets/Round4/DS5/Train/Gypsy.fa',
		'Datasets/Round4/DS5/Train/Bel-Pao.fa',
		'Datasets/Round4/DS5/Train/ERV.fa',
		'Datasets/Round4/DS5/Train/L1.fa',
#		'Datasets/Round4/DS5/Train/SINE.fa',
		'Datasets/Round4/DS5/Train/Mariner.fa',
		'Datasets/Round4/DS5/Train/hAT.fa',
		'Datasets/Round4/DS5/Train/Mutator.fa',
		'Datasets/Round4/DS5/Train/PIF.fa',
#		'Datasets/Round4/DS5/Train/CACTA.fa',
		'Datasets/Round4/DS5/Train/Random.fa'
	]
test_ds = [
		'Datasets/Round4/DS5/Test/Copia.fa',
		'Datasets/Round4/DS5/Test/Gypsy.fa',
		'Datasets/Round4/DS5/Test/Bel-Pao.fa',
		'Datasets/Round4/DS5/Test/ERV.fa',
		'Datasets/Round4/DS5/Test/L1.fa',
#		'Datasets/Round4/DS5/Test/SINE.fa',
		'Datasets/Round4/DS5/Test/Mariner.fa',
		'Datasets/Round4/DS5/Test/hAT.fa',
		'Datasets/Round4/DS5/Test/Mutator.fa',
		'Datasets/Round4/DS5/Test/PIF.fa',
#		'Datasets/Round4/DS5/Test/CACTA.fa',
		'Datasets/Round4/DS5/Test/Random.fa'
	]
"""
train_ds = [
		'Datasets/Round4/DS4/Train/LTR.fa',
		'Datasets/Round4/DS4/Train/LINE.fa',
		'Datasets/Round4/DS4/Train/SINE.fa',
		'Datasets/Round4/DS4/Train/TIR.fa',
		'Datasets/Round4/DS4/Train/Random.fa'
	]
test_ds = [
		'Datasets/Round4/DS4/Test/LTR.fa',
		'Datasets/Round4/DS4/Test/LINE.fa',
		'Datasets/Round4/DS4/Test/SINE.fa',
		'Datasets/Round4/DS4/Test/TIR.fa',
		'Datasets/Round4/DS4/Test/Random.fa'
	]
"""
train_ds = [
		'Datasets/lncRNA/train/lncRNA.fa',
		'Datasets/lncRNA/train/mRNA.fa'
	]
test_ds = [
		'Datasets/lncRNA/test/lncRNA.fa',
		'Datasets/lncRNA/test/mRNA.fa'
	]
"""
conv_filters = [[30, 64],[30, 32],[30, 16]]
pool_filters = [20, 20, 10]
fc_filters = [1500, 500]
train_batch_size = 32
test_batch_size = 32
num_epochs = 50
dropout_param = 0.5
filename_prefix = 'Round4_DS4'

db = dh.DataHandler(train_ds, test_ds, region_size=conv_filters[0][0], pool_size=pool_filters[0])
classes = [name.split('.fa')[0] for name in db.classes]

print("\n\n++++++++++++++++++++\n%s\n++++++++++++++++++++" % "RUN INFO")
print("%30s  %s" % ("Classes",str(classes)))
print("%30s  %7d" % ("Train size", db.train_size))
print("%30s  %7d" % ("Test size", db.test_size))
print("%30s  %7d" % ("Longest sequence", db.max_len))
print("%30s  %7d" % ("Vocabulary size", db.vocab_size))
print("%30s  %10s" % ("Convolutional filters", str(conv_filters)))
print("%30s  %10s" % ("Pooling filters", str(pool_filters)))
print("%30s  %10s" % ("FC filters", str(fc_filters)))
print("%30s  %7d" % ("Train batch size",train_batch_size))
print("%30s  %7d" % ("Validation batch size",test_batch_size))
print("%30s  %7d" % ("Number of epochs",num_epochs))
print("%30s  %7f" % ("Dropout rate",dropout_param))

#np.random.seed(10)
#tf.compat.v1.random.set_random_seed(10)
shuffled = np.random.permutation(range(db.train_size))
db.x_train = db.x_train[shuffled]
db.y_train = db.y_train[shuffled]

shuffled = np.random.permutation(range(db.test_size))
db.x_test = db.x_test[shuffled]
db.y_test = db.y_test[shuffled]

labels, predictions, accuracies, best_result, training_time, test_times = train_evaluate(db.x_train, db.y_train, db.x_test, db.y_test, db.vocab_size, db.max_len, classes, conv_filters, pool_filters, fc_filters, train_batch_size, test_batch_size, num_epochs, dropout_param)

m = metrics.Metric(labels, best_result[1], classes=classes, filename_prefix=filename_prefix)
m.print_report()
m.save_report()

m.save_confusion_matrix(title='CNN classifying DS4')
m.save_learning_curve(accuracies, acc=0)

print('\n\nTraining time: '+str(training_time)+'\nAverage test time: '+str(sum(test_times)/len(test_times)))
