import sys
from train_parser import get_options
from train_parser import print_options

options = get_options(sys.argv[1:])
report_out = print_options(options)

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import tensorflow.compat.v1 as tf
from cnn_model import CNN_model
import data_handler as dh
import metrics
import time
import datetime

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

def get_files(root):
	train_files = [root+'/Train/'+train_file for train_file in os.listdir(root+'/Train')]
	test_files = [root+'/Test/'+test_file for test_file in os.listdir(root+'/Test')]
	return train_files, test_files

def get_optimizer(optimizer_option, learning_rate):
	if optimizer_option == 'ADAM':
		return tf.train.AdamOptimizer(learning_rate=learning_rate)
	elif optimizer_option == 'ADADELTA':
		return tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
	elif optimizer_option == 'ADAGRAD':
		return tf.train.AdagradOptimizer(learning_rate=learning_rate)
	elif optimizer_option == 'FTRL':
		return tf.train.FtrlOptimizer(learning_rate=learning_rate)
	elif optimizer_option == 'RMSPROP':
		return tf.train.RMSPropOptimizer(learning_rate=learning_rate)
	elif optimizer_option == 'SGD':
		return tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

def print_files(root):
	out = '*' * 79 + '\n**' + ' ' * 34 + ' FILES ' + ' ' * 34 + '**\n' + '*' * 79 + '\n'
	out += 'Train'+''.join('\n\t%s' % t for t in os.listdir(options.root+'/Train')) + '\n'
	out += 'Test'+''.join('\n\t%s' % t for t in os.listdir(options.root+'/Test')) + '\n'
	if options.verbose: print(out)
	return out

def print_data_info(db):
	out = '*' * 79 + '\n**' + ' ' * 27 + ' CLASSIFICATION INFO ' + ' ' * 27 + '**\n' + '*' * 79 + '\n'
	out += "%20s %s\n" % ('Classes:',', '.join(db.classes))
	out += "%20s %d\n" % ("Train size:", db.train_size)
	out += "%20s %d\n" % ("Test size:", db.test_size)
	out += "%20s %d\n" % ("Longest sequence:", db.max_len)
	out += "%20s %d\n" % ("Vocabulary size:", db.vocab_size)
	if options.verbose: print(out)
	return out

def train_evaluate(x_train, y_train, x_test, y_test, vocab_size, max_len, classes, num_classes, architecture, functions, widths, strides, feature_maps, optimizer, train_batch_size, test_batch_size, epochs, dropout=0.5, output_file='Models/'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')):
	out = ''
	train_length = len(y_train)
	test_length = len(y_test)
	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn = CNN_model(
				num_classes,
				classes,
				architecture,
				functions,
				widths,
				strides,
				feature_maps,
				vocab_size,
				max_len
			)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			pre_x = tf.placeholder(tf.uint8,[None, max_len], name='pre_x')
			pre_y = tf.placeholder(tf.uint8,[None], name='pre_y')
			one_hot_x = tf.one_hot(pre_x, vocab_size, dtype=tf.float32, name='one_hot_x')
			one_hot_y = tf.one_hot(pre_y, num_classes, dtype=tf.float32, name='one_hot_y')

			accuracies = []
			training_time = 0
			test_times = []
			best_result = [0, []]
			sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

			def train_step(x_batch, y_batch):
				feed_dict = {
					cnn.x_input: x_batch,
					cnn.y_input: y_batch,
					cnn.dropout: dropout
				}
				_, step = sess.run([train_op, global_step], feed_dict)

			def eval_step(x_batch, y_batch):
				feed_dict = {
					cnn.x_input: x_batch,
					cnn.y_input: y_batch,
					cnn.dropout: dropout
				}
				predictions = sess.run(cnn.layers['pred'], feed_dict)
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
			for epoch in range(epochs):
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
				out += time_str+': '+str(accuracies[-1]) + '\n'
				if options.verbose: print(time_str+': '+str(accuracies[-1]))
			training_time = time.time() - training_time
			if options.save_model:
				tf.saved_model.simple_save(
					sess,
					options.model_export_dir,
					inputs={'x_input': cnn.x_input},
					outputs={'prediction': cnn.layers['pred']}
				)
			#TEST
			test_times.append(time.time())
			predictions = evaluate(epoch, test_length, test_batch_size, x_test, y_test)
			test_times[-1] = time.time() - test_times[-1]
	return y_test, np.array(predictions, dtype=np.uint8), accuracies, best_result, training_time, test_times, out

train_files, test_files = get_files(options.root)
report_out += print_files(options.root)

db = dh.DataHandler(train_files, test_files)
report_out += print_data_info(db)

shuffled = np.random.permutation(range(db.train_size))
db.x_train = db.x_train[shuffled]
db.y_train = db.y_train[shuffled]

shuffled = np.random.permutation(range(db.test_size))
db.x_test = db.x_test[shuffled]
db.y_test = db.y_test[shuffled]

labels, predictions, accuracies, best_result, training_time, test_times, training_out = train_evaluate(
	db.x_train,
	db.y_train,
	db.x_test,
	db.y_test,
	db.vocab_size,
	db.max_len,
	db.classes,
	db.num_classes,
	options.architecture,
	options.functions,
	options.widths,
	options.strides,
	options.feature_maps,
	get_optimizer(options.optimizer, options.learning_rate),
	options.train_batch_size,
	options.test_batch_size,
	options.epochs,
	options.dropout
)

report_out += training_out

m = metrics.Metric(labels, best_result[1], classes=db.classes, filename_prefix=options.prefix)
classification_report = m.get_report()
if options.verbose: print(classification_report)
report_out += classification_report

if options.save_graphs:
	m.save_confusion_matrix(title=options.cm_title)

if options.save_graphs:
	m.save_learning_curve(accuracies, title=options.lc_title, acc_type=0)

if options.verbose: print('\n\nTraining time: '+str(training_time)+'\nAverage test time: '+str(sum(test_times)/len(test_times)))

if options.save_report:
	report_out += '\n\nTraining time: '+str(training_time)+'\nAverage test time: '+str(sum(test_times)/len(test_times))
	with open('Outputs/Report_'+options.prefix+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w+') as f:
		f.write(report_out)

exit(best_result[0])
