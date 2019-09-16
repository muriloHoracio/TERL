import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'   
class CNN_model(object):
	def __init__(self,num_classes, conv_filters, pool_filters, fc_filters, vocab_len, max_len, l2_reg_lambda=0.0):
		last_conv_out_shape = max_len - conv_filters[0][0] + 1
		for i in pool_filters:
			last_conv_out_shape = int(np.floor(last_conv_out_shape/i))
		flatten_shape = last_conv_out_shape*conv_filters[-1][1]

		#the x_input will have a size of max_len
		self.x_input = tf.placeholder(tf.float32, [None, max_len, vocab_len, 1], name="x_input")
		self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
		self.dropout_param = tf.placeholder(tf.float32, name="dropout_param")

		#initializes the first layer with conv_filters[0][1] filters, each of size conv_filters[0][0], the initial random weights and biases.
		W_conv1 = tf.Variable(tf.truncated_normal([
				conv_filters[0][0],
				vocab_len,
				1,
				conv_filters[0][1]
			], stddev=0.1, dtype=tf.float32), name="W_conv1")
		b_conv1 = tf.Variable(tf.truncated_normal([conv_filters[0][1]],stddev=0.1, dtype=tf.float32),name="b_conv1")

		#initializes the second layer with conv_filters[1][1] filters, each of size conv_filters[1][0], the initial random weights and biases.
		W_conv2 = tf.Variable(tf.truncated_normal([
				conv_filters[1][0], 
				1, 
				conv_filters[0][1], 
				conv_filters[1][1]
			], stddev=0.1, dtype=tf.float32), name="W_conv2")
		b_conv2 = tf.Variable(tf.truncated_normal([conv_filters[1][1]],stddev=0.1, dtype=tf.float32),name="b_conv2")

		#initializes the third layer with conv_filters[2][1] filters, each of size conv_filters[2][0], the initial random weights and biases.
		W_conv3 = tf.Variable(tf.truncated_normal([
				conv_filters[2][0],
				1,
				conv_filters[1][1],
				conv_filters[2][1]
			], stddev=0.1, dtype=tf.float32), name='W_conv3')
		b_conv3 = tf.Variable(tf.truncated_normal([conv_filters[2][1]],stddev=0.1,dtype=tf.float32),name='b_conv3')

		#conv-relu-pool layer1
		self.conv1 = tf.nn.conv2d(
			self.x_input,
			W_conv1,
			strides = [1, 1, 1, 1],
			padding="VALID"
		)
		self.relu1 = tf.nn.relu(tf.nn.bias_add(self.conv1, b_conv1))
		#conv1_dropout = tf.nn.dropout(relu1, rate=self.dropout_param)
		self.pool1 = tf.nn.avg_pool(
			self.relu1,
			ksize = [1,pool_filters[0],1,1],
			strides = [1,pool_filters[0],1,1],
			padding="VALID"
		)

		#conv-relu-pool layer2
		self.conv2 = tf.nn.conv2d(
			self.pool1,
			W_conv2,
			strides=[1, 1, 1, 1],
			padding="VALID"
		)
		self.relu2 = tf.nn.relu(tf.nn.bias_add(self.conv2, b_conv2))
		#conv2_dropout = tf.nn.dropout(relu2, rate=self.dropout_param)
		self.pool2 = tf.nn.avg_pool(
			self.relu2,
			ksize = [1,pool_filters[1],1,1],
			strides = [1,pool_filters[1],1,1],
			padding="VALID"
		)

		#conv-relu-pool layer3
		self.conv3 = tf.nn.conv2d(
			self.pool2,
			W_conv3,
			strides=[1, 1, 1, 1],
			padding="SAME"
		)
		self.relu3 = tf.nn.relu(tf.nn.bias_add(self.conv3, b_conv3))
		#conv2_dropout = tf.nn.dropout(relu2, rate=self.dropout_param)
		self.pool3 = tf.nn.avg_pool(
			self.relu3,
			ksize = [1,pool_filters[2],1,1],
			strides = [1,pool_filters[2],1,1],
			padding="VALID"
		)

		#pool1_dropout = tf.nn.dropout(pool1, rate=self.dropout_param)

		#flatten
		self.pool_flat = tf.reshape(self.pool3, [-1, flatten_shape])

		#fc1
		W_fc1 = tf.Variable(tf.truncated_normal([flatten_shape, fc_filters[0]], stddev=0.1),name="W_fc1")
		b_fc1 = tf.Variable(tf.truncated_normal([fc_filters[0]], stddev=0.1),name="b_fc1")
		self.fc1 = tf.nn.relu(tf.matmul(self.pool_flat,W_fc1) + b_fc1, name="relu_fc1")

		#fc2
		W_fc2 = tf.Variable(tf.truncated_normal([fc_filters[0],fc_filters[1]], stddev=0.1),name="W_fc2")
		b_fc2 = tf.Variable(tf.truncated_normal([fc_filters[1]], stddev=0.1),name="b_fc2")
		self.fc2 = tf.nn.relu(tf.matmul(self.fc1,W_fc2) + b_fc2, name="relu_fc2")

		W_fc3 = tf.Variable(tf.truncated_normal([fc_filters[1],num_classes], stddev=0.1),name="W_fc3")
		b_fc3 = tf.Variable(tf.truncated_normal([num_classes], stddev=0.1),name="b_fc3")
		self.scores = tf.matmul(self.fc2,W_fc3)+b_fc3

		self.predictions = tf.argmax(self.scores, 1, name="predictions")
		self.labels = tf.argmax(self.y_input, 1)

		#losses
		losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.y_input)
		self.loss = tf.reduce_mean(losses) + l2_reg_lambda * (
			tf.nn.l2_loss(W_conv1) + 
			tf.nn.l2_loss(b_conv1) + 
			tf.nn.l2_loss(W_conv2) +
			tf.nn.l2_loss(b_conv2) +
			tf.nn.l2_loss(W_conv3) +
			tf.nn.l2_loss(b_conv3) +
			tf.nn.l2_loss(W_fc1) +
			tf.nn.l2_loss(b_fc1) +
			tf.nn.l2_loss(W_fc2) +
			tf.nn.l2_loss(b_fc2) +
			tf.nn.l2_loss(W_fc3) +
			tf.nn.l2_loss(b_fc3)
			)

		#predictions
		self.correct_predictions = tf.equal(self.predictions, self.labels)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy") #calulates the accuracy by summing all elements of correct_predictions and dividing it by the length of it
	def print_layers_shape(self):
		out = 'conv1 shape: '+str(self.conv1.shape)+'\n'
		out += 'relu1 shape: '+str(self.relu1.shape)+'\n'
		out += 'pool1 shape: '+str(self.pool1.shape)+'\n'
		out += 'conv2 shape: '+str(self.conv2.shape)+'\n'
		out += 'relu2 shape: '+str(self.relu2.shape)+'\n'
		out += 'pool2 shape: '+str(self.pool2.shape)+'\n'
		out += 'conv3 shape: '+str(self.conv2.shape)+'\n'
		out += 'relu3 shape: '+str(self.relu3.shape)+'\n'
		out += 'pool3 shape: '+str(self.pool3.shape)+'\n'
		out += 'flat shape: '+str(self.pool_flat.shape)+'\n'
		out += 'fc1 shape: '+str(self.fc1.shape)+'\n'
		out += 'fc2 shape: '+str(self.fc2.shape)+'\n'
		out += 'scores shape: '+str(self.scores.shape)+'\n'
		print(out)
