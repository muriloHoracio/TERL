import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def calculate_flatten_shape(architecture, widths, feature_maps, max_len):
	last_shape = max_len - widths[0] + 1
	for i, layer in enumerate(architecture):
		if layer == 'pool':
			last_shape = int(np.ceil(last_shape/widths[i]))
	return last_shape * feature_maps[-1]

 
class CNN_model(object):
	def __init__(self, num_classes, classes, architecture, functions, widths, strides, feature_maps, vocab_size, max_len, l2_reg_lambda=0.0):
		#initializes weights and biases
		architecture.append('pred')
		functions.append('pred')
		flatten_shape = calculate_flatten_shape(architecture, widths, feature_maps, max_len)

		#instatiate constants
		tf.constant(num_classes, name='num_classes')
		tf.constant(classes, name='classes')
		tf.constant(architecture, name='architecture')
		tf.constant(functions, name='functions')
		tf.constant(widths, name='widths')
		tf.constant(strides, name='strides')
		tf.constant(feature_maps, name='feature_maps')
		tf.constant(vocab_size, name='vocab_size')
		tf.constant(max_len, name='max_len')
		
		self.x_input = tf.placeholder(tf.float32, [None, max_len, vocab_size, 1], name="x_input")
		self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
		self.dropout = tf.placeholder(tf.float32, name="dropout")

		self.W, self.B = self.create_learnable_params(architecture, widths, feature_maps, vocab_size, num_classes, flatten_shape)
		self.layers = self.create_layers(architecture, widths, strides, functions, flatten_shape)

		#losses
		loss_sum = 0
		for b in self.B:
			loss_sum += tf.nn.l2_loss(self.B[b])
		for w in self.W:
			loss_sum += tf.nn.l2_loss(self.W[w])

		losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.layers['scores'], labels=self.y_input)
		self.loss = tf.reduce_mean(losses) + (l2_reg_lambda * loss_sum)

		#predictions
		self.labels = tf.argmax(self.y_input, 1)
		self.correct_predictions = tf.equal(self.layers['pred'], self.labels)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy")
	def create_learnable_params(self, architecture, widths, feature_maps, vocab_size, num_classes, flatten_shape):
		W = dict()
		B = dict()
		conv_counter = 0
		fc_counter = 0
		for i, layer in enumerate(architecture):
			if layer == 'conv':
				height = vocab_size if conv_counter == 0 else 1
				prev_layer_dim = 1 if conv_counter == 0 else feature_maps[conv_counter - 1]
				W['conv' + str(i)] = tf.Variable(tf.truncated_normal([widths[i], height, prev_layer_dim, feature_maps[conv_counter]], stddev=0.1, dtype=tf.float32))
				B['conv' + str(i)] = tf.Variable(tf.truncated_normal([feature_maps[conv_counter]], stddev=0.1, dtype=tf.float32))
				conv_counter += 1
			elif layer == 'fc':
				height = flatten_shape if fc_counter == 0 else widths[i - 1]
				W['fc' + str(i)] = tf.Variable(tf.truncated_normal([height, widths[i]], stddev=0.1))
				B['fc' + str(i)] = tf.Variable(tf.truncated_normal([widths[i]], stddev=0.1))
				fc_counter += 1
			elif layer == 'pred':
				W['pred'] = tf.Variable(tf.truncated_normal([widths[i - 1], num_classes], stddev=0.1))
				B['pred'] = tf.Variable(tf.truncated_normal([num_classes], stddev=0.1))
		return W, B
	def create_layers(self, architectures, widths, strides, functions, flatten_shape):
		layers = dict()
		fc_counter = 0
		prev_layer = self.x_input
		for i, layer in enumerate(architectures):
			if layer == 'conv':
				layers['conv' + str(i)] = tf.nn.conv2d(prev_layer, self.W['conv' + str(i)], strides=[strides[i], 1, 1, 1], padding='SAME' if i != 0 else 'VALID')
				if functions[i] == 'relu':
					layers['relu' + str(i)] = tf.nn.relu(tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
					prev_layer = layers['relu' + str(i)]
				elif functions[i] == 'tanh':
					layers['tanh' + str(i)] = tf.nn.tanh(tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
					prev_layer = layers['tanh' + str(i)]
				elif functions[i] == 'sigmoid':
					layers['sigmoid' + str(i)] = tf.nn.sigmoid(tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
					prev_layer = layers['sigmoid' + str(i)]
				elif functions[i] == 'leaky_relu':
					layers['leaky_relu' + str(i)] = tf.nn.leaky_relu(tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
					prev_layer = layers['leaky_relu' + str(i)]
				elif functions[i] == 'elu':
					layers['elu' + str(i)] = tf.nn.elu(tf.nn.bias_add(layers['conv' + str(i)], self.B['conv' + str(i)]))
					prev_layer = layers['elu' + str(i)]
				prev_layer = tf.nn.dropout(prev_layer,rate=self.dropout)
			elif layer == 'pool':
				if functions[i] == 'avg':
					layers['pool' + str(i)] = tf.nn.avg_pool(prev_layer, ksize=[1, widths[i], 1, 1], strides=[1, strides[i], 1, 1], padding='SAME')
				elif functions[i] == 'max':
					layers['pool' + str(i)] = tf.nn.max_pool(prev_layer, ksize=[1, widths[i], 1, 1], strides=[1, strides[i], 1, 1], padding='SAME')
				prev_layer = layers['pool' + str(i)]
			elif layer == 'fc':
				if fc_counter == 0:
					layers['flatten'] = tf.reshape(prev_layer, [-1, flatten_shape])
					prev_layer = layers['flatten']
				if functions[i] == 'relu':
					layers['relu' + str(i)] = tf.nn.relu(tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
					prev_layer = layers['relu' + str(i)]
				elif functions[i] == 'tanh':
					layers['tanh' + str(i)] = tf.nn.tanh(tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
					prev_layer = layers['tanh' + str(i)]
				elif functions[i] == 'sigmoid':
					layers['sigmoid' + str(i)] = tf.nn.sigmoid(tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
					prev_layer = layers['sigmoid' + str(i)]
				elif functions[i] == 'leaky_relu':
					layers['leaky_relu' + str(i)] = tf.nn.leaky_relu(tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
					prev_layer = layers['leaky_relu' + str(i)]
				elif functions[i] == 'elu':
					layers['elu' + str(i)] = tf.nn.elu(tf.matmul(prev_layer, self.W['fc' + str(i)]) + self.B['fc' + str(i)])
					prev_layer = layers['elu' + str(i)]
				prev_layer = tf.nn.dropout(prev_layer,rate=self.dropout)
				fc_counter += 1
			elif layer == 'pred':
				layers['scores'] = tf.matmul(prev_layer, self.W['pred']) + self.B['pred']
				layers['pred'] = tf.argmax(layers['scores'], 1, name='prediction')
		return layers
	def print_layers_shape(self):
		print('*' * 79 + '\n**' + ' ' * 32 + ' RUN  INFO ' + ' ' * 32 + '**\n' + '*' * 79)
		print('W:')
		for w in self.W:
			print('\t'+w+'\t'+str(self.W[w]))
		print('B:')
		for b in self.B:
			print('\t'+b+'\t'+str(self.B[b]))

		print('*' * 79 + '\n**' + ' ' * 33 + ' TENSORS ' + ' ' * 33 + '**\n' + '*' * 79)
		for layer in self.layers:
			print(str(layer))


"""
activation functions:
	relu
	tanh
	sigmoid
	leaky_relu
	elu
pool function:
	avg
	max
"""
