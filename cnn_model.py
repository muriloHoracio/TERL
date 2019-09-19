import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import learn
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.disable_v2_behavior()

def calculate_flatten_shape(architecture, widths, feature_maps, max_len):
	last_shape = max_len - widths[0] + 1
	last_feature_map = 0
	conv_layer = 0
	for i in range(len(architecture)):
		if architecture[i] == 'pool':
			last_shape = int(np.ceil(last_shape/widths[i]))
		if architecture[i] == 'conv':
			last_feature_map = feature_maps[conv_layer]
			conv_layer += 1
	return last_shape * last_feature_map

def print_tensors(W, B, layers):
	print('*' * 79 + '\n**' + ' ' * 32 + ' RUN  INFO ' + ' ' * 32 + '**\n' + '*' * 79)
	print('W:')
	for w in W:
		print('\t'+w+'\t'+str(W[w]))
	print('B:')
	for b in B:
		print('\t'+b+'\t'+str(B[b]))

	print('*' * 79 + '\n**' + ' ' * 33 + ' TENSORS ' + ' ' * 33 + '**\n' + '*' * 79)
	for layer in layers:
		print(str(layer))

def create_layers(architecture, functions, widths, strides, feature_maps,	vocab_len, max_len, x_input, W, B, flatten_shape):
	layers = [x_input]
	conv_layer_counter = 0
	pool_layer_counter = 0
	fc_layer_counter = 0
	for layer in range(len(architecture)):
		if architecture[layer] == 'conv':
			layers.append(
				tf.nn.conv2d(
					layers[-1],
					W['conv'+str(layer)],
					strides = [1, 1, strides[layer], 1],
					padding = 'VALID' if layer == 0 else 'SAME'
				)
			)
			layers.append(
				tf.nn.relu(tf.nn.bias_add(layers[-1], B['conv'+str(layer)]))
			)
			conv_layer_counter += 1
		elif architecture[layer] == 'pool':
			layers.append(
				tf.nn.avg_pool(
					layers[-1],
					ksize = [1, widths[layer], 1, 1],
					strides = [1, 1, strides[layer], 1],
					padding="SAME"
				)
			)
			pool_layer_counter += 1
		elif architecture[layer] == 'fc':
			if fc_layer_counter == 0:
				layers.append(
					tf.reshape(
						layers[-1],
						[-1, flatten_shape]
					)
				)
			layers.append(
				tf.nn.relu(
					tf.matmul(
						layers[-1],
						W['fc'+str(layer)] + B['fc'+str(layer)],
						name='relu_fc'+str(layer)
					)
				)
			)
			fc_layer_counter += 1
		elif architecture[layer] == 'pred':
			layers.append(
				tf.matmul(layers[-1], W['pred']) + B['pred']
			)
	return layers

def create_learnable_parameters_tensors(architecture, widths, feature_maps, vocab_len, flatten_shape):
	W = dict()
	B = dict()
	conv_layer_counter = 0
	fc_layer_counter = 0
	for layer in range(len(architecture)):
		if architecture[layer] == 'conv':
			W['conv'+str(layer)] = tf.Variable(
				tf.truncated_normal(
					[
						vocab_len if conv_layer_counter == 0 else 1,
						widths[layer],
						1 if conv_layer_counter == 0 else feature_maps[conv_layer_counter-1],
						feature_maps[conv_layer_counter]
					],
					stddev=0.1,
					dtype=tf.float32
				),
				name='W_conv'+str(layer)
			)
			B['conv'+str(layer)] = tf.Variable(
				tf.truncated_normal(
					[
						feature_maps[conv_layer_counter]
					],
					stddev=0.1,
					dtype=tf.float32
				),
				name='b_conv'+str(layer)
			)
			conv_layer_counter += 1
		elif architecture[layer] == 'fc':
			W['fc'+str(layer)] = tf.Variable(
				tf.truncated_normal(
					[
						flatten_shape if fc_layer_counter == 0 else widths[layer - 1],
						widths[layer]
					],
					stddev=0.1
				),
				name='W_fc'+str(layer)
			)
			B['fc'+str(layer)] = tf.Variable(
				tf.truncated_normal(
					[
						widths[layer]
					],
					stddev=0.1
				),
				name='b_fc'+str(layer)
			)
			fc_layer_counter += 1
		elif architecture[layer] == 'pred':
			W['pred'] = tf.Variable(
				tf.truncated_normal(
					[
						widths[layer-1],
						widths[layer]
					],
					stddev=0.1
				),
				name='W_pred'
			)
			B['pred'] = tf.Variable(
				tf.truncated_normal(
					[
						widths[layer]
					],
					stddev=0.1
				),
				name='b_pred'
			)
	return W, B

class CNN_model:
	def __init__(self, num_classes, architecture, functions, widths, strides, feature_maps,	vocab_len, max_len,	l2_reg_lambda=0.0):
		flatten_shape = calculate_flatten_shape(architecture, widths, feature_maps, max_len)
		architecture.append('pred')
		widths.append(num_classes)

		#the x_input will have a size of max_len
		self.x_input = tf.placeholder(tf.float32, [None, vocab_len, max_len, 1], name="x_input")
		self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
		self.dropout_param = tf.placeholder(tf.float32, name="dropout_param")

		#initializes the bias and weigth tensors
		W, B = create_learnable_parameters_tensors(architecture, widths, feature_maps, vocab_len, flatten_shape)

		#create the cnn's layers
		self.layers = create_layers(architecture, functions, widths, strides, feature_maps,	vocab_len, max_len, self.x_input, W, B, flatten_shape)

		#prints tensors
		print_tensors(W, B, self.layers)

		self.predictions = tf.argmax(self.layers[-1], 1, name="predictions")
		self.labels = tf.argmax(self.y_input, 1)

		#losses
		losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.layers[-1], labels=self.y_input)
		loss_sum = 0
		for layer in range(len(architecture)):
			if architecture[layer] == 'conv':
				loss_sum += tf.nn.l2_loss(W['conv'+str(layer)])
				loss_sum += tf.nn.l2_loss(B['conv'+str(layer)])
			elif architecture[layer] == 'fc':
				loss_sum += tf.nn.l2_loss(W['fc'+str(layer)])
				loss_sum += tf.nn.l2_loss(B['fc'+str(layer)])
		self.loss = tf.reduce_mean(losses) + l2_reg_lambda * loss_sum

		#predictions
		self.correct_predictions = tf.equal(self.predictions, self.labels)
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_predictions, "float"), name="accuracy") #calulates the accuracy by summing all elements of correct_predictions and dividing it by the length of it
