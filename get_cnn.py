import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

class seqCNN(object):
    def __init__(self,graph, num_classes, num_filters, pool_size, vocabulary_length, region_size, max_sentence_length, filter_lengths=3, l2_reg_lambda=0.0):
        filter_length = vocabulary_length*region_size
        sentence_length = max_sentence_length*vocabulary_length

        self.x_input = tf.placeholder(tf.float32, [None, sentence_length, 1, 1], name="x_input")
        self.y_input = tf.placeholder(tf.float32, [None, num_classes], name="y_input")
        self.dropout_param = tf.placeholder(tf.float32, name="dropout_param")

        cnn_filter_shape = [filter_length, 1, 1, num_filters[0]]
        W_CN = graph.get_tensor_by_name('W_CN:0')
        b_CN = graph.get_tensor_by_name('b_CN:0')

        cnn_filter2_shape = [filter_lengths, 1, num_filters[0], num_filters[1]]
        W_CN2 = graph.get_tensor_by_name('W_CN2:0')
        b_CN2 = graph.get_tensor_by_name('b_CN2:0')

        #conv-relu-pool layer
        conv1 = tf.nn.conv2d(
                        self.x_input,
                        W_CN,
                        strides=[1, vocabulary_length, 1, 1],
                        padding="VALID",
                        name="conv1"
                        )

        relu1 = tf.nn.relu(
                        tf.nn.bias_add(conv1, b_CN),
                        name="relu1"
                        )

        conv2 = tf.nn.conv2d(
                        relu1,
                        W_CN2,
                        strides=[1, 1, 1, 1],
                        padding="SAME",
                        name="conv2"
                        )
        relu2 = tf.nn.relu(
                        tf.nn.bias_add(conv2, b_CN2),
                        name="relu2"
                        )

        pool_stride = [1,pool_size,1,1]
        pool1 = tf.nn.avg_pool(
                        relu2,
                        ksize = pool_stride,
                        strides = pool_stride,
                        padding="VALID",
                        name="pool1"
                        )

        #dropout
        drop1 = tf.nn.dropout(
                        pool1,
                        self.dropout_param,
                        name="drop1"
                        )

        #response normalization
        normalized = tf.nn.local_response_normalization(drop1)

        #feature extraction and flatting for future 
        self.pool_flat = tf.reshape(normalized, [-1, int(np.ceil((max_sentence_length - region_size + 1)/pool_size*1.0)*num_filters[1])])

        #affine layer
        affine_filter_shape = [int(np.ceil((max_sentence_length - region_size + 1)/pool_size*1.0)*num_filters[1]), num_classes]
        W_AF = graph.get_tensor_by_name('W_AF:0')
        b_AF = graph.get_tensor_by_name('b_AF:0')

        self.scores = tf.matmul(self.pool_flat,W_AF)+b_AF
        self.predictions = tf.argmax(self.scores, 1, name="predictions")
        self.labels = tf.argmax(self.y_input, 1)

        #losses
        losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.y_input)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * ( tf.nn.l2_loss(W_CN) + tf.nn.l2_loss(b_CN) + tf.nn.l2_loss(W_AF) + tf.nn.l2_loss(b_AF) )

        #predictions
        correct_predictions = tf.equal(self.predictions, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy") #calulates the accuracy by summing all elements of correct_predictions and dividing it by the length of it
 
