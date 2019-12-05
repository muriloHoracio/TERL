import warnings
warnings.filterwarnings('ignore')
from tensorflow import logging
logging.set_verbosity(logging.ERROR)

import sys
import os
import numpy as np
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, AveragePooling2D, ZeroPadding2D, Activation, GlobalAveragePooling2D, Dense, Flatten
from keras.models import Model
from keras.backend import one_hot
from resnet_blocks import conv_block, identity_block
import data_handler as dh
from data_generator import DataGenerator
import metrics
import matplotlib.pyplot as plt
import time
import datetime
from train_parser import get_options
from train_parser import print_options

options = get_options(sys.argv[1:])
report_out = print_options(options)

def get_files(root):
	train_files = [root+'/Train/'+train_file for train_file in os.listdir(root+'/Train')]
	test_files = [root+'/Test/'+test_file for test_file in os.listdir(root+'/Test')]
	return train_files, test_files

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

train_files, test_files = get_files(options.root)
report_out += print_files(options.root)

db = dh.DataHandler(train_files, test_files)
report_out += print_data_info(db)
train_generator = DataGenerator(db.x_train, db.y_train, db.num_classes, batch_size=32)
test_generator = DataGenerator(db.x_test, db.y_test, db.num_classes, batch_size=32)

def One_hot(x, vocab_size):
	return one_hot(x, vocab_size)[:,:,:,np.newaxis]

inputs = Input(shape=(db.max_len,), dtype='uint8')
x = Lambda(One_hot, arguments={'vocab_size': db.vocab_size}, name='one_hot')(inputs)
x = Conv2D(16, (30, db.vocab_size), strides=(2, db.vocab_size), activation='relu', padding='valid', kernel_initializer='he_normal', name='conv1')(x)
x = AveragePooling2D((20, 1), strides=(20, 1), name='pool1')(x)
x = Conv2D(32, (30, 1), strides=(1, 1), padding='same', activation='relu', name='conv2')(x)
x = AveragePooling2D((20, 1), strides=(20, 1), name='pool2')(x)
x = Conv2D(64, (10, 1), strides=(1, 1), padding='same', activation='relu', name='conv3')(x)
x = AveragePooling2D((10, 1), strides=(10, 1), name='pool3')(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
"""
x = BatchNormalization(name='bn_conv1')(x)
x = Activation('relu')(x)
x = AveragePooling2D((30, 1), strides=(30, 1))(x)

x = conv_block(x, (3,1), [16, 16, 32], stage=2, block='a', strides=(1, 1))
x = identity_block(x, (3, 1), [16, 16, 32], stage=2, block='b')
x = identity_block(x, (3, 1), [16, 16, 32], stage=2, block='c')

#x = conv_block(x, (3, 1), [32, 32, 64], stage=3, block='a')
#x = identity_block(x, (3, 1), [32, 32, 64], stage=3, block='b')
#x = identity_block(x, (3, 1), [32, 32, 64], stage=3, block='c')
#x = identity_block(x, (3, 1), [32, 32, 64], stage=3, block='d')

#x = conv_block(x, (3, 1), [256, 256, 1024], stage=4, block='a')
#x = identity_block(x, (3, 1), [256, 256, 1024], stage=4, block='b')
#x = identity_block(x, (3, 1), [256, 256, 1024], stage=4, block='c')
#x = identity_block(x, (3, 1), [256, 256, 1024], stage=4, block='d')
#x = identity_block(x, (3, 1), [256, 256, 1024], stage=4, block='e')
#x = identity_block(x, (3, 1), [256, 256, 1024], stage=4, block='f')

x = conv_block(x, (3, 1), [16, 16, 32], stage=4, block='a')
x = identity_block(x, (3, 1), [16, 16, 32], stage=4, block='b')
x = identity_block(x, (3, 1), [16, 16, 32], stage=4, block='c')
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
"""
preds = Dense(db.num_classes, activation='softmax')(x)
model = Model(inputs, preds, name='resnet50')
model.summary()

model.compile(optimizer=options.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(generator=train_generator, validation_data=test_generator, epochs=options.epochs)

y_prob = model.predict(db.x_test)
y_predicted_classes = y_prob.argmax(axis=-1)

m = metrics.Metric(db.y_test, y_predicted_classes, classes=db.classes, filename_prefix=options.prefix)
classification_report = m.get_report()
if options.verbose: print(classification_report)
report_out += classification_report

if options.save_graphs:
	m.save_confusion_matrix(title=options.cm_title)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('Outputs/'+options.prefix+'_model_accuracies_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.png')

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
plt.savefig('Outputs/'+options.prefix+'_model_loss_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.png')


#if options.save_graphs:
#	m.save_learning_curve(accuracies, title=options.lc_title, acc_type=0)

#if options.verbose: print('\n\nTraining time: '+str(training_time)+'\nAverage test time: '+str(sum(test_times)/len(test_times)))

if options.save_report:
	#report_out += '\n\nTraining time: '+str(training_time)+'\nAverage test time: '+str(sum(test_times)/len(test_times))
	with open('Outputs/Report_'+options.prefix+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w+') as f:
		f.write(report_out)

#exit(best_result[0])
