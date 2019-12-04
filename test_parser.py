import os
import argparse
import time
import datetime

HELP_MSG = '\n\nUse python cnn_train.py -h to see all options and examples of usage'

def get_options(arguments):
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-m', '--model-dir',
		dest = 'model_dir',
		default = [''],
		required = False,
		help = """Path to the model folder to be loaded"""
	)

	parser.add_argument(
		'-f', '--files',
		dest = 'files',
		required = True,
		help = """Relative or absolute path of the files to be classified"""
	)

	parser.add_argument(
		'-b', '--batch-size',
		dest = 'batch_size',
		default = [32],
		required = False,
		help = """Batch size to split the data"""
	)

	parser.add_argument(
		'-sg', '--save-graphs',
		dest = 'save_graphs',
		const = True,
		action = 'store_const',
		default = False,
		required = False,
		help = """Sets the graphs to be saved on Outputs directory"""
	)

	parser.add_argument(
		'-p','--prefix',
		dest = 'prefix',
		type = str,
		nargs = 1,
		default = ['RUN_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')],
		required = False,
		help = """Prefix of the file names to be created"""
	)

	parser.add_argument(
		'-md', '--model-export-dir',
		dest = 'model_export_dir',
		type = str,
		nargs = 1,
		default = ['Models/Model_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')],
		required = False,
		help = """Folder in which the trained model will be stored 
		to be used as a classifier by the cnn_test.py program. If 
		the folder already exists the user will be prompted to overwrite
		the current existing folder with the trained model or to enter
		a new folder to store the model."""
	)

	parser.add_argument(
		'-nv', '--no-verbose',
		dest = 'verbose',
		const = False,
		action = 'store_const',
		default = True,
		required = False,
		help = """Sets verbosity on"""
	)

	options =  parser.parse_args(arguments)

	options.root = options.root[0]
	if options.root[-1] == '/':
		options.root = options.root[0:-1]
	options.number_of_layers = options.number_of_layers[0]
	options.train_batch_size = options.train_batch_size[0]
	options.test_batch_size = options.test_batch_size[0]
	options.epochs = options.epochs[0]
	options.dropout = options.dropout[0]
	options.optimizer = options.optimizer[0].upper()
	options.learning_rate = options.learning_rate[0]
	options.cm_title = ' '.join(options.cm_title)
	options.lc_title = ' '.join(options.lc_title)
	options.prefix = options.prefix[0]
	options.model_export_dir = options.model_export_dir[0]

	if not os.path.isdir(options.root + '/Train'):
		print('ERROR\n\nPath: '+options.root+'/Train does not exist'+HELP_MSG)
		exit(-1)
	if not os.path.isdir(options.root + '/Test'):
		print('ERROR\n\nPath: '+options.root+'/Test does not exist'+HELP_MSG)
		exit(-2)
	if options.number_of_layers > 0 and not (options.number_of_layers == len(options.architecture)):
		print('ERROR\n\nNumber of layers is not equal to the number of layers defined on --architecture parameter'+HELP_MSG)
		exit(-3)
	if options.number_of_layers > 0 and not (options.number_of_layers == len(options.functions)):
		print('ERROR\n\nNumber of functions is not equal to the number of layers defined on --architecture parameter'+HELP_MSG)
		exit(-4)
	if options.number_of_layers > 0 and not (options.number_of_layers == len(options.widths)):
		print('ERROR\n\nNumber of widths is not equal to the number of layers defined on --architecture parameter'+HELP_MSG)
		exit(-5)
	if not (len([a for a in options.architecture if a=='conv']) == len(options.feature_maps)):
		print('ERROR\n\nNumber of feature maps is not equal to the number of convolution layers defined on --architecture parameter'+HELP_MSG)
		exit(-6)
	if not (len([a for a in options.architecture if a=='conv' or a=='pool']) == len(options.strides)):
		print('ERROR\n\nNumber of strides is not equal to the amount of convolution and pooling layers defined on --architecture parameter'+HELP_MSG)
		exit(-7)
	if options.optimizer not in ['ADAM','ADADELTA','ADAGRAD','FTRL','RMSPROP','GRAD_DESC']:
		print('ERROR\n\nOptimizer '+options.optimizer+' is not a valid optimizer option!\nAvailable optimizers are: adam, adadelta, adagrad, ftrl, rmsprop and grad_desc\nPlease choose one of the above optimizer to train your model'+HELP_MSG)
		exit(-8)
	for l in options.architecture:
		if l.upper() not in ['CONV','POOL','FC']:
			print('ERROR\n\nArchitecture parameter "'+l+'" is not a valid layer type option!\nAvailable layer types are: conv, pool, fc\nPlease choose one of the above layer types'+HELP_MSG)
			exit(-9)
	for func in options.functions:
		if func.upper() not in ['RELU','TANH','SIGMOID','LEAKY_RELU','ELU','AVG','MAX']:
			print('ERROR\n\nFunctions parameter "'+func+'" is not a valid function option!\nAvailable functions are: relu, tanh, sigmoid, leaky_relu, elu, avg and max\nPlease choose one of the above layer types'+HELP_MSG)
			exit(-10)
	if os.path.isdir(options.model_export_dir):
		old_export_dir = options.model_export_dir
		options.model_export_dir = 'Models/Model_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
		print('WARNING\n\nThe informed folder to export the model ('+old_export_dir+') already exists.\nThe model will be saved on folder: '+options.model_export_dir)

	options.number_of_layers = len(options.architecture)
		
	return options

def print_options(options):
	feature_maps_string = '%20s ' % 'Feature maps:'
	j = 0
	for i in range(len(options.architecture)):
		if options.architecture[i] == 'conv':
			feature_maps_string += '%-8s' % str(options.feature_maps[j])
			j += 1
		else:
			feature_maps_string += '%-8s' % '-'
	out = '*' * 79 + '\n**' + ' ' * 33 + ' OPTIONS ' + ' ' * 33 + '**\n' + '*' * 79 + '\n'
	out += '%20s %s\n' % ('Root:',options.root)
	out += '%20s %d\n' % ('Train batch:',options.train_batch_size)
	out += '%20s %d\n' % ('Test batch:',options.test_batch_size)
	out += '%20s %d\n' % ('Epochs:',options.epochs)
	out += '%20s %.2f\n' % ('Dropout:',options.dropout)
	out += '%20s %d\n' % ('Number of layers:',options.number_of_layers)
	out += '%20s %s\n' % ('Optimizer:',options.optimizer)
	out += '%20s %f\n' % ('Learning rate:',options.learning_rate)

	if options.save_graphs:
		out += '%20s\n' % ('Saving graphs:')
		out += '%20s %s\n' % ('Confusion Matrix Title:',options.cm_title)
		out += '%20s %s\n' % ('Learning Curve Title:',options.lc_title)
	else:
		out += '%20s\n' % ('Not saving graphs:')

	if options.save_report:
		out += '%20s\n' % ('Saving report:')
	else:
		out += '%20s\n' % ('Not saving report:')

	if options.save_model:
		out += '%20s\n' % ('Saving model:')
		out += '%20s %s\n' % ('Model export dir:',options.model_export_dir)
	else:
		out += '%20s\n' % ('Not saving model:')

	if options.save_graphs or options.save_report:
		out += '%20s %s\n' % ('Prefix:',options.prefix)

	out += '%20s %s\n' % ('Architecture:',''.join('%-8s' % t for t in options.architecture))
	out += '%20s %s\n' % ('Functions:',''.join('%-8s' % t for t in options.functions))
	out += '%20s %s\n' % ('Widths:',''.join('%-8s' % t for t in options.widths))
	out += '%20s %s\n' % ('Strides:',''.join('%-8s' % t for t in options.strides))
	out += feature_maps_string + '\n'
	if options.verbose: print(out)
	return out

