import os
import argparse
import time
import datetime

HELP_MSG = '\n\nUse python cnn_train.py -h to see all options and examples of usage'

def get_options(arguments):
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-r','--root',
		dest = 'root',
		type = str,
		nargs = 1,
		required = True,
		help = 	"""Root containg "train" and  "test" folders  that
				contain train files and test files.  Each  file
				must be named according  to  the  correspondent
				class"""
	)
	
	parser.add_argument(
		'-l', '--layers',
		dest = 'number_of_layers',
		type = int,
		nargs = 1,
		default = [8],
		required = False,
		help = 	"""Number of layers, not including the classification
				layer"""
	)

	parser.add_argument(
		'-arc', '--architecture',
		dest = 'architecture',
		metavar = 'LAYER',
		type = str,
		nargs = '+',
		default = ["conv", "pool", "conv", "pool", "conv", "pool", "fc", "fc"],
		required = False,
		help = 	"""Architecture structure of the network. Must  be
				entered as strings separeted  by  spaces.  Each
				string defines the types of the  layers. "conv"
				stands for convolution  layers,  "pool"  stands
				for pooling layers and "fc"  stands  for  fully
				connected layers. The classification layer must
				not be included. Example: --architecture   conv
				pool conv pool conv pool fc fc"""
	)

	parser.add_argument(
		'-f', '--functions',
		dest = 'activation_functions',
		metavar = 'FUNCTION',
		type = str,
		nargs = '+',
		default = ["relu", "avg", "relu", "avg", "relu", "avg", "relu", "relu"],
		required = False,
		help = 	"""Activation functions and type of pooling layers
				ordered according to --architecture parameter's
				layers order. Must not include the function  of
				the classification layer
				Example of architectures:
				--architecture conv pool conv pool conv pool fc fc fc
				Example of functions:
				--fuctions relu avg relu avg relu avg relu relu"""
	)

	parser.add_argument(
		'-w','--widths',
		dest = 'widths',
		metavar = 'LAYER_WIDTH',
		type = int,
		nargs = '+',
		default = [30, 20, 30, 20, 30, 10, 1500, 500],
		required = False,
		help = 	"""Width  of  each  layer  ordered   according  to
				--architecture  paramater. For "fc" layers, the
				number of neurons must be entered. The number of
				neurons for the classification layer must not be
				entered, since the number  of  neurons  on  this
				layer is defined  according  to  the  amount  of
				classes in train and test folders.
				Example of architecture:
				--architecture conv pool conv pool conv pool fc fc fc
				Example of widths:
				--widths 30 20 30 20 30 10 1500 500"""
	)

	parser.add_argument(
		'-s','--strides',
		dest = 'strides',
		metavar = 'STRIDE',
		type = int,
		nargs = '+',
		default = [1, 20, 1, 20, 1, 10],
		required = False,
		help = 	"""Strides  of  convolution  and  pooling  layers
				ordered according to --architecture parameter.
				Example of architecture:
				--architecture conv pool conv pool conv pool fc fc fc
				Example of strides:
				--strides 1 20 1 20 1 10
				Being 1 1 1 the strides of convolution layers
				and 20 20 20 the strides of pooling layers"""
	)

	parser.add_argument(
		'-fm','--feature-maps',
		dest = 'feature_maps',
		metavar = 'NUMBER_OF_FILTERS',
		type = int,
		nargs = '+',
		default = [64, 32, 16],
		required = False,
		help = 	"""Number of feature maps per  convolution  layer,
				separated by spaces.
				Example of architecture:
				--architecture conv pool conv pool conv pool fc fc fc
				Example of feature-maps:
				--feature-maps 64 32 16"""
	)
	
	parser.add_argument(
		'-trb', '--train-batch-size',
		dest = 'train_batch_size',
		type = int,
		nargs = 1,
		default = [32],
		required = False,
		help = """Train batch size"""
	)

	parser.add_argument(
		'-tsb','--test-batch-size',
		dest = 'test_batch_size',
		type = int,
		nargs = 1,
		default = [32],
		required = False,
		help = """Test batch size"""
	)
	
	parser.add_argument(
		'-e','--epochs',
		dest = 'epochs',
		type = int,
		nargs = 1,
		default = [10],
		required = False,
		help = """Number of epochs"""
	)

	parser.add_argument(
		'-d','--dropout',
		dest = 'dropout',
		type = float,
		nargs = 1,
		default = [0.5],
		required = False,
		help = """Dropout rate"""
	)

	parser.add_argument(
		'-gt', '--graph-title',
		dest = 'graph_title',
		type = str,
		nargs = '+',
		default = ['Confusion Matrix'],
		required = False,
		help = """Title of the graph that will be generated containing the confusion matrix of test set classification"""
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

	options =  parser.parse_args(arguments)

	options.root = options.root[0]
	if options.root[-1] == '/':
		options.root = options.root[0:-1]
	options.number_of_layers = options.number_of_layers[0]
	options.train_batch_size = options.train_batch_size[0]
	options.test_batch_size = options.test_batch_size[0]
	options.epochs = options.epochs[0]
	options.dropout = options.dropout[0]
	options.graph_title = ' '.join(options.graph_title)
	options.prefix = options.prefix[0]
	options.model_export_dir = options.model_export_dir[0]

	if not os.path.isdir(options.root + '/Train'):
		print('ERROR\n\nPath: '+options.root+'/Train does not exist'+HELP_MSG)
		exit(-1)
	if not os.path.isdir(options.root + '/Test'):
		print('ERROR\n\nPath: '+options.root+'/Test does not exist'+HELP_MSG)
		exit(-2)
	if not (options.number_of_layers == len(options.architecture)):
		print('ERROR\n\nNumber of layers is not equal to the number of layers defined on --architecture parameter'+HELP_MSG)
		exit(-3)
	if not (options.number_of_layers == len(options.activation_functions)):
		print('ERROR\n\nNumber of functions is not equal to the number of layers defined on --architecture parameter'+HELP_MSG)
		exit(-4)
	if not (options.number_of_layers == len(options.widths)):
		print('ERROR\n\nNumber of widths is not equal to the number of layers defined on --architecture parameter'+HELP_MSG)
		exit(-5)
	if not (len([a for a in options.architecture if a=='conv']) == len(options.feature_maps)):
		print('ERROR\n\nNumber of feature maps is not equal to the number of convolution layers defined on --architecture parameter'+HELP_MSG)
		exit(-6)
	if not (len([a for a in options.architecture if a=='conv' or a=='pool']) == len(options.strides)):
		print('ERROR\n\nNumber of strides is not equal to the amount of convolution and pooling layers defined on --architecture parameter'+HELP_MSG)
		exit(-7)
	if os.path.isdir(options.model_export_dir):
		old_export_dir = options.model_export_dir
		options.model_export_dir = 'Models/Model_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
		print('WARNING\n\nThe informed folder to export the model ('+old_export_dir+') already exists.\nThe model will be saved on folder: '+options.model_export_dir)
		
	return options

def print_options(options):
	print('*' * 79 + '\n**' + ' ' * 33 + ' OPTIONS ' + ' ' * 33 + '**\n' + '*' * 79)
	print('%20s %s' % ('Root:',options.root))
	print('%20s %d' % ('Train batch:',options.train_batch_size))
	print('%20s %d' % ('Test batch:',options.test_batch_size))
	print('%20s %d' % ('Epochs:',options.epochs))
	print('%20s %.2f' % ('Dropout:',options.dropout))
	print('%20s %d' % ('Number of layers:',options.number_of_layers))
	print('%20s %s' % ('Graph Title:',options.graph_title))
	print('%20s %s' % ('Prefix:',options.prefix))
	print('%20s %s' % ('Model export dir:',options.model_export_dir))
	print('%20s %s' % ('Architecture:',''.join('%-8s' % t for t in options.architecture)))
	print('%20s %s' % ('Functions:',''.join('%-8s' % t for t in options.activation_functions)))
	print('%20s %s' % ('Widths:',''.join('%-8s' % t for t in options.widths)))
	print('%20s %s' % ('Strides:',''.join('%-8s' % t for t in options.strides)))
	feature_maps_string = '%20s ' % 'Feature maps:'
	j = 0
	for i in range(len(options.architecture)):
		if options.architecture[i] == 'conv':
			feature_maps_string += '%-8s' % str(options.feature_maps[j])
			j += 1
		else:
			feature_maps_string += '%-8s' % '-'
	print(feature_maps_string)
