import os
import argparse
import time
import datetime

HELP_MSG = '\n\nUse python cnn_train.py -h to see all options and examples of usage'

def get_arguments(arguments):
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-r','--root',
		dest = 'root',
		type = str,
		nargs = 1,
		required = True,
		help = """Root containg "train" and  "test" folders  that 
contain train files and test files.  Each  file
must be named according  to  the  correspondent
class"""
	)

	parser.add_argument(
		'-l', '--layers',
		dest = 'number_of_layers',
		type = int,
		nargs = 1,
		default = 8,
		required = False,
		help = """Number of layers, not including the classification
layer"""
	)

	parser.add_argument(
		'-arc', '--architecture',
		dest = 'architecture',
		type = str,
		nargs = '+',
		default = ["conv", "pool", "conv", "pool", "conv", "pool", "fc", "fc"],
		required = False,
		help = """Architecture structure of the network. Must  be
entered as strings separeted  by  spaces.  Each
string defines the types of the  layers. "conv"
stands for convolution  layers,  "pool"  stands
for pooling layers and "fc"  stands  for  fully
connected layers. The classification layer must
not be included
Example:
--architecture conv pool conv pool conv pool fc fc"""
	)

	parser.add_argument(
		'-f', '--functions',
		dest = 'functions',
		type = str,
		nargs = '+',
		default = ["relu", "avg", "relu", "avg", "relu", "avg", "relu", "relu"],
		required = False,
		help = """Activation functions and type of pooling layers
ordered according to --architecture parameter's
layers order. Must not include the function  of
the classification layer
Example of --architecture:
--architecture conv pool conv pool conv pool fc fc fc
Example of --functions:
--fuctions relu avg relu avg relu avg relu relu"""
	)

	parser.add_argument(
		'-w','--widths',
		dest = 'widths',
		type = int,
		nargs = '+',
		default = [30, 20, 30, 20, 30, 10, 1500, 500],
		required = False,
		help = """Width  of  each  layer  ordered   according  to
--architecture  paramater. For "fc" layers, the
number of neurons must be entered. The number of
neurons for the classification layer must not be
entered, since the number  of  neurons  on  this
layer is defined  according  to  the  amount  of
classes in train and test folders.
Example of --architecture:
--architecture conv pool conv pool conv pool fc fc fc
Example of --widths:
--widths 30 20 30 20 30 10 1500 500"""
	)

	parser.add_argument(
		'-cs','--strides',
		dest = 'strides',
		type = int,
		nargs = '+',
		default = [1, 20, 1, 20, 1, 10],
		required = False,
		help = """Strides  of  convolution  and  pooling  layers
ordered according to --architecture parameter.
Example of --architecture:
--architecture conv pool conv pool conv pool fc fc fc
Example of --strides:
--strides 1 20 1 20 1 10
Being 1 1 1 the strides of convolution layers
and 20 20 20 the strides of pooling layers"""
	)

	parser.add_argument(
		'-fm','--feature-maps',
		dest = 'feature_maps',
		type = int,
		nargs = '+',
		default = [64, 32, 16],
		required = False,
		help = """Number of feature maps per  convolution  layer,
separated by spaces.
Example of --architecture:
--architecture conv pool conv pool conv pool fc fc fc
Example of --feature-maps:
--feature-maps 64 32 16"""
	)

	parser.add_argument(
		'-trb', '--train-batch-size',
		dest = 'train_batch_size',
		type = int,
		nargs = 1,
		default = 32,
		required = False,
		help = """Train batch size"""
	)

	parser.add_argument(
		'-tsb','--test-batch-size',
		dest = 'test_batch_size',
		type = int,
		nargs = 1,
		default = 32,
		required = False,
		help = """Test batch size"""
	)

	parser.add_argument(
		'-e','--epochs',
		dest = 'epochs',
		type = int,
		nargs = 1,
		default = 30,
		required = False,
		help = """Number of epochs"""
	)

	parser.add_argument(
		'-d','--dropout',
		dest = 'dropout',
		type = float,
		nargs = 1,
		default = 0.5,
		required = False,
		help = """Dropout rate"""
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

	options =  parser.parse_args(arguments)

	if not os.path.isdir(options.root[0] + '/Train'):
		print('ERROR\n\nPath: '+options.root[0]+'/Train does not exist'+HELP_MSG)
		exit(1)
	if not os.path.isdir(options.root[0] + '/Test'):
		print('ERROR\n\nPath: '+options.root[0]+'/Test does not exist'+HELP_MSG)
		exit(2)
	if not (options.number_of_layers == len(options.architecture)):
		print('ERROR\n\nNumber of layers is not equal to the number of layers defined on --architecture parameter'+HELP_MSG)
		exit(3)
	if not (options.number_of_layers == len(options.functions)):
		print('ERROR\n\nNumber of functions is not equal to the number of layers defined on --architecture parameter'+HELP_MSG)
		exit(4)
	if not (options.number_of_layers == len(options.widths)):
		print('ERROR\n\nNumber of widths is not equal to the number of layers defined on --architecture parameter'+HELP_MSG)
		exit(5)
	if not (len([a for a in options.architecture if a=='conv']) == len(options.feature_maps)):
		print('ERROR\n\nNumber of feature maps is not equal to the number of convolution layers defined on --architecture parameter'+HELP_MSG)
		exit(6)
	if not (len([a for a in options.architecture if a=='conv' or a=='pool']) == len(options.strides)):
		print('ERROR\n\nNumber of strides is not equal to the amount of convolution and pooling layers defined on --architecture parameter'+HELP_MSG)
		exit(7)
	return options

def print_options(options):
	msg = 'Root: '+options.root[0]
	msg += '\nTrain batch size: '+str(options.train_batch_size)
	msg += '\nTest batch size: '+str(options.test_batch_size)
	msg += '\nEpochs: '+str(options.epochs)
	msg += '\nDropout: '+str(options.dropout)
	msg += '\nNumber of layers: '+str(options.number_of_layers)
	msg += '\nArchitecture:\n\t'+'\t'.join([str(a) for a in options.architecture])
	msg += '\nFunctions:\n\t'+'\t'.join([str(f) for f in options.functions])
	msg += '\nWidths:\n\t'+'\t'.join([str(w) for w in options.widths])
	msg += '\nStrides:\n\t'+'\t'.join([str(s) for s in options.strides])
	msg += '\nFeature maps:\n\t'+'\t'.join([str(f) for f in options.feature_maps])
	msg += '\nPrefix: '+options.prefix[0]
	print(msg)
