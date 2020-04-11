import os
import argparse
import time
import datetime

HELP_MSG = '\n\nUse python cnn_train.py -h to see all options and examples of usage'

def get_options(arguments):
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-m', '--model',
		dest = 'model',
		default = [''],
		required = False,
		help = """Path to the model folder to be loaded"""
	)

	parser.add_argument(
		'-f', '--files',
		dest = 'files',
		nargs='+',
		required = True,
		help = """Relative or absolute path of the files to be classified"""
	)

	parser.add_argument(
		'-b', '--batch',
		dest = 'batch',
		default = [32],
		required = False,
		help = """Batch size to split the data"""
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
		'-e', '--export-dir',
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

	options.model = options.model
	options.batch = options.batch[0]
	options.prefix = options.prefix[0]
	options.model_export_dir = options.model_export_dir[0]

	if not os.path.isdir(options.model):
		print('ERROR\n\nPath: '+options.model>' does not exist'+HELP_MSG)
		exit(-1)
	if os.path.isdir(options.model_export_dir):
		old_export_dir = options.model_export_dir
		options.model_export_dir = 'Models/Model_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
		print('WARNING\n\nThe informed folder to export the model ('+old_export_dir+') already exists.\nThe model will be saved on folder: '+options.model_export_dir)
	return options

def print_options(options):
	out = ''
	out += '%20s %s\n' % ('Model:',options.model)
	out += '%20s\n' % ('Files:',)
	for f in options.files:
		out += '%20s %s\n' % ('',f)
	out += '%20s %d\n' % ('Batch:',options.batch)
	out += '%20s %s\n' % ('Prefix:',options.prefix)
	print(out)
