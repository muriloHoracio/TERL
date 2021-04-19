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
        '-it', '--input-type',
        dest = 'input_type',
        type = str,
        nargs = 1,
        default = ['genomic'],
        choices = ['genomic', 'information-matrix'],
        required = False,
        help = """Input type. Defines which preprocessing steps
               will be used to prepare data for network input
               layer. Must be either "genomic" or "information-
               matrix". "genomic" is the default value"""
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
        default = ['TERL_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'],
        required = False,
        help = """Prefix of the file names to be created. Default prefix is '\
            '\"TERL_YYYYmmdd_HHMMSS\" where YYYY, mm, dd is the current year,' \
            'month and day. And HH, MM, SS is the current hour, minute and '\
            'seconds"""
    )

    parser.add_argument(
        '-q', '--quiet',
        dest = 'verbose',
        const = False,
        action = 'store_const',
        default = True,
        required = False,
        help = """Sets verbosity on"""
    )

    options =  parser.parse_args(arguments)

    options.model = options.model
    options.input_type = options.input_type[0]
    options.batch = options.batch[0]
    options.prefix = options.prefix[0]

    if not os.path.isdir(options.model):
        print('ERROR\n\nPath: '+options.model+' does not exist'+HELP_MSG)
        exit(-1)
    for fl in options.files:
        if not os.path.exists(fl):
            print('ERROR\n\nPath: '+fl+' does not exist'+HELP_MSG)
            exit(-2)
        if not os.path.isfile(fl):
            print('ERROR\n\nPath: '+fl+' does not represent a file'+HELP_MSG)
            exit(-2)
    return options

def print_options(options):
    out = '*' * 79 + '\n**' + ' ' * 33 + ' OPTIONS ' + ' ' * 33 + '**\n' + '*' * 79 + '\n'
    out += '%20s %s\n' % ('Model:', options.model)
    out += '%20s %s\n' % ('Input type:', options.input_type)
    out += '%20s\n' % ('Files:')
    for f in options.files:
        out += '%20s %s\n' % ('',f)
    out += '%20s %d\n' % ('Batch:', options.batch)
    out += '%20s %s\n' % ('Prefix:', options.prefix)
    out += '%20s %s\n' % ('Verbose:', options.verbose)
    print(out)
