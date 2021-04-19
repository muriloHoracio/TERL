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
        help =  """Root containg "train" and  "test" folders  that
                contain train files and test files.  Each  file
                must be named according  to  the  correspondent
                class"""
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
        '-l', '--layers',
        dest = 'number_of_layers',
        type = int,
        nargs = 1,
        default = [8],
        required = False,
        help =  """Number of layers, not including the classification
                layer"""
    )

    parser.add_argument(
        '-a', '--architecture',
        dest = 'architecture',
        metavar = 'LAYER',
        type = str,
        nargs = '+',
        default = ["conv", "pool", "conv", "pool", "conv", "pool", "fc", "fc"],
        required = False,
        help =  """Architecture structure of the network. Must  be
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
        help =  """Activation functions and type of pooling layers
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
        default = [20, 10, 20, 15, 35, 15, 1000, 500],
        required = False,
        help =  """Width  of  each  layer  ordered   according  to
                --architecture  paramater. For "fc" layers, the
                number of neurons must be entered. The number of
                neurons for the classification layer must not be
                entered, since the number  of  neurons  on  this
                layer is defined  according  to  the  amount  of
                classes in train and test folders.
                Example of architecture:
                --architecture conv pool conv pool conv pool fc fc fc
                Example of widths:
                --widths 20 10 20 15 35 15 1000 500"""
    )

    parser.add_argument(
        '-s','--strides',
        dest = 'strides',
        metavar = 'STRIDE',
        type = int,
        nargs = '+',
        default = [1, 10, 1, 15, 1, 15],
        required = False,
        help =  """Strides  of  convolution  and  pooling  layers
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
        default = [64, 32, 32],
        required = False,
        help =  """Number of feature maps per  convolution  layer,
                separated by spaces.
                Example of architecture:
                --architecture conv pool conv pool conv pool fc fc fc
                Example of feature-maps:
                --feature-maps 64 32 32"""
    )

    parser.add_argument(
        '-o','--optimizer',
        dest = 'optimizer',
        metavar = 'OPTIMIZER',
        type = str,
        nargs = 1,
        default = ['adam'],
        required = False,
        help =  """Optimizer to be used to optimize the learnable
                parameters. Optimizer can be: adam, adadelta, adagrad, ftrl, 
                rmsprop, grad_desc. Default optimizer is adam. Each optimizer 
                is documented on the tensorflow API's page under the optimizer 
                section of tensorflow 1.x: www.tf.compat.v1.train"""
    )

    parser.add_argument(
        '-lr','--learning-rate',
        dest = 'learning_rate',
        metavar = 'LEARNING_RATE',
        type = float,
        nargs = 1,
        default = [0.001],
        required = False,
        help =  """Learning rate to be used by the optimizer. 
                Default learning rate is 0.001"""
    )

    parser.add_argument(
        '-l2','--l2-regulation',
        dest = 'l2',
        metavar = 'L2',
        type = float,
        nargs = 1,
        default = [0.001],
        required = False,
        help = """L2 regulation weight to be applied in the sum of all weights and biases to be discounted on the loss function. Default value is 0.001"""
    )

    parser.add_argument(
        '-dl','--dilations',
        dest = 'dilations',
        metavar = 'DILATIONS',
        type = int,
        nargs = '+',
        default = [1,1,1],
        required = False,
        help = """Dilations to be applied on each convolution layer. Default is: [1, 1, 1]"""
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
        help = """Title of the graph that will be generated containing
        the confusion matrix of test set classification"""
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
        '-sg', '--save-graph',
        dest = 'save_graph',
        const = True,
        action = 'store_const',
        default = False,
        required = False,
        help = """Sets the graphs to be saved on Outputs directory"""
    )

    parser.add_argument(
        '-sm', '--save-model',
        dest = 'save_model',
        const = True,
        action = 'store_const',
        default = False,
        required = False,
        help = """Sets the model to be saved on --model-export-dir option"""
    )

    parser.add_argument(
        '-sr', '--save-report',
        dest = 'save_report',
        const = True,
        action = 'store_const',
        default = False,
        required = False,
        help = """Sets the report to be saved on Outputs directory"""
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
    options.input_type = options.input_type[0]
    options.number_of_layers = options.number_of_layers[0]
    options.train_batch_size = options.train_batch_size[0]
    options.test_batch_size = options.test_batch_size[0]
    options.epochs = options.epochs[0]
    options.dropout = options.dropout[0]
    options.optimizer = options.optimizer[0].upper()
    options.learning_rate = options.learning_rate[0]
    options.graph_title = ' '.join(options.graph_title)
    options.prefix = options.prefix[0]
    options.model_export_dir = options.model_export_dir[0]
    options.l2 = options.l2[0]

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
    if not (len([a for a in options.architecture if a=='conv']) == len(options.dilations)):
        print('ERROR\n\nNumber of dilations is not equal to the number of convolution layers defined on --architecture parameter'+HELP_MSG)
        exit(-7)
    if not (len([a for a in options.architecture if a=='conv' or a=='pool']) == len(options.strides)):
        print('ERROR\n\nNumber of strides is not equal to the amount of convolution and pooling layers defined on --architecture parameter'+HELP_MSG)
        exit(-8)
    if options.optimizer not in ['ADAM','ADADELTA','ADAGRAD','FTRL','RMSPROP','GRAD_DESC']:
        print('ERROR\n\nOptimizer '+options.optimizer+' is not a valid optimizer option!\nAvailable optimizers are: adam, adadelta, adagrad, ftrl, rmsprop and grad_desc\nPlease choose one of the above optimizer to train your model'+HELP_MSG)
        exit(-9)
    if os.path.isdir(options.model_export_dir):
        old_export_dir = options.model_export_dir
        options.model_export_dir = 'Models/Model_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        print('WARNING\n\nThe informed folder to export the model ('+old_export_dir+') already exists.\nThe model will be saved on folder: '+options.model_export_dir)
    if not options.l2 > 0:
        print('ERROR\n\nL2 regulation weight must be a positive float value, like 0.001\nPlease try again with a positive value'+HELP_MSG)
        exit(-10)

    return options

def print_options(options):
    feature_maps_string = '%20s ' % 'Feature maps:'
    dilations_string = '%20s ' % 'Dilations:'
    j = 0
    for i in range(len(options.architecture)):
        if options.architecture[i] == 'conv':
            feature_maps_string += '%-8s' % str(options.feature_maps[j])
            dilations_string += '%-8s' % str(options.dilations[j])
            j += 1
        else:
            feature_maps_string += '%-8s' % '-'
            dilations_string += '%-8s' % '-'

    out = '*' * 79 + '\n**' + ' ' * 33 + ' OPTIONS ' + ' ' * 33 + '**\n' + '*' * 79 + '\n'
    out += '%20s %s\n' % ('Root:', options.root)
    out += '%20s %s\n' % ('Input type:', options.input_type)
    out += '%20s %d\n' % ('Train batch:', options.train_batch_size)
    out += '%20s %d\n' % ('Test batch:', options.test_batch_size)
    out += '%20s %d\n' % ('Epochs:', options.epochs)
    out += '%20s %.2f\n' % ('Dropout:', options.dropout)
    out += '%20s %d\n' % ('Number of layers:', options.number_of_layers)
    out += '%20s %s\n' % ('Optimizer:', options.optimizer)
    out += '%20s %f\n' % ('Learning rate:', options.learning_rate)
    out += '%20s %f\n' % ('L2 regulation:', options.l2)

    if options.save_graph:
        out += '%20s\n' % ('Saving graph:')
        out += '%20s %s\n' % ('Graph Title:', options.graph_title)
    else:
        out += '%20s\n' % ('Not saving graph:')

    if options.save_report:
        out += '%20s\n' % ('Saving report:')
    else:
        out += '%20s\n' % ('Not saving report:')

    if options.save_model:
        out += '%20s\n' % ('Saving model:')
        out += '%20s %s\n' % ('Model export dir:', options.model_export_dir)
    else:
        out += '%20s\n' % ('Not saving model:')

    if options.save_graph or options.save_report:
        out += '%20s %s\n' % ('Prefix:',options.prefix)

    out += '%20s %s\n' % ('Architecture:', ''
        .join('%-8s' % t for t in options.architecture))
    out += '%20s %s\n' % ('Functions:', ''
        .join('%-8s' % t for t in options.activation_functions))
    out += '%20s %s\n' % ('Widths:', ''
        .join('%-8s' % t for t in options.widths))
    out += '%20s %s\n' % ('Strides:', ''
        .join('%-8s' % t for t in options.strides))
    out += feature_maps_string + '\n'
    out += dilations_string + '\n'
    if options.verbose: print(out)
    return out
