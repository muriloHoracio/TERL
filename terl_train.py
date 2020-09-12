import sys
import warnings
warnings.filterwarnings('ignore')

from train_parser import get_options
from train_parser import print_options
options = get_options(sys.argv[1:])

from typing import List, Tuple
import os
import time
import datetime
import pickle
import numpy as np
import metrics
import tensorflow.compat.v1 as tf
from cnn_model import Model
from data_handler import DataHandler

tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()

report_out = print_options(options)

def get_files(root: str) -> Tuple[List[str], List[str]]:
    train_files = [os.path.join(root, 'Train', train_file
        ) for train_file in os.listdir(os.path.join(root, 'Train'))]
    test_files = [os.path.join(root, 'Test', test_file
        ) for test_file in os.listdir(os.path.join(root,'Test'))]
    return train_files, test_files

def print_files(root: str) -> str:
    out = f'{"*" * 79}\n**{" " * 34} FILES {" " * 34}**\n{"*" * 79}\n'
    
    train_files_str = ''.join(f'\n\t{t}' for t in os.listdir(
        os.path.join(options.root, 'Train')))
    out += f'Train{train_files_str}\n'

    test_files_str = ''.join(f'\n\t{t}' for t in os.listdir(
        os.path.join(options.root, 'Test')))
    out += f'Test{test_files_str}\n'

    if options.verbose:
        print(out)
    return out

class Terl:
    
    def __init__(self,
                 dataset:DataHandler,
                 architecture: List[str],
                 activation_functions: List[str],
                 widths: List[int],
                 strides: List[int],
                 dilations: List[int],
                 feature_maps: List[int],
                 optimizer: str,
                 learning_rate: float,
                 l2: float,
                 train_batch: int,
                 test_batch: int,
                 epochs: int,
                 dropout: float,
                 verbose: bool=True,
                 save_model: bool=True,
                 model_export_dir: str=''):
        self.ds = dataset
        self.architecture = architecture
        self.activation_functions = activation_functions
        self.widths = widths
        self.strides = strides
        self.dilations = dilations
        self.feature_maps = feature_maps
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.l2 = l2
        self.train_batch = train_batch
        self.test_batch = test_batch
        self.epochs = epochs
        self.dropout = dropout
        self.verbose = verbose
        self.save_model = save_model
        self.model_export_dir = model_export_dir

        self.cnn = None
        self.sess = None
        self.train_op = None
        self.global_step = None
        self._one_hot_x = None
        self._one_hot_y = None
        self._pre_x = None
        self._pre_y = None
        self.print_out = ''
        
        if self.model_export_dir == '':
            self.model_export_dir = 'Models/'+datetime.datetime.now(
                ).strftime('%Y%m%d_%H%M%S')

        self.train_length = len(self.ds.y_train)
        self.test_length = len(self.ds.y_test)
        self.accuracies = []
        self.training_time = 0
        self.test_times = []
        self.best_result = [0, [], []]
        self._optimizers = {
            'ADAM': tf.train.AdamOptimizer,
            'ADADELTA': tf.train.AdadeltaOptimizer,
            'ADAGRAD': tf.train.AdagradOptimizer,
            'FTRL': tf.train.FtrlOptimizer,
            'RMSPROP': tf.train.RMSPropOptimizer,
            'GRAD_DESC': tf.train.GradientDescentOptimizer
        }

        self.optimizer = self.get_optimizer()
    
    def get_optimizer(self):
        return self._optimizers[self.optimizer_name](
            learning_rate=self.learning_rate)

    def _train_step(self, x_batch, y_batch):
        if not self.sess:
            raise AssertionError("There is no session")
        if not self.train_op:
            raise AssertionError("There is no train_op")

        feed_dict = {
            self.cnn.x_input: x_batch,
            self.cnn.y_input: y_batch
        }
        _, step = self.sess.run([self.train_op,
            self.global_step], feed_dict)
    
    def _eval_step(self, x_batch, y_batch):
        if not self.sess or not self.cnn:
            raise Exception("There is no session or cnn")

        feed_dict = {
            self.cnn.x_input: x_batch,
            self.cnn.y_input: y_batch
        }
        predictions, scores = self.sess.run([self.cnn.layers['pred'],
            self.cnn.layers['scores']],feed_dict)
        return predictions, scores
    
    def _evaluate(self, batch_size: int, epoch: int):
        predictions = np.array([], dtype=np.uint8)
        scores = None
        for i in range(0, self.test_length, batch_size):
            x_batch = self.ds.x_test[i : i + batch_size]
            y_batch = self.ds.y_test[i : i + batch_size]
            pre_xo = self.sess.run(self._one_hot_x,
                feed_dict={self._pre_x:x_batch})
            x_batch = pre_xo.reshape(x_batch.shape[0], self.ds.max_len,
                self.ds.vocab_size, 1)
            y_batch = self.sess.run(self._one_hot_y,
                feed_dict={self._pre_y:y_batch})
            preds, scr = self._eval_step(x_batch, y_batch)
            predictions = np.concatenate([predictions, preds])
            if scores is not None:
                scores = np.concatenate([scores, scr])
            else:
                scores = scr

        m = metrics.Metric(self.ds.y_test, predictions,
            classes=self.ds.classes)
        self.accuracies.append([epoch, m.accuracy_M, m.accuracy_m,
            m.accuracy])
        return predictions, scores

    def initialize_vars(self):
        # initialize model
        self.cnn = Model(
            self.ds.num_classes,
            self.ds.classes,
            self.architecture,
            self.activation_functions,
            self.widths,
            self.strides,
            self.dilations,
            self.feature_maps,
            self.ds.vocab_size,
            self.ds.max_len,
            self.l2
        )

        # create tensorflow variables
        self.global_step = tf.Variable(0, name="global_step",
            trainable=False)
        grads_and_vars = self.optimizer.compute_gradients(
            self.cnn.loss)
        self.train_op = self.optimizer.apply_gradients(grads_and_vars,
            global_step=self.global_step)

        self._pre_x = tf.placeholder(tf.uint8, [None, self.ds.max_len],
            name='pre_x')
        self._pre_y = tf.placeholder(tf.uint8, [None], name='pre_y')
        self._one_hot_x = tf.one_hot(self._pre_x, self.ds.vocab_size,
            dtype=tf.float32, name='one_hot_x')
        self._one_hot_y = tf.one_hot(self._pre_y, self.ds.num_classes,
            dtype=tf.float32, name='one_hot_y')
        
        # initialize global and local variables
        self.sess.run([tf.global_variables_initializer(),
            tf.local_variables_initializer()])

    def save(self):
        if self.save_model:
            if self.verbose:
                print("Saving Model")
            now = datetime.datetime.now()
            time_str = f'{now.year}_{now.month}_{now.day}__'\
                f'{now.hour}_{now.minute}_{now.second}'
            
            tf.saved_model.simple_save(
                self.sess,
                f'{self.model_export_dir}_{time_str}',
                inputs={'x_input': self.cnn.x_input},
                outputs={'prediction': self.cnn.layers['pred']}
            )

    def train(self):
        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True,
                log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                self.initialize_vars()
                
                self.training_time = time.time()
                for epoch in range(self.epochs):
                    for batch in range(0, self.train_length,
                            self.train_batch):
                        x_batch = self.ds.x_train[
                            batch:batch + self.train_batch]
                        y_batch = self.ds.y_train[
                            batch:batch + self.train_batch]
                        pre_xo = self.sess.run(self._one_hot_x,
                            feed_dict={self._pre_x: x_batch})
                        x_batch = pre_xo.reshape(x_batch.shape[0],
                            self.ds.max_len, self.ds.vocab_size, 1)
                        y_batch = self.sess.run(self._one_hot_y,
                            feed_dict={self._pre_y:y_batch})
                        self._train_step(x_batch, y_batch)
                        current_step = tf.train.global_step(self.sess,
                            self.global_step)

                    # shuffles indices for next epoch
                    shuffle_indices = np.random.permutation(
                        range(self.train_length))
                    self.ds.x_train = self.ds.x_train[
                        shuffle_indices]
                    self.ds.y_train = self.ds.y_train[
                        shuffle_indices]
                    
                    # test net
                    self.test_times.append(time.time())
                    predictions, scores = self._evaluate(self.test_batch,
                        epoch)
                    self.test_times[-1] = time.time() - self.test_times[-1]
                    if self.accuracies[-1][1] > self.best_result[0]:
                        self.best_result = [self.accuracies[-1][1],
                            np.copy(predictions), np.copy(scores)]
                    time_str = datetime.datetime.now().isoformat()
                    self.print_out += f'{time_str}: {self.accuracies[-1]}\n'
                    if self.verbose:
                        print(f'{time_str}: {self.accuracies[-1]}')

                self.training_time = time.time() - self.training_time
                self.save()

    def test(self):
        self.test_times.append(time.time())
        predictions = self._evaluate(self.test_batch, self.epochs)
        self.test_times[-1] = time.time() - self.test_times[-1]
        return np.array(predictions[1], dtype=np.uint8)


train_files, test_files = get_files(options.root)
report_out += print_files(options.root)

ds = DataHandler(train_files, test_files)
if options.verbose:
    print(ds.get_info_str())
report_out += ds.get_info_str()

shuffled = np.random.permutation(range(ds.train_size))
ds.x_train = ds.x_train[shuffled]
ds.y_train = ds.y_train[shuffled]

terl = Terl(ds,
    options.architecture,
    options.activation_functions,
    options.widths,
    options.strides,
    options.dilations,
    options.feature_maps,
    options.optimizer,
    options.learning_rate,
    options.l2,
    options.train_batch_size,
    options.test_batch_size,
    options.epochs,
    options.dropout,
    options.verbose,
    options.save_model,
    options.model_export_dir)

terl.train()
terl.test()

report_out += terl.print_out

model_data = {
    'y': terl.ds.y_test,
    'scores': terl.best_result,
    'classes': ds.classes
}
pickle.dump(model_data, open(f'model_data_{os.path.basename(options.root)}.p',
    'wb'))

m = metrics.Metric(terl.ds.y_test, terl.best_result[1],
    classes=terl.ds.classes, filename_prefix=options.prefix)

metrics_report = m.get_report()

if options.verbose:
    print(metrics_report)

report_out += metrics_report

if options.save_graph:
    print('Saving graph')
    m.save_confusion_matrix(title=options.graph_title)
    m.save_learning_curve(terl.accuracies,
        f'Learning Curve - {os.path.basename(options.root)}', acc_type=0)

if options.verbose:
    print(f'\n\nTraining time: {terl.training_time}\n'\
        f'Average test time: {np.average(terl.test_times)}')

if options.save_report:
    print('Saving report')
    report_out += f'\n\nTraining time: {terl.training_time}\n'\
        f'Average test time: {np.average(terl.test_times)}'
    with open(f'Outputs/Report_{options.prefix}_'\
        f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
        'w+') as f:
        f.write(report_out)
