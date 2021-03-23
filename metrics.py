import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools

import time
import datetime
from sklearn import metrics as sk_m
from typing import List

class Metric:
    def __init__(self,
                    labels: List[int], 
                    predictions: List[int], 
                    classes: List[str]=[], 
                    f_beta: float=1.0, 
                    cm: List[List[int]]=[], 
                    output_dir: str='./Outputs', 
                    filename_prefix: str=''):
        self.labels = labels
        self.predictions = predictions
        self.output_dir=output_dir
        self.filename_prefix=filename_prefix
        self.predictions = predictions
        self.f_beta = f_beta
        if cm == []:
            self.cm = sk_m.confusion_matrix(self.labels, self.predictions)
        else:
            self.cm = cm

        self.tp = self.cm.diagonal()
        self.fp = sum(self.cm) - self.tp
        self.fn = sum(np.transpose(self.cm)) - self.tp
        self.tn = sum(sum(self.cm)) - (self.tp + self.fp + self.fn)

        self.n = len(self.tp)
        if classes != []:
            self.classes = classes
        else:
            self.classes = [f'Class {i}' for i in range(self.n)]
        self.num_classes = len(self.classes)

        self.sum_tp = sum(self.tp)
        self.sum_fp = sum(self.fp)
        self.sum_fn = sum(self.fn)
        self.sum_tn = sum(self.tn)

        self.accuracies = (self.tp + self.tn)/(self.tp + self.tn + self.fp
            + self.fn + 0.000000001) * 1.0
        self.accuracy_M = sum(self.accuracies)/self.n * 1.0
        self.accuracy_m = (self.sum_tp + self.sum_tn)/(self.sum_tp
            + self.sum_tn + self.sum_fp + self.sum_fn + 0.000000001) * 1.0
        self.accuracy = sum(self.tp)/(sum(sum(self.cm)) + 0.000000001)*1.0

        self.error_rates = (self.fp + self.fn)/(self.tp + self.tn + self.fp
            + self.fn + 0.000000001)*1.0
        self.error_rate_M = sum(self.error_rates)/self.n * 1.0
        self.error_rate_m = (self.sum_fp + self.sum_fn)/(self.sum_tp
            + self.sum_fp + self.sum_fn + self.sum_tn + 0.000000001) * 1.0

        self.precisions = (self.tp)/(self.tp + self.fp + 0.000000001) * 1.0
        self.precision_M = sum(self.precisions)/self.n*1.0
        self.precision_m = self.sum_tp/(self.sum_tp + self.sum_fp
            + 0.000000001) * 1.0

        self.recalls = (self.tp)/(self.tp + self.fn + 0.000000001) * 1.0
        self.recall_M = sum(self.recalls)/self.n*1.0
        self.recall_m = self.sum_tp/(self.sum_tp + self.sum_fn + 0.000000001)

        self.fscores = (self.f_beta**2 + 1.0)*self.precisions*self.recalls/(
            (self.f_beta**2)*self.precisions + self.recalls + 0.000000001)
        self.fscore_M = (self.f_beta**2+1.0)*self.precision_M*self.recall_M/(
            (self.f_beta**2)*self.precision_M + self.recall_M + 0.000000001)
        self.fscore_m = (self.f_beta**2+1.0)*self.precision_m*self.recall_m/(
            (self.f_beta**2)*self.precision_m + self.recall_m + 0.000000001)

        self.specificity = (self.tn)/(self.tn+self.fp + 0.0000000001)*1.0
        self.specificity_M = sum(self.specificity)/self.n*1.0
        self.specificity_m = self.sum_tn/(
            self.sum_tn + self.sum_fp + 0.00000000001)*1.0

    def get_report(self):
        out = f'{"*" * 79}\n**{" " * 26} CLASSIFICATION REPORT '\
            f'{" " * 26}**\n{"*" * 79}\n'
        out += 'Confusion Matrix (row = true, column = predicted):\n'
        out += str(self.cm) + '\n'
        out += '\nStatistics:\n'
        out += f'{"Classes":10s} {"Accuracy":10s} {"Error":10s} '\
            f'{"Precision":10s} {"Recall":10s} {"Specificity":10s} '\
            f'{"F1-score":10s}\n'
        for i in range(self.n):
            out += f'{self.classes[i]:10s} {self.accuracies[i]:10.3f} '\
                f'{self.error_rates[i]:10.3f} {self.precisions[i]:10.3f} '\
                f'{self.recalls[i]:10.3f} {self.specificity[i]:10.3f} '\
                f'{self.fscores[i]:10.3f}\n'
        out += f'\n{"Macro mean":10s} {self.accuracy_M:10.3f} '\
            f'{self.error_rate_M:10.3f} {self.precision_M:10.3f} '\
            f'{self.recall_M:10.3f} {self.specificity_M:10.3f} '\
            f'{self.fscore_M:10.3f}\n'
        out += f'{"Micro mean":10s} {self.accuracy_m:10.3f} '\
            f'{self.error_rate_m:10.3f} {self.precision_m:10.3f} '\
            f'{self.recall_m:10.3f} {self.specificity_m:10.3f} '\
            f'{self.fscore_m:10.3f}\n'
        out += f'{"Accuracy*":10s} {self.accuracy:10.3f}\n'
        return out

    def save_report(self, message='Report'):
        out = f'{"*" * 79}\n**{" " * 26} CLASSIFICATION REPORT '\
            f'{" " * 26}**\n{"*" * 79}\n'
        out += 'Confusion Matrix (row = true, column = predicted):\n'
        out += str(self.cm) + '\n'
        out += '\nStatistics:\n'
        out += f'{"Classes":10s} {"Accuracy":10s} {"Error":10s} '\
            f'{"Precision":10s} {"Recall":10s} {"Specificity":10s} '\
            f'{"F1-score":10s}\n'
        for i in range(self.n):
            out += f'{self.classes[i]:10s} {self.accuracies[i]:10.3f} '\
                f'{self.error_rates[i]:10.3f} {self.precisions[i]:10.3f} '\
                f'{self.recalls[i]:10.3f} {self.specificity[i]:10.3f} '\
                f'{self.fscores[i]:10.3f}\n'
        out += f'\n{"Macro mean":10s} {self.accuracy_M:10.3f} '\
            f'{self.error_rate_M:10.3f} {self.precision_M:10.3f} '\
            f'{self.recall_M:10.3f} {self.specificity_M:10.3f} '\
            f'{self.fscore_M:10.3f}\n'
        out += f'{"Micro mean":10s} {self.accuracy_m:10.3f} '\
            f'{self.error_rate_m:10.3f} {self.precision_m:10.3f} '\
            f'{self.recall_m:10.3f} {self.specificity_m:10.3f} '\
            f'{self.fscore_m:10.3f}\n'
        out += f'{"Accuracy*":10s} {self.accuracy:10.3f}\n'
        time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f'{self.output_dir}/PR_{self.filename_prefix}_'\
                    f'{time_str}.txt', 'w+') as f:
            f.write(out)

    def plot_confusion_matrix(self,
                                normalize: bool=False,
                                title: str='Confusion Matrix',
                                cmap=plt.cm.Blues,
                                cm: List[List[int]]=[]):
        if cm==[]: cm = np.array([[i for i in j] for j in self.cm])
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        data = [[self.classes[i], self.classes[j], cm[i,j]] for i in range(self.num_classes) for j in range(self.num_classes)]
        df = pd.DataFrame(data, columns=['True','Predicted','Amount'])
        df = df.pivot('True','Predicted','Amount')

        plt.figure(figsize=(10,10))
        plt.subplots_adjust(left=0.19, bottom=0.20, right=0.98, top=0.88)
        cm = sns.heatmap(df, annot=True, cmap=plt.cm.Blues, fmt='d')
        cm.set_title('Superfamily Classification')
        cm.set_xticklabels(cm.get_xticklabels(), rotation=45, horizontalalignment='right')
        cm.set_yticklabels(cm.get_yticklabels(), rotation=45, horizontalalignment='right')

        plt.show()
        plt.clf()
        plt.close('all')

    def save_confusion_matrix(self, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        cm = np.array([[i for i in j] for j in self.cm])
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        data = [[self.classes[i], self.classes[j], cm[i,j]] for i in range(self.num_classes) for j in range(self.num_classes)]
        df = pd.DataFrame(data, columns=['True','Predicted','Amount'])
        df = df.pivot('True','Predicted','Amount')

        plt.figure(figsize=(10,10))
        plt.subplots_adjust(left=0.19, bottom=0.20, right=0.98, top=0.88)
        cm = sns.heatmap(df, annot=True, cmap=plt.cm.Blues, fmt='d')
        cm.set_title('Superfamily Classification')
        cm.set_xticklabels(cm.get_xticklabels(), rotation=45, horizontalalignment='right')
        cm.set_yticklabels(cm.get_yticklabels(), rotation=45, horizontalalignment='right')

        plt.savefig(self.output_dir+'/CM_'+self.filename_prefix+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.png')
        plt.clf()
        plt.close('all')

    def plot_learning_curve(self,accuracies,acc=0):
        plt.figure()
        if(acc==0):
            plt.plot([e[0] for e in accuracies],[e[1] for e in accuracies])
        elif(acc==1):
            plt.plot([e[0] for e in accuracies],[e[2] for e in accuracies])
        elif(acc==2):
            plt.plot([e[0] for e in accuracies],[e[3] for e in accuracies])
        plt.show()
        plt.clf()
        plt.close('all')

    def save_learning_curve(self,accuracies,title,acc_type=0):
        acc_type_axis_titles = ['Accuracy(micro)','Accuracy(macro)','Accuracy']
        plt.figure()
        if(acc_type==0):
            plt.plot([e[0] for e in accuracies],[e[1] for e in accuracies])
        elif(acc_type==1):
            plt.plot([e[0] for e in accuracies],[e[2] for e in accuracies])
        elif(acc_type==2):
            plt.plot([e[0] for e in accuracies],[e[3] for e in accuracies])
        plt.title(title)
        plt.xlabel('Epochs')
        plt.ylabel(acc_type_axis_titles[acc_type])
        plt.savefig(self.output_dir+'/LC_'+self.filename_prefix+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.png')
        plt.clf()
        plt.close('all')
