import numpy as np
import matplotlib.pyplot as plt
import itertools

import time
import datetime
from sklearn import metrics as sk_m

class Metric:
    def __init__(self, labels, predictions, classes=[], f_beta=1.0, cm=[], output_dir='./Outputs', filename_prefix=''):
        self.output_dir=output_dir
        self.filename_prefix=filename_prefix
        self.predictions = predictions
        self.f_beta = f_beta
        self.cm = sk_m.confusion_matrix(labels, predictions) if cm == [] else cm

        self.tp = self.cm.diagonal()
        self.fp = sum(self.cm) - self.tp
        self.fn = sum(np.transpose(self.cm)) - self.tp
        self.tn = sum(sum(self.cm)) - (self.tp + self.fp + self.fn)

        self.n = len(self.tp)
        self.classes = classes if classes != [] else ['Class '+str(i) for i in range(self.n)]

        self.sum_tp = sum(self.tp)
        self.sum_fp = sum(self.fp)
        self.sum_fn = sum(self.fn)
        self.sum_tn = sum(self.tn)

        self.accuracies = (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn + 0.000000001)*1.0
        self.accuracy_M = sum(self.accuracies)/self.n*1.0
        self.accuracy_m = (self.sum_tp + self.sum_tn)/(self.sum_tp + self.sum_tn + self.sum_fp + self.sum_fn + 0.000000001)*1.0
        self.accuracy = sum(self.tp)/(sum(sum(self.cm)) + 0.000000001)*1.0

        self.error_rates = (self.fp + self.fn)/(self.tp + self.tn + self.fp + self.fn + 0.000000001)*1.0
        self.error_rate_M = sum(self.error_rates)/self.n*1.0
        self.error_rate_m = (self.sum_fp + self.sum_fn)/(self.sum_tp + self.sum_fp + self.sum_fn + self.sum_tn + 0.000000001)*1.0

        self.precisions = (self.tp)/(self.tp + self.fp + 0.000000001)*1.0
        self.precision_M = sum(self.precisions)/self.n*1.0
        self.precision_m = self.sum_tp/(self.sum_tp + self.sum_fp + 0.000000001)*1.0

        self.recalls = (self.tp)/(self.tp + self.fn + 0.000000001)*1.0
        self.recall_M = sum(self.recalls)/self.n*1.0
        self.recall_m = self.sum_tp/(self.sum_tp + self.sum_fn + 0.000000001)

        self.fscores = (self.f_beta**2 + 1.0)*self.precisions*self.recalls/((self.f_beta**2)*self.precisions + self.recalls + 0.000000001)
        self.fscore_M = (self.f_beta**2 + 1.0)*self.precision_M*self.recall_M/((self.f_beta**2)*self.precision_M + self.recall_M + 0.000000001)
        self.fscore_m = (self.f_beta**2 + 1.0)*self.precision_m*self.recall_m/((self.f_beta**2)*self.precision_m + self.recall_m + 0.000000001)

    def get_report(self):
        out = '*' * 79 + '\n**' + ' ' * 26 + ' CLASSIFICATION REPORT ' + ' ' * 26 + '**\n' + '*' * 79 + '\n'
        out += 'Confusion Matrix (row = true, column = predicted):\n'
        out += str(self.cm) + '\n'
        out += '\nStatistics:\n'
        out += "%10s %10s %10s %10s %10s %10s\n" % ('Classes','Accuracy','Error','Precision','Recall','F1-score')
        for i in range(self.n):
            out += "%10s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (self.classes[i],self.accuracies[i],self.error_rates[i], self.precisions[i], self.recalls[i], self.fscores[i])
        out += "\n%10s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % ('Macro mean',self.accuracy_m,self.error_rate_M,self.precision_M,self.recall_M,self.fscore_M)
        out += "%10s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % ('Micro mean',self.accuracy_m,self.error_rate_m,self.precision_m,self.recall_m,self.fscore_m)
        out += "%10s %10.3f\n" % ('Accuracy*',self.accuracy)
        return out

    def save_report(self, message='Report'):
        out = "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n%60s\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n" % (message)
        out += "Confusion Matrix (row = true, column = predicted):\n"
        out += str(self.cm)+'\n'
        out += "\nStatistics:\n"
        out += "%10s %10s %10s %10s %10s %10s\n" % ('Classes','Accuracy','Error','Precision','Recall','F1-score')
        for i in range(self.n):
            out += "%10s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % (self.classes[i],self.accuracies[i],self.error_rates[i], self.precisions[i], self.recalls[i], self.fscores[i])
        out += "\n%10s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % ('Macro mean',self.accuracy_m,self.error_rate_M,self.precision_M,self.recall_M,self.fscore_M)
        out += "%10s %10.3f %10.3f %10.3f %10.3f %10.3f\n" % ('Micro mean',self.accuracy_m,self.error_rate_m,self.precision_m,self.recall_m,self.fscore_m)
        out += "%10s %10.3f\n" % ('Accuracy*',self.accuracy)
        with open(self.output_dir+'/PR_'+self.filename_prefix+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt', 'w+') as f:
            f.write(out)

    def plot_confusion_matrix(self, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues, cm=[]):
        if cm==[]: cm = np.array([[i for i in j] for j in self.cm])
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)
        plt.ylim(len(self.classes)-0.5, -0.5) 

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max()/2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
        plt.clf()
        plt.close('all')

    def save_confusion_matrix(self, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
        cm = np.array([[i for i in j] for j in self.cm])
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)
        plt.ylim(len(self.classes)-0.55, -0.55) 

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max()/2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
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
