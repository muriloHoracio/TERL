import numpy as np
class DataHandler:
    def __init__(self, train, test, region_size=30, pool_size=40):
        self.train = train
        self.test = test
        self.classes = [e.split('/')[-1].split('.')[0] for e in self.test]
        self.num_classes = len(self.classes)
        self.max_len = 0
        self.vocabulary = ['A','C','G','T','N',5] # 5 is the background signal for padding
        self.vocab_size = len(self.vocabulary)
        self.x_train, self.y_train = self.get_data(self.train)
        self.x_test, self.y_test = self.get_data(self.test)
        self.train_size = len(self.y_train)
        self.test_size = len(self.y_test)
#        while(self.max_len - region_size + 1)%pool_size != 0:
#            self.max_len += 1
        self.x_train = np.array([np.pad(self.x_train[i],(0,self.max_len-len(self.x_train[i])),'constant',constant_values=(0,0)) for i in range(self.train_size)]) # pads with zero the sequences with length different from max_len
        self.x_test = np.array([np.pad(self.x_test[i],(0,self.max_len-len(self.x_test[i])),'constant',constant_values=(0,0)) for i in range(self.test_size)]) #pads with zero the sequences with length different from max_len
    def get_data(self, files):
        x = []
        y = []
        for fl in files:
            with open(fl,'r') as f:
                cl = self.classes.index(fl.split('/')[-1].split('.')[0])
                for l in f.readlines():
                    if l[0] == '>':
                        if x != []:
                            x[-1] = np.array([1 if c=='A' else 2 if c=='C' else 3 if c=='G' else 4 if c=='T' else 5 for c in x[-1]],dtype=np.uint8)
                            if len(x[-1]) > self.max_len:
                                self.max_len = len(x[-1])
                        x.append('')
                        y.append(cl)
                    else:
                        x[-1] += l.upper().strip()
                x[-1] = np.array([1 if c=='A' else 2 if c=='C' else 3 if c=='G' else 4 if c=='T' else 5 for c in x[-1]],dtype=np.uint8)
                if len(x[-1]) > self.max_len:
                    self.max_len = len(x[-1])
        return np.array([e for e in x]), np.array([e for e in y])
