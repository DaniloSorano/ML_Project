import random as rn
from sklearn.preprocessing import *
import matplotlib.pyplot as plt
import numpy as np
class Dataset_Loader():
    def __init__(self, filename):
        self.name=filename
        print( self.name)
    def load_monk1(self):
        self.x = []
        self.y = []
        f = open(self.name, 'r')
        for line in f:
            line = line.strip()
            elements = line.split()

            self.y.append([float(elements[0])]) 
            #print elements[0]
            elements = elements[1:-1]
            feature = [[0 for i in range(0, n_classes)] for n_classes in [3, 3, 2, 3, 4, 2]]
            for i, n_classes in enumerate([3, 3, 2, 3, 4, 2]):
                feature[i][int(elements[i])-1] = 1
            final_features = []
            for v in feature:
                for el in v:
                    final_features.append(el)
            self.x.append(final_features)
        f.close()
    def load_cup_train(self):
        self.x = []
        self.y = []
        f = open(self.name, 'r')
        train = []
        for line in f:
            line = line.strip()
            if line and line[0]!='#':
                elements = line.split(',')
                #for element in elements:
                train.append([float(el) for el in elements[1:]])
                self.y.append([float(el) for el in elements[-2:]])
                #print elements[0]
                elements = [float(el) for el in elements[1:-2]]

                self.x.append(elements)
        print len(self.x[0])
        f.close()
        #print max(train)

    def save_blind(self,predicted):
        f = open('networks_benders_ML-CUP17-TS.csv', 'w')
        nomi = '# Valerio Bonsignori, Danilo Sorano\n'
        nick = '# networks_benders\n'
        task = '# ML-CUP17\n'
        data = '# 3/2/2018\n'
        f.write(nomi)
        f.write(nick)
        f.write(task)
        f.write(data)
        for i,z_pred in enumerate(predicted):
            s = str(i+1)+','
            for el in z_pred:
                s=s+str(el)+','
            print str(s[:-1])
            f.write(s[:-1]+'\n')
        f.close()
        #print max(train)

    def load_blind(self):
        self.x = []
        self.y = []
        f = open(self.name, 'r')
        train = []
        for line in f:
            line = line.strip()
            if line and line[0]!='#':
                elements = line.split(',')
                #for element in elements:
                #print elements[0]
                elements = [float(el) for el in elements[1:]]
                print elements
                self.x.append(elements)
        print len(self.x[0])
        f.close()
        #print max(train)

    def normalize(self):
        min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
        attribute = min_max_scaler.fit_transform(self.x)
        print attribute[0]
        target = min_max_scaler.fit_transform(self.y)
        print target[0]
        self.x = []
        self.y = []
        for vector_a in attribute:
            self.x.append([float(el) for el in vector_a])
        for vector_t in target:
            self.y.append([float(el) for el in vector_t])
        #print self.x
        #print self.y

    def split_train_get_test(self, percentuage = 0.2):
        n_test_pattern = int(len(self.x)*percentuage)
        self.x_test = []
        self.y_test = []
        for i in range(0,n_test_pattern):
            chose = rn.randrange(0,len(self.x))
            self.x_test.append(self.x.pop(chose))
            self.y_test.append(self.y.pop(chose))
    def plot_test_2D(self,z1,z2,name=''):
        plt.clf()
        plt.figure(figsize=(20,10))
        pltname = 'Test 2D'
        x1 =[ el[0] for el in z1 ]
        y1 =[ el[1] for el in z1 ]
        x2 =[ el[0] for el in z2 ]
        y2 =[ el[1] for el in z2 ]
        plt.plot(x1,y1,'^',label='Target')
        plt.plot(x2,y2,'s',label='Output')
        plt.legend(fontsize=20,loc=9)
        if name == '':
            plt.show()
        else:
            plt.savefig(name)
            #plt.close()
    def rescale(self, range=[0.,1.]):
        a = [[] for el in self.x[0]]
        for i,v in enumerate(self.x[0]):
            for p in self.x:
                a[i].append(p[i])
        maxmins = [{'max':max(el),'min':min(el)} for el in a]        
        for p in self.x:
            for i,v in enumerate(p):
                p[i]= (p[i]-maxmins[i]['min'])/(maxmins[i]['min']-maxmins[i]['max']) * range[1]
    def standardize(self):
        a = [[] for el in self.x[0]]
        for i,v in enumerate(self.x[0]):
            for p in self.x:
                a[i].append(p[i])
        means_standards = [{'mean':np.array(el).mean(),'std':np.array(el).std()} for el in a]
        print means_standards
        for p in self.x:
            for i,v in enumerate(p):
                p[i]= (p[i]-means_standards[i]['mean'])/(means_standards[i]['std'])
        
