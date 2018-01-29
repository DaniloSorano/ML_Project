import numpy as np
import random as rn

def derivata_segno(d): return (1 if d>0 else -1)

class Layer():
    def __init__(self, inputs=4, weights=[], sorta=np.sign, derivata=derivata_segno, num_unit=2,fun_in=False):
        if weights==[]:
            weights = [[(rn.random()*1.4 -.7) * (2./inputs if fun_in else 1.) for i in range(0, inputs+1)] for j in range(0, num_unit)]
        #print (inputs,'inputs')
        #print (num_unit,'neurons')
        self.sinapsi = np.array(weights)    #the weights
        self.o = sorta  #the output function of the neruon
        self.derivata_sorta = derivata
        self.gradients_refresh()
        self.momentums_refresh()
    def momentums_refresh(self):
        self.momentum = []
        for i, n in enumerate(self.sinapsi):
            self.momentum.append([0 for el in n])
    def gradients_refresh(self):
        self.gradients = []
        for i, n in enumerate(self.sinapsi):
            self.gradients.append([0 for el in n])

    def divide_gradient(self,val=.1):
            for i, n in enumerate(self.sinapsi):
                self.gradients[i] = list(np.array(self.gradients[i]).dot(val))

    def norm(self):
        norm = 0
        for neurone in self.sinapsi:
            norm = norm + np.sum(np.square(neurone))
        return norm
    def weighted_sum(self, x):
        x = [1]+x #for the bias

        self.previous_out = x
        #   print self.previous_out, 'moltiplicato per'
        #print self.sinapsi, 'fa'

        self.net = self.sinapsi.dot(x) # xA

        #print self.net, 'applicata la funzione viene'
        return self.net

    def output(self, x):
        self.out = [self.o(el) for el in self.weighted_sum(x)]
        #print self.out
        return self.out

    def compute_deltas(self, next_layer):
        self.deltas = []
        for i, weights in enumerate(self.sinapsi): #i-th unit
            self.deltas.append(1)
            #print next_layer.deltas
            #print next_layer.sinapsi
            #print 'weights del neurone',i,np.array([sinapsi[1:][i] for sinapsi in next_layer.sinapsi])


            self.deltas[i] = np.array(next_layer.deltas).dot(np.array([sinapsi[1:][i] for sinapsi in next_layer.sinapsi])) * self.derivata_sorta(self.net[i])



    def compute_gradient(self):
        for i, n in enumerate(self.sinapsi):
            for j, w_n in enumerate(n): #the weigths that go into n from, i
                #i-esimo output del layer precedente
                self.gradients[i][j] = self.gradients[i][j] + \
                                     self.deltas[i] * self.previous_out[j]

            #print 'layer',c,i,n,self.gradients[i],self.deltas[i]

    def upgrade_weights(self,lamb=0.,eta=0.5,momentum=.0):
        for j, n in enumerate(self.sinapsi):
            for i, w in enumerate(n):
                Cambiamento = eta * self.gradients[j][i] - (lamb*w) + self.momentum[j][i]
                self.momentum[j][i] = momentum * Cambiamento  #momentum for the next Cambiamento
                self.sinapsi[j][i] = w + Cambiamento #  if i!=0 else 0)
                #print self.sinapsi[j][i] ,
            #print ''

class Output_Layer(Layer):
    def __init__(self, inputs=4, weights=[], sorta=np.sign, derivata=derivata_segno, num_unit=2):
        if weights==[]:
            #no fun in
            weights = [[(rn.random()*1.4 -.7) for i in range(0, inputs+1)] for j in range(0, num_unit)]
        #print (inputs,'inputs')
        #print (num_unit,'neurons')
        self.sinapsi = np.array(weights)    #the weights
        self.o = sorta  #the output function of the neruon
        self.derivata_sorta = derivata
        self.gradients_refresh()
        self.momentums_refresh()
    def compute_deltas(self, y, norm=0):
        target = np.array(y)

        self.deltas = [0 for i in target]
        for i in range(len(self.deltas)):
            self.deltas[i] = (target[i]-self.out[i]) * (self.derivata_sorta(self.net[i]))
