import numpy as np
import random as rn

def identity(d): return (1 if d>0 else -1)

class Layer():
    def __init__(self, weights, sorta, derivata=identity):
        self.sinapsi = np.array(weights)    #the weights
        self.o = sorta  #the output function of the neruon
        self.derivata_sorta = derivata

    def weighted_sum(self, x):
        x = [1]+x #for the bias
        self.previous_out = x
        #print self.previous_out, 'moltiplicato per'
        #print self.sinapsi,'fa'
        self.net = self.sinapsi.dot(x) # xA
        #print self.net,'applicata la funzione viene'
        #print out
        return self.net

    def output(self, x):
        self.out = [self.o(el) for el in self.weighted_sum(x)]
        return self.out

    def compute_deltas(self, next_layer):
        self.deltas = []
        for i,weights in enumerate(self.sinapsi): #i-th unit
            self.deltas.append(1)
            #print next_layer.deltas
            #print next_layer.sinapsi
            #print 'weights del neurone',i,np.array([sinapsi[1:][i] for sinapsi in next_layer.sinapsi])
            self.deltas[i] = np.array(next_layer.deltas).dot(np.array([sinapsi[1:][i] for sinapsi in next_layer.sinapsi])) * self.derivata_sorta(self.net[i])

        
        
    def compute_gradient(self,c):
        self.gradients = []
        for i, n in enumerate(self.sinapsi):
            self.gradients.append( [1 for el in n] )
            for j, w_n in enumerate(n): #the weigths that go into n from, i
                #i-esimo output del layer precedente
                self.gradients[i][j] = self.deltas[i] * self.previous_out[j] # the output of the i-th previous neuron'''
#            print 'layer',c,i,self.gradients[i],self.previous_out,self.deltas[i],n

    def update_weights(self,eta):
        for j, n in enumerate(self.sinapsi):
            for i, w in enumerate(n):
                self.sinapsi[j][i] = w - eta * self.gradients[j][i]
            

class Output_Layer(Layer):
    def compute_deltas(self, y):
        target = np.array(y)
        self.deltas = [0 for i in target]
        for i in range(len(self.deltas)):
            self.deltas[i] = (target[i]-self.out[i]) * (self.derivata_sorta(self.net[i]))
        
            
class Net():
    def __init__(self, layers):
        self.layers = layers  #the layers of the network

    def predict(self, x): #compute the result of the approximate function
        inpu = x
        for layer in self.layers:   #for each layer it computes the output
            inpu = layer.output(inpu)
        return inpu

    def compute_gradient(self, y):
        self.layers[-1].compute_deltas(y)
        old_layer = self.layers[-1]
        c = 2
        old_layer.compute_gradient(c)
        for layer in self.layers[:-1][::-1]:
            c = c-1
            layer.compute_deltas(old_layer)
            layer.compute_gradient(c)
            '''old_layer=layer
            for i,n in enumerate(layer.sinapsi):
                gradient = [1 for el in n]
                for j, w_n in enumerate(n): #the weigths that go into n from, i
                    #i-esimo output del layer precedente
                    gradient[j] = layer.deltas[i] * layer.previous_out[i] # the output of the i-th previous neuron'''
            
        print ''

    def update_layers(self,eta):
        for l in self.layers:
            l.update_weights(eta)
    #x list inputs, y relative desired outputs
    def MSE(self, x, y): #Mean Square Error
        s = 0
        for i, p in enumerate(x):
            hx = self.predict(p)
            print 'pattern', i, hx, y[i]
            '''print np.array(y[i]) - hx
            print np.square((np.array(y[i]) - hx))
            print np.sum(np.square((np.array(y[i]) - hx)))'''
            s = s + np.sum(np.square((np.array(y[i]) - hx))) #the error
            
            self.compute_gradient(y[i])
            self.update_layers(0.05)
        print 'MSE', s / len(x)
        return s / len(x)
    def MEE(self, x, y): #Mean Eucludian Error
        s = 0
        for i, p in enumerate(x):
            hx = self.predict(p)
            s = s + np.sqrt(np.sum(np.square((np.array(y[i]) - hx)))) #the error
        print 'MSE', s / len(x)
        return s / len(x)

   



#pesi corretti
wh1 = [[-1.5, 1, 1], [-0.5, 1, 1]]
wo = [[-0.5, -1, 1], [0, -1, 0]]

#pesi random
wh1 = [[rn.random()*4 -2 for i in range(0,3)]for el in [1,2]]
wh2 = [[rn.random()*4 -2 for i in range(0,3)]for el in [1,2]]
wo = [[rn.random()*4 -2 for i in range(0,3)]for el in [1,2,3]]
#wh1 = [[1, 2, 1], [-1, -1.5, -10]]
#wo = [[-1, -1, 0], [0, -1, 2]]


first_layer = Layer(wh1, np.sign)
middle_layer = Layer(wh2,np.sign)
out_layer = Output_Layer(wo, np.sign)


xor_Nand_nn = Net([first_layer,middle_layer, out_layer])



a = [1, 1]
d = [0, 0]
b = [0, 1]
c = [1, 0]


'''print '1 XOR 1'
print xor_Nand_nn.predict(a)
print '0 XOR 0'
print xor_Nand_nn.predict(d)
print '0 XOR 1'
print xor_Nand_nn.predict(b)
print '1 XOR 0'
print xor_Nand_nn.predict(c)
'''

targets=[[-1,-1,1],[-1,1,1],[1,1,-1],[1,1,1]]
loss = xor_Nand_nn.MSE([a,d,b,c],targets)
cout=1
while loss >2:
     loss=xor_Nand_nn.MSE([a,d,b,c],targets)
     if cout%20==0:
        for l in xor_Nand_nn.layers:
            print l.sinapsi
        cana= raw_input()
     cout = cout +1
loss=xor_Nand_nn.MSE([a,d,b,c],targets)
#out_layer = Layer(wo.append([0,-1,0]),np.sign) #aggiungo un neurone sullo stesso layer

