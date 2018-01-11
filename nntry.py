import numpy as np

class layer():
    def __init__(self, weights, sorta):
        self.sinapsi = np.array(weights)    #the weights
        self.o = sorta  #the output function of the neruon

    def product(self, x):
        out = []
        x = [1]+x #for the bias
        out = self.sinapsi.dot(x) # xA
        #print out
        return [self.o(el) for el in out]   #output vector
    
class net():
    def __init__(self, layers):
        layers
        self.layers = layers  #the layers of the network

    def predict(self,x): #compute the result of the approximate function
        inpu = x
        for layer in self.layers:   #for each layer it computes the output
            inpu = layer.product(inpu)
        return inpu
#x list inputs, y relative desired outputs
    def LMS(self,x,y):  #Least Mean Square
        s = 0
        for i,p in enumerate(x):
            hx = self.predict(p)
            print 'pattern',i,hx,y[i]   
            s = s + np.sum(0.5 * np.square((np.array(y[i]) - hx ))) #the error
        print 'LMS',s
        return s

    def MSE(self,x,y): #Mean Square Error
        s = 0
        for i,p in enumerate(x):
            hx = self.predict(p)
            '''print 'pattern',i,hx,y[i]
            print np.array(y[i]) - hx
            print np.square((np.array(y[i]) - hx))
            print np.sum(np.square((np.array(y[i]) - hx)))'''
            s = s + np.sum(np.square((np.array(y[i]) - hx ))) #the error
        print 'MSE', s / len(x)
        return s / len(x)
    def MEE(self, x, y): #Mean Eucludian Error
        s = 0
        for i,p in enumerate(x):
            hx = self.predict(p)
            s = s + np.sqrt(np.sum(np.square((np.array(y[i]) - hx )))) #the error
        print 'MSE', s / len(x)
        return s / len(x)
    #def littletrain(self,x,y,eta):  #gradient discent algorithm
    #    self.eta = eta
    #    error = self.LMS(x,y)
    #    for l in layers[::-1]:
    #        l.gradient_discent(error)
        
        

wh1 = [[-1.5, 1, 1],[-0.5, 1, 1]]
wo = [[-0.5, -1, 1],[0,-1,0]]

first_layer = layer(wh1,np.sign)
out_layer = layer(wo,np.sign)


xor_Nand_nn = net([first_layer, out_layer])



a = [1, 1]
d = [0, 0]
b = [0, 1]
c = [1, 0]


print '1 XOR 1'
print xor_Nand_nn.predict(a)
print '0 XOR 0'
print xor_Nand_nn.predict(d)
print '0 XOR 1'
print xor_Nand_nn.predict(b)
print '1 XOR 0'
print xor_Nand_nn.predict(c)
xor_Nand_nn.LMS([d,a,c],[[1,-1],[1,-1],[-1,-1]])
xor_Nand_nn.MSE([d,a,c],[[1,-1],[1,-1],[-1,-1]])
xor_Nand_nn.MEE([d,a,c],[[1,-1],[1,-1],[-1,-1]])
xor_Nand_nn.LMS([a,d,b,c],[[1,1],[1,0],[-1,1],[1,1]])
xor_Nand_nn.MSE([a,d,b,c],[[1,1],[1,0],[-1,1],[1,1]])
xor_Nand_nn.MEE([a,d,b,c],[[1,1],[1,0],[-1,1],[1,1]])


out_layer =layer(wo.append([0,-1,0]),np.sign) #aggiungo un neurone sullo stesso layer

