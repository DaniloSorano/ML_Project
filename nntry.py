from Loader import *
from Layer import *
import matplotlib.pyplot as plt

            
class Net():
    def __init__(self, layers):
        self.layers = layers  #the layers of the network

    def predict(self, x): #compute the result of the approximate function
        inpu = x
        for layer in self.layers:   #for each layer it computes the output
            inpu = layer.output(inpu)
        return inpu

    def gradients_refresh(self):
        for layer in self.layers: 
            layer.gradients_refresh()
    def divide_gradient(self, val):
        for layer in self.layers: 
            layer.divide_gradient(1./val)

    def shuffle_data(self,x,y):
        self.x=[]
        self.y=[]
        p=list(x)
        t=list(y)
        while p!=[]:
            chose = int(rn.random()*len(p))
            self.x.append(p.pop(chose))
            self.y.append(t.pop(chose))

    def little_fit(self,x,y,eta,mode='online',batch_size=30,momentum=0):        
        x=self.x
        y=self.y
        predicted = []
        mse = 0
        mee = 0

        for i, p in enumerate(x):
            hx = self.predict(p)
            predicted.append(hx)
            mse = mse + np.sum(.5 * np.square((np.array(y[i]) - hx))) #the error
            mee = mee + np.sqrt(np.sum(np.square((np.array(y[i]) - hx))))

            self.compute_gradient(y[i],eta,momentum)

            if mode == 'online':
                self.upgrade_layers()
                self.gradients_refresh()
            if mode == 'minibatch' and (((i+1) % batch_size) == 0 or (i==len(x)-1)):
                self.divide_gradient(batch_size)
                self.upgrade_layers()
                self.gradients_refresh()
        if mode == 'batch':
            
            self.divide_gradient(len(self.x))
            self.upgrade_layers()
            self.gradients_refresh()

        return mse / len(x), mee/len(x) ,predicted
    def accuracy(self,predicted,y): #TP+TN / N
        s=0
        for i,t in enumerate(y):
            if t[0]==(0. if predicted[i][0] < 0.5 else 1.):
                s =s+1.
            else:
                print (t[0],(0. if predicted[i][0] < 0.5 else 1.))
        
        print (s)
        return s/len(predicted)

    def fit(self, x, y,eta,mode='online',batch_size=30,epochs=100,decay_eta=True,momentum=0):
        self.shuffle_data(x,y)
        tau = 100
        eta0=eta
        loss = []
        acc = []
        c=0
        for i in range(0,epochs):
            c=c+1 # for the plot
            print ('Eta',eta,'Epoch',i)
            mse,mee,predicted = self.little_fit(self.x, self.y,eta,mode,batch_size,momentum)

            acc.append(self.accuracy(predicted,self.y))
            loss.append(mse)

            print ('MSE', mse,' MEE',mee,' ACC',acc[-1])
            if decay_eta:
                if eta > (eta0/100):
                    alpha = i/tau
                    eta=(1- (alpha))*eta0 + alpha*(eta/100)
            if acc[-1]==1.:
                print('100% Accuracy!')
                break
        plt.plot(range(0,c),loss,'r--',range(0,c),acc,'bs')
        plt.ylabel('Loss/Acc')
        plt.xlabel('ephocs')
            
        plt.savefig('eta_'+str(eta).replace('.',',')+'_mode_'+mode)
        plt.show()

    def compute_gradient(self, y,eta=0.5,momentum=0):
        self.layers[-1].compute_deltas(y)
        old_layer = self.layers[-1]
        c = len(self.layers)
        old_layer.compute_gradient(c,eta,momentum)
        for layer in self.layers[:-1][::-1]:
            c = c-1
            layer.compute_deltas(old_layer)
            layer.compute_gradient(c,eta,momentum)
            old_layer = layer
        

    def upgrade_layers(self):
        for l in self.layers:
            l.upgrade_weights()
            
    #x list inputs, y relative desired outputs
    def MSE(self, x, y): #Mean Square Error
        s = 0
        for i, p in enumerate(x):
            hx = self.predict(p)
            print ('pattern', i, hx, y[i])
            '''print np.array(y[i]) - hx
            print np.square((np.array(y[i]) - hx))
            print np.sum(np.square((np.array(y[i]) - hx)))'''
            s = s + np.sum(.5 * np.square((np.array(y[i]) - hx))) #the error

        print ('MSE', s / len(x))
        return s / len(x)
        
    def MEE(self, x, y): #Mean Eucludian Error
        s = 0
        for i, p in enumerate(x):
            hx = self.predict(p)
            s = s + np.sqrt(np.sum(np.square((np.array(y[i]) - hx)))) #the error
        print ('MEE', s / len(x))
        return s / len(x)

   




#pesi corretti
wh1 = [[-1.5, 1, 1], [-0.5, 1, 1]]
wo = [[-0.5, -1, 1], [0, -1, 0]]

#pesi nel range
#weights = [[rn.random()*1.4 -.7 for i in range(0, num_inputs+1)] for j in range(0, num_hidden_units)]
#pesi fun in
#weights = [[ (rn.random()*1.4 -.7)*(2/(num_inputs+1)) for i in range(0, num_inputs+1)] for j in range(0, num_hidden_units)]

#pesi random
#wh1 = [[rn.random()*.1 -.05 for i in range(0, 3)]for el in [1, 2]]
#wh2 = [[rn.random()*.1 -.05 for i in range(0, 3)]for el in [1, 2]]
#wo = [[rn.random()*.1 -.05 for i in range(0, 3)]for el in [1, 2]]
wh1 = [[.35, .15, .20], [.35, .25, .3]]
wo = [[.6, .4, .45], [.6, .5, .55]]


def derivata_tanh(x) : return np.tanh(x)*(1-np.tanh(x))
def logistic(x): return 1/(1+np.exp(-x))
def derivata_logistic(x): return logistic(x)*(1-logistic(x))



first_layer = Layer(weights = wh1, sorta = logistic, derivata = derivata_logistic)
out_layer = Output_Layer(weights = wo, sorta=logistic, derivata = derivata_logistic)

xor_Nand_nn = Net([first_layer, out_layer])


#layer1 = Layer(5,sorta=logistic,derivata=derivata_logistic)
#layer2 = Layer(5,sorta=logistic,derivata=derivata_logistic)
#secondNet = Net([layer1,layer2])



a = [1, 1]
d = [0, 0]
b = [0, 1]
c = [1, 0]

targets=[[-1,-1],[-1,1],[1,1],[1,1]]

a = [.05,.1]
t=[.01,.99]
loss=xor_Nand_nn.MSE([a],[t])


l = Dataset_Loader('monks-1.train')
l.load_monk1()


l1 = Layer(inputs=17,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=17)
#l2 = Layer(inputs=17,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=7)
l3 = Output_Layer(inputs=17,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=1)

net_monk = Net([l1,l3])
net_monk.fit(l.x,l.y,eta=0.3,mode='batch',batch_size=50,epochs=500,decay_eta=False,momentum=0.7)


'''#loss = xor_Nand_nn.MSE([a,d,b,c],targets)
cout=1
while loss >0.5:
     if cout%20==0:
        for l in xor_Nand_nn.layers:
            print l.sinapsi
        cana= raw_input()
     cout = cout +1
loss=xor_Nand_nn.MSE([a,d,b,c],targets)
#out_layer = Layer(wo.append([0,-1,0]),np.sign) #aggiungo un neurone sullo stesso layer

'''
