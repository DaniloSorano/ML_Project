from Loader import *
from Layer import *
import pickle as pkl
import datetime as dt
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

    def hold_out_generation(self,ratio):
        n_pattern = int(len(self.train_x)*ratio)
        x = []
        y = []
        print len(self.train_x),n_pattern
        for i in range(len(self.train_x)-1,len(self.train_x) - n_pattern-1,-1):
            print i
            x.append(self.train_x.pop(i))
            y.append(self.train_y.pop(i))
        return x,y

    def shuffle_data(self,x,y):
        p=list(x)
        t=list(y)
        self.train_x=[]
        self.train_y=[]
        while p!=[]:
            chose = int(rn.random()*len(p))
            self.train_x.append(p.pop(chose))
            self.train_y.append(t.pop(chose))

    def little_fit(self,x,y,eta,mode='online',batch_size=30,momentum=0,lamb=0):
        x=self.train_x
        y=self.train_y
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
            
            self.divide_gradient(len(self.train_x))
            self.upgrade_layers()
            self.gradients_refresh()

        return mse / len(x), mee/len(x) ,predicted
    def accuracy(self,predicted,y): #TP+TN / N
        s=0
        for i,t in enumerate(y):
            if t[0]==(0. if predicted[i][0] < 0.5 else 1.):
                s =s+1.
            else:
                print (t[0],predicted[i][0])
        
        print (s)
        return s/len(predicted)

    def fit(self, x, y,eta,mode='online',batch_size=30,epochs=100,decay_eta=False,momentum=0.,lamb=0.,hold_out=0.,validation_data=([],[])):
        self.name = 'eta_'+str(eta).replace('.',',')+'_mode_'+mode+('' if momentum==0 else '_momentum_'+str(momentum).replace('.',','))
        self.train_x = x 
        self.train_y = y
        #eta decay
        tau = 100
        eta0=eta
        #for plot
        loss = []
        acc = []

        val_acc = []
        val_loss = []

        #self.shuffle_data(self.train_x,self.train_y)
        
        c=0
        if validation_data==([],[]):
            self.val_x,self.val_y=self.hold_out_generation(hold_out)
        else:
            self.val_x,self.val_y=validation_data
            hold_out = 1.

        for i in range(0,epochs):
            #if not mode=='batch':
            self.shuffle_data(self.train_x,self.train_y) #now in self.train_x there are the patterns, so in self.train_y
            c=c+1 # for the plot
            print ('Eta',eta,'Epoch',i)
            mse,mee,predicted = self.little_fit(self.train_x, self.train_y,eta,mode,batch_size,momentum,lamb=lamb)
            
            acc.append(self.accuracy(predicted,self.train_y))
            loss.append(mse)
            
            if hold_out>0.:
                val_mse, val_mee, val_accuracy=self.metrics(self.val_x,self.val_y)
                val_acc.append(val_accuracy)
                val_loss.append(val_mse)

            print ('MSE', mse,' MEE',mee,' ACC',acc[-1],(('VAL_ACC '+str(val_acc[-1])+' VAL_MSE '+str(val_loss[-1])) if hold_out>0. else ''))
            if decay_eta:
                if eta > (eta0/100):
                    alpha = i/tau
                    eta=(1- (alpha))*eta0 + alpha*(eta/100)

        #            if np.sum(acc[-5:])/float(len(acc[-5:]))==1.:
        #                print('100% Accuracy!')
        #                break

        plt.plot(range(0,c),loss,'r--',range(0,c),acc,'k')
        if hold_out>0.:
            plt.plot(range(0,c),val_acc,'g',range(0,c),val_loss,'y')
        plt.ylabel('Loss/Acc')
        plt.xlabel('ephocs')
            
        plt.savefig(self.name)
        plt.show()
        return acc[-1]

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
    def metrics(self, x, y):
        mee = 0
        mse = 0
        acc = 0
        for i, p in enumerate(x):
            hx = self.predict(p)
            #print ('pattern', i, hx, y[i])
            '''print np.array(y[i]) - hx
            print np.square((np.array(y[i]) - hx))
            print np.sum(np.square((np.array(y[i]) - hx)))'''
            mse = mse + np.sum(.5 * np.square((np.array(y[i]) - hx))) #the error
            mee = mee + np.sqrt(np.sum(np.square((np.array(y[i]) - hx)))) #the error
            if y[i][0]==(0. if hx[0] < 0.5 else 1.):
                acc =acc+1.
            else:
                print y[i][0],hx[0]
        
        return mse / len(x), mee/len(x), acc/len(x)
        
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



#first_layer = Layer(weights = wh1, sorta = logistic, derivata = derivata_logistic)
#out_layer = Output_Layer(weights = wo, sorta=logistic, derivata = derivata_logistic)

#xor_Nand_nn = Net([first_layer, out_layer])


#layer1 = Layer(5,sorta=logistic,derivata=derivata_logistic)
#layer2 = Layer(5,sorta=logistic,derivata=derivata_logistic)
#secondNet = Net([layer1,layer2])



#a = [1, 1]
#d = [0, 0]
#b = [0, 1]
#c = [1, 0]

#targets=[[-1,-1],[-1,1],[1,1],[1,1]]

#a = [.05,.1]
#t=[.01,.99]
#loss=xor_Nand_nn.MSE([a],[t])



#if final_acc==1.:

