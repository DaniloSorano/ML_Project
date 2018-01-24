from Loader import *
from Layer import *
from nncore import *
import time

start_time = time.time()

l_train = Dataset_Loader('monks-1.train')
l_train.load_monk1()

l_test = Dataset_Loader('monks-1.test')
l_test.load_monk1()

n_inputs = len(l_train.x[0])
n_outputs = len(l_train.y[0])
nets = []
c= -1
for n_units in [17,13,10]:
    for tries in range(0,5):
        c=c+1
        layers = []
        layers.append(Layer(inputs=n_inputs,sorta=logistic,derivata=derivata_logistic,num_unit=n_units))
        layers.append(Output_Layer(inputs=n_units,sorta=logistic,derivata=derivata_logistic,num_unit=n_outputs))
        n = Net(layers,name='Net_'+str(c))
        pkl.dump(n,open('exp/'+n.name,'wb')) #+'_'+str(dt.datetime.now()).replace(' ','_').split('.')[0])
config = 0
for j in range(0,15):
    config = 0
    for mode in ['batch','online','minibatch']:
        for eta in range(3,8):
            for momentum in range(2,7):
                n = pkl.load(open('exp/Net_'+str(j),'rb'))
                if mode == 'minibatch':
                    for batch_size in [10,30,50,100]:
                        print n.name+' config=',str(config)
                        loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta/10., mode=mode, epochs=500, momentum=momentum/10.,batch_size=batch_size)
                        predicted = []
                        for i,p in enumerate(l_test.x):
                            hx = n.predict(p)
                            predicted.append(hx)
                        test_acc= n.accuracy(predicted,l_test.y)
                        n.save_conf_and_score(config,loss,acc,val_loss,val_acc,test_acc)
                        config=config+1
                else:
                    print n.name+' config=',str(config)
                    loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta/10., mode=mode, epochs=500, momentum=momentum/10.)
                    predicted = []
                    for i,p in enumerate(l_test.x):
                        hx = n.predict(p)
                        predicted.append(hx)
                    test_acc= n.accuracy(predicted,l_test.y)
                    n.save_conf_and_score(config,loss,acc,val_loss,val_acc,test_acc)
                    config=config+1
print("--- %s seconds ---" % (time.time() - start_time))
