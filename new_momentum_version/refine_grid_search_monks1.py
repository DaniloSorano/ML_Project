from Loader import *
from Layer import *
from nncore import *
import time


task = 'monks-1'
l_train = Dataset_Loader(task + '.train')
l_train.load_monk1()

l_test = Dataset_Loader(task + '.test')
l_test.load_monk1()

n_inputs = len(l_train.x[0])
n_outputs = len(l_train.y[0])
folder = 'fine_' + task + '/'
#start fine tuning

nets = []

topology=1
n_units=7
for tries in range(0,20):
    layers = []
    layers.append(Layer(inputs=n_inputs,sorta=logistic,derivata=derivata_logistic,num_unit=n_units))
    layers.append(Output_Layer(inputs=n_units,sorta=logistic,derivata=derivata_logistic,num_unit=n_outputs))
    n = Net(layers,name='Net_'+str(topology)+'_try_'+str(tries))
    pkl.dump(n,open(folder +n.name,'wb')) #+'_'+str(dt.datetime.now()).replace(' ','_').split('.')[0])


start_time = time.time()
teta = -1
for mode in ['online']:
    for eta in [.85, .9, .95]:
        for momentum in [.65, .7, .75]:
            teta = teta + 1
            for tries in range(0,20):
                n = pkl.load(open(folder + 'Net_'+str(topology)+'_try_'+str(tries),'rb'))
                print str((n.name.split('_'))),eta,momentum,mode,str(teta)

                loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta,mode=mode, epochs=100, momentum=momentum, hold_out=0.2)
                predicted = []

                for i,p in enumerate(l_test.x):
                    hx = n.predict(p)
                    predicted.append(hx)
                test_acc= n.accuracy(predicted,l_test.y)

                print 'MSE_val = ', val_loss[-1],'Validation Accuracy = ', val_acc[-1], 'Test Accuracy = ', test_acc

                n.save_conf_and_score(folder, teta,loss,acc,val_loss,val_acc,test_acc)



topology=0
n_units=5
for tries in range(0,20):
    layers = []
    layers.append(Layer(inputs=n_inputs,sorta=logistic,derivata=derivata_logistic,num_unit=n_units))
    layers.append(Output_Layer(inputs=n_units,sorta=logistic,derivata=derivata_logistic,num_unit=n_outputs))
    n = Net(layers,name='Net_'+str(topology)+'_try_'+str(tries))
    pkl.dump(n,open(folder +n.name,'wb')) #+'_'+str(dt.datetime.now()).replace(' ','_').split('.')[0])


start_time = time.time()
teta = -1
for mode in ['online']:
    for eta in [.25, .3, .35]:
        for momentum in [.25, .3, .35]:
            teta = teta + 1
            for tries in range(0,20):
                n = pkl.load(open(folder + 'Net_'+str(topology)+'_try_'+str(tries),'rb'))
                print str((n.name.split('_'))),eta,momentum,mode,str(teta)

                loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta,mode=mode, epochs=200, momentum=momentum, hold_out=0.2)
                predicted = []

                for i,p in enumerate(l_test.x):
                    hx = n.predict(p)
                    predicted.append(hx)
                test_acc= n.accuracy(predicted,l_test.y)

                print 'MSE_val = ', val_loss[-1],'Validation Accuracy = ', val_acc[-1], 'Test Accuracy = ', test_acc

                n.save_conf_and_score(folder, teta,loss,acc,val_loss,val_acc,test_acc)

print("--- %s seconds ---" % (time.time() - start_time))