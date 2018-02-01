from Loader import *
from Layer import *
from nncore import *
import time

start_time = time.time()

task = "monks-3"
early = False

l_train = Dataset_Loader(task + '.train')
l_train.load_monk1()

l_test = Dataset_Loader(task + '.test')
l_test.load_monk1()

n_inputs = len(l_train.x[0])
n_outputs = len(l_train.y[0])
nets = []
topology = -1
folder = 'myexp_' + task + "/"

for n_units in [5,7,10]:
    topology = topology + 1
    for tries in range(0,5):
        layers = []
        layers.append(Layer(inputs=n_inputs,sorta=logistic,derivata=derivata_logistic,num_unit=n_units))
        layers.append(Output_Layer(inputs=n_units,sorta=logistic,derivata=derivata_logistic,num_unit=n_outputs))
        n = Net(layers,name='Net_'+str(topology)+'_try_'+str(tries))
        pkl.dump(n,open(folder +n.name,'wb')) #+'_'+str(dt.datetime.now()).replace(' ','_').split('.')[0])


for topology in range(0,3):
    teta = -1
    for mode in ['batch','online','minibatch']:
        for eta in [3, 5, 9]:
            for momentum in [3, 7, 8]:
                for lamb in [0.0, 0.01, 0.02]:
                    if mode == 'minibatch':
                        for batch_size in [10,30,50]:
                            teta = teta + 1
                            for tries in range(0,5):
                                #validation_accuracy_for_topology=[]
                                n = pkl.load(open(folder + 'Net_'+str(topology)+'_try_'+str(tries),'rb'))
                                print str((n.name.split('_'))),eta,momentum,mode,lamb,\
                                        (batch_size if mode =='minibatch' else '' ),str(teta)

                                loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta/10.,
                                                mode=mode, epochs=500, momentum=momentum/10.,
                                                batch_size=batch_size, early=early, lamb=lamb, hold_out=0.2)
                                predicted = []
                                for i,p in enumerate(l_test.x):
                                    hx = n.predict(p)
                                    predicted.append(hx)
                                test_acc= n.accuracy(predicted,l_test.y)
                                print 'MSE_val = ', val_loss[-1],'Validation Accuracy = ', val_acc[-1], 'Test Accuracy = ', test_acc
                                n.save_conf_and_score(folder, teta,loss,acc,val_loss,val_acc,test_acc)
                    else:
                        teta = teta + 1
                        for tries in range(0,5):
                            n = pkl.load(open(folder + 'Net_'+str(topology)+'_try_'+str(tries),'rb'))
                            print str((n.name.split('_'))),eta,momentum,mode,lamb,(batch_size if mode =='minibatch' else '' ),str(teta)

                            loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta/10., mode=mode,
                                                            epochs=(100 if mode=='online' else 500), momentum=momentum/10., lamb=lamb,
                                                            early=early, hold_out=0.2)
                            predicted = []
                            for i,p in enumerate(l_test.x):
                                hx = n.predict(p)
                                predicted.append(hx)
                            test_acc= n.accuracy(predicted,l_test.y)
                            print 'MSE_val = ', val_loss[-1],'Validation Accuracy = ', val_acc[-1], 'Test Accuracy = ', test_acc
                            n.save_conf_and_score(folder, teta,loss,acc,val_loss,val_acc,test_acc)
                            
print("--- %s seconds ---" % (time.time() - start_time))
