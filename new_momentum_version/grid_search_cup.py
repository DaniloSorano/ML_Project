from Loader import *
from Layer import *
from nncore import *
import time

start_time = time.time()

early = False

l_train = Dataset_Loader('ML-CUP17-TR.csv')
l_train.load_cup_train()
l_train.split_train_get_test(0.2)

n_inputs = len(l_train.x[0])
n_outputs = len(l_train.y[0])
nets = []
topology = -1
folder = 'exp_cup/'
#i = 0

def identity(x): return x
def derivata_identity(x):  return 1

for n_units in [4,7,10]:
    topology = topology + 1
    for tries in range(0,5):
        layers = []
        layers.append(Layer(inputs=n_inputs,sorta=logistic,derivata=derivata_logistic,num_unit=n_units, fun_in=True))
        layers.append(Output_Layer(inputs=n_units,sorta=identity,derivata=derivata_identity,num_unit=n_outputs))
        n = Net(layers,name='Net_'+str(topology)+'_try_'+str(tries))
        pkl.dump(n,open(folder +n.name,'wb')) #+'_'+str(dt.datetime.now()).replace(' ','_').split('.')[0])


for topology in range(0,3):
    teta = -1
    for mode in ['batch','minibatch']:
        for eta in [0.05, 0.2, 0.6]:
            for momentum in [.0, .5, .7]:
                if mode == 'minibatch':
                    for lamb in [0.0, 0.02]:
                        for batch_size in [50, 100]:
                            teta = teta + 1
                            for tries in range(0,5):
                                #validation_accuracy_for_topology=[]
                                n = pkl.load(open(folder + 'Net_'+str(topology)+'_try_'+str(tries),'rb'))
                                print str((n.name.split('_'))),eta,momentum,mode,lamb,(batch_size if mode =='minibatch' else '' ),str(teta)

                                loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta/10., mode=mode, epochs=1, momentum=momentum/10.,batch_size=batch_size,early=early, lamb=lamb, hold_out=0.2)
                                mse, mee, predicted = n.metrics_reg(l_train.x_test,l_train.y_test)
                                #i = i + 1
                                print 'MEE = ', loss[-1],'Validation MEE = ', val_loss[-1], 'Test MEE = ', mee

                                print("--- %s seconds ---" % (time.time() - start_time))

                                n.save_conf_and_score_cup(folder, teta,loss,val_loss,mee)

                                l_train.plot_test_2D(l_train.y_test, predicted, name=folder + n.name + '_' + str(teta) + '_test-' + str(mee).replace('.',','))



                else:
                    for lamb in [0.0, 0.02]:
                        teta = teta + 1
                        for tries in range(0,5):
                            n = pkl.load(open(folder + 'Net_'+str(topology)+'_try_'+str(tries),'rb'))
                            print str((n.name.split('_'))),eta,momentum,mode,lamb,(batch_size if mode =='minibatch' else '' ),str(teta)

                            loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta/10., mode=mode, epochs=1, momentum=momentum/10., lamb=lamb, early=early, hold_out=0.2)
                            mse, mee, predicted = n.metrics_reg(l_train.x_test,l_train.y_test)
                            #i = i + 1
                            print 'MEE = ', loss[-1], 'Validation MEE = ', val_loss[-1], 'Test MEE = ', mee

                            print("--- %s seconds ---" % (time.time() - start_time))
                            n.save_conf_and_score_cup(folder, teta, loss, val_loss, mee)

                            l_train.plot_test_2D(l_train.y_test, predicted,name=folder + n.name + '_' + str(teta) + '_test-' + str(mee).replace('.', ','))

print("--- %s seconds ---" % (time.time() - start_time))
#print i