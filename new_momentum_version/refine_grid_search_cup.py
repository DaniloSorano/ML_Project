from Loader import *
from Layer import *
from nncore import *
import time


task = 'cup_20'
l_train = Dataset_Loader('ML-CUP17-TR.csv')
l_train.load_cup_train()

n_inputs = len(l_train.x[0])
n_outputs = len(l_train.y[0])
folder = 'fine_' + task + '/'
#start fine tuning

nets = []

def identity(x): return x
def derivata_identity(x):  return 1

topology = 1
n_units = 20
for tries in range(0,5):
    layers = []
    layers.append(Layer(inputs=n_inputs,sorta=logistic,derivata=derivata_logistic,num_unit=n_units))
    layers.append(Output_Layer(inputs=n_units,sorta=identity,derivata=derivata_identity,num_unit=n_outputs))
    n = Net(layers,name='Net_'+str(topology)+'_try_'+str(tries))
    pkl.dump(n,open(folder +n.name,'wb')) #+'_'+str(dt.datetime.now()).replace(' ','_').split('.')[0])


start_time = time.time()
teta = -1
for mode in ['minibatch']:
    for eta in [.05, .1, .15]:
        for momentum in [.3, .7]:
            for batch_size in [100, 128]:
                teta = teta + 1
                for tries in range(0,5):
                    n = pkl.load(open(folder + 'Net_'+str(topology)+'_try_'+str(tries),'rb'))
                    print str((n.name.split('_'))),eta,momentum,mode,str(teta)

                    loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta,mode=mode, batch_size=batch_size, epochs=300, momentum=momentum, hold_out=0.2)

                    print 'MEE = ', loss[-1],'val_MEE = ', val_loss[-1]

                    n.save_conf_and_score_cup(folder, teta,loss,val_loss,val_loss[-1])

print("--- %s seconds ---" % (time.time() - start_time))