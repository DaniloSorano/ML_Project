from Loader import *
from Layer import *
from nncore import *
import time


task = 'consegna'
l_train = Dataset_Loader('ML-CUP17-TR.csv')
l_train.load_cup_train()

n_inputs = len(l_train.x[0])
n_outputs = len(l_train.y[0])
folder = '' + task + '/'
#start fine tuning
nome='noMomentum_cup_20_batch'
nets = []

def identity(x): return x
def derivata_identity(x):  return 1

topology = 0
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
    for eta in [.15]:
        for momentum in [0.3]:
            for batch_size in [128]:
                teta = teta + 1
                for tries in range(0,5):
                    n = pkl.load(open(folder + 'Net_'+str(topology)+'_try_'+str(tries),'rb'))
                    print str((n.name.split('_'))),eta,momentum,mode,str(teta)

                    loss,acc,val_loss,val_acc=n.fit(l_train.x, l_train.y, eta=eta,mode=mode, batch_size=batch_size,lamb=0.0, epochs=300, momentum=momentum)
                    #mse,mee,predicted = n.metrics_reg(l_train.x_test,l_train.y_test)
                    #print 'MEE = ', loss[-1],('val_MEE = '+ str(val_loss[-1]) if val_loss else ''),'Test Mee',mee
                    #n.save_conf_and_score_cup(folder, teta,loss,val_loss,mee)#(val_loss[-1] if val_loss else []))
                    #l_train.plot_test_2D(l_train.y_test,predicted,folder + n.name +'_Test-'+str(mee).replace('.',','))
                    pkl.dump(n,open(n.name,'wb')) #+'_'+str(dt.datetime.now()).replace(' ','_').split('.')[0])
print("--- %s seconds ---" % (time.time() - start_time))