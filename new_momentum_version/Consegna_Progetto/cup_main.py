import pickle as pkl
from Loader import *
from nncore import *
import time

start_time = time.time()

topology = 20

l_train = Dataset_Loader('ML-CUP17-TR.csv')
l_train.load_cup_train()
#l_train.split_train_get_test(0.2)
#l_train.normalize()
def identity(x): return x 
def derivata_identity(x):  return 1.

for i in [.2]:
    l1 = Layer(inputs=10,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=20,fun_in=True)
    l3 = Output_Layer(inputs=20,weights=[],sorta=identity,derivata=derivata_identity,num_unit=2)
    net_cup = Net([l1,l3])

    
    a,b,c,d = net_cup.fit(l_train.x, l_train.y, eta=0.15,momentum=0.,lamb=0.0, mode='minibatch', epochs=300,batch_size=100,hold_out=i)
    #mse,mee,predicted = net_cup.metrics_reg(l_train.x_test,l_train.y_test)
    print ('Val MEE = ',a[-1])
    #l_train.plot_test_2D(l_train.y_test, predicted,'My_Net_'+str(topology)+'_test-'+str(mee).replace('.',','))
    net_cup.plot_reg(a,c)
print("--- %s seconds ---" % (time.time() - start_time))





