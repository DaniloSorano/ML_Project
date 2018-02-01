import pickle as pkl
from Loader import *
from nncore import *
import time

start_time = time.time()

l_train = Dataset_Loader('ML-CUP17-TR.csv')
l_train.load_cup_train()
l_train.split_train_get_test(0.1)
#l_train.normalize()
def identity(x): return x
def derivata_identity(x):  return 1
for i in [.2]:
    l1 = Layer(inputs=10,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=20,fun_in=True)
    l3 = Output_Layer(inputs=20,weights=[],sorta=identity,derivata=derivata_identity,num_unit=2)
    net_cup = Net([l1,l3])

    #a,b,c,d = net_cup.fit(l_train.x, l_train.y, eta=0.01,momentum=0.9,lamb=0.00, mode='online', epochs=300,batch_size=100,hold_out=i)
    a,b,c,d = net_cup.fit(l_train.x, l_train.y, eta=0.1,momentum=0.3,lamb=0.00, mode='minibatch', epochs=300,batch_size=100,hold_out=i) # THE BESTT!! 1.1854 MEE 20 Unita
    #a,b,c,d = net_cup.fit(l_train.x, l_train.y, eta=0.6,momentum=0.6, lamb=0.0, mode='batch', epochs=500,hold_out=i) non e' male cosi'
    mse,mee,acc = net_cup.metrics(l_train.x_test,l_train.y_test)
    predicted = []
    for J,p in enumerate(l_train.x_test):
        hx = net_cup.predict(p)
        predicted.append(hx)
    l_train.plot_test_2D(l_train.y_test, predicted)
    print ('Test MEE = ',mee)
    net_cup.plot_reg(a,c)
print("--- %s seconds ---" % (time.time() - start_time))
