import pickle as pkl
from Loader import *
from nncore import *
import time

start_time = time.time()

l_train = Dataset_Loader('ML-CUP17-TR.csv')
l_train.load_cup_train()
#l_train.normalize()
def identity(x): return x
def derivata_identity(x):  return 1
for i in [.2]:
    l1 = Layer(inputs=10,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=4)
    l3 = Output_Layer(inputs=4,weights=[],sorta=identity,derivata=derivata_identity,num_unit=2)
    net_cup = Net([l1,l3])

    a,b,c,d = net_cup.fit(l_train.x, l_train.y, eta=0.3,momentum=0.6, lamb=0.0, mode='online', epochs=300,hold_out=i)

    #predicted = []
    #for J,p in enumerate(l_test.x):
     #   hx = net_monk.predict(p)
     #   predicted.append(hx)
    #print ('Test Accuracy = ',net_monk.accuracy(predicted,l_test.y))
    net_cup.plot_reg(a,c)

print("--- %s seconds ---" % (time.time() - start_time))
