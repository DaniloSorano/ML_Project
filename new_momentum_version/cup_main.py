import pickle as pkl
from Loader import *
from nncore import *
import time

start_time = time.time()

l_train = Dataset_Loader('ML-CUP17-TR.csv')
l_train.load_cup_train()



for i in [.2]:
    l1 = Layer(inputs=10,weights=[],sorta=np.tanh,derivata=derivata_tanh,num_unit=6)
    l3 = Output_Layer(inputs=6,weights=[],sorta=np.tanh,derivata=derivata_tanh,num_unit=2)
    net_monk = Net([l1,l3])
    a,b,c,d = net_monk.fit(l_train.x, l_train.y, eta=0.7,momentum=0.6, mode='batch', epochs=100,hold_out=i)

    #predicted = []
    #for J,p in enumerate(l_test.x):
     #   hx = net_monk.predict(p)
     #   predicted.append(hx)
    #print ('Test Accuracy = ',net_monk.accuracy(predicted,l_test.y))
    #net_monk.plot_stats(a,b,c,d)

print("--- %s seconds ---" % (time.time() - start_time))
