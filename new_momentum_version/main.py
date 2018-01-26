import pickle as pkl
from Loader import *
from nncore import *
import time

start_time = time.time()

l_train = Dataset_Loader('monks-1.train')
l_train.load_monk1()

l_test = Dataset_Loader('monks-1.test')
l_test.load_monk1()

for i in [.2]:
    l1 = Layer(inputs=17,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=17)
    #l2 = Layer(inputs=17,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=7)
    l3 = Output_Layer(inputs=17,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=1)
    net_monk = Net([l1,l3])
    a,b,c,d = net_monk.fit(l_train.x, l_train.y, eta=0.3, momentum=0.8, mode='online', batch_size=10, epochs=500,hold_out=i)

    predicted = []
    for J,p in enumerate(l_test.x):
        hx = net_monk.predict(p)
        predicted.append(hx)
    print ('Test Accuracy = ',net_monk.accuracy(predicted,l_test.y))
    net_monk.plot_stats(a,b,c,d)
#if final_acc==1.:
#    pkl.dump(net_monk,open('net_monk_2','wb')) #+'_'+str(dt.datetime.now()).replace(' ','_').split('.')[0])



print("--- %s seconds ---" % (time.time() - start_time))
