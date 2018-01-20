import pickle as pkl
from Loader import *
from nncore import *
l_train = Dataset_Loader('monks-3.train')
l_train.load_monk1()

l_test = Dataset_Loader('monks-3.test')
l_test.load_monk1()
l1 = Layer(inputs=17,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=17)
#l2 = Layer(inputs=17,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=7)
l3 = Output_Layer(inputs=17,weights=[],sorta=logistic,derivata=derivata_logistic,num_unit=1)

net_monk = Net([l1,l3])

final_acc = net_monk.fit(l_train.x,l_train.y,eta=0.5,mode='batch',batch_size=50,epochs=250,momentum=0.7,lamb = 0.01) #validation_data=(l_train.x,l_train.y))

if final_acc==1.:
    pkl.dump(net_monk,open('net_monk_3','wb')) #+'_'+str(dt.datetime.now()).replace(' ','_').split('.')[0])

predicted = []
for i,p in enumerate(l_test.x):
    hx = net_monk.predict(p)
    predicted.append(hx)
print ('Test Accuracy = ',net_monk.accuracy(predicted,l_test.y))