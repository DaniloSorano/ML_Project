import pickle as pkl
from Loader import *
from nncore import *

net_monk = pkl.load(open('net_monk','rb'))

print ('loaded')
l_test = Dataset_Loader('monks-1.test')
l_test.load_monk1()

predicted=[]

for i,p in enumerate(l_test.x):
    hx = net_monk.predict(p)
    predicted.append(hx)
print ('Test Accuracy = ',net_monk.accuracy(predicted,l_test.y))