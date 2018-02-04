import pickle as pkl
from Loader import *
from nncore import *
import time

l_test = Dataset_Loader('ML-CUP17-TS.csv')
l_test.load_blind()
folder = 'consegna/'
topology = 0
tries = 0

def identity(x): return x
def derivata_identity(x):  return 1

n = pkl.load(open(folder + 'Net_'+str(topology)+'_try_'+str(tries),'rb'))

out = []
for p in l_test.x:
    out.append(n.predict(p))


l_test.save_blind(out)