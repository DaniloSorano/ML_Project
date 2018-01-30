from Loader import *

cup_train = Dataset_Loader("ML-CUP17-TR.csv")
cup_train.load_cup_train()
cup_train.normalize()


