
import keras
import numpy as np
from keras.models import Sequential,load_model
from keras.layers.core import Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import *
from Loader import *
from keras import regularizers


def mee(y_true, y_pred):
    return np.sqrt(np.sum(np.square(np.array(y_true)-y_pred)))

class Regresser(object):
    
    def __init__(self,eta,momentum,lamb,n_unit = 30):
        self.loader = Dataset_Loader('ML-CUP17-TR.csv')
        self.loader.load_cup_train()
        self.loader.split_train_get_test(0.2)
        self.input_dim = len(self.loader.x[0])
        self.lamb = lamb
        self.n_unit = n_unit
        self.eta=eta
        self.momentum = momentum
    def init_model(self):
        
        self.model = Sequential()
#        self.model.add(Dropout(0.25))
        self.model.add(Dense(input_dim=self.input_dim,units=self.n_unit, activation='softmax',))#,W_regularizer=regularizers.l2(self.lamb)))
        self.model.add(Dense(units=2, activation='linear'))#,W_regularizer=regularizers.l2(self.lamb)))

        sgd = SGD(lr=self.eta, momentum=self.momentum, nesterov=False)
        self.model.compile(loss='mean_squared_error', optimizer=sgd)

    def fit_model(self, epochs=300, batch_size=100):
        self.batch_size = batch_size
        print self.model.summary()

        # TODO: use a param
        #checkpointer = ModelCheckpoint(filepath=model, verbose=1, save_best_only=True, period = 1)
        #earlystopping = EarlyStopping(verbose=True, patience=5, monitor='val_loss')

        self.model.fit(
            self.loader.x,
            self.loader.y,
            #validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            verbose=False,
            #callbacks=[checkpointer, earlystopping]
           )
    def evaluate_model(self,c):
        mee = 0
        hx = self.model.predict(np.array(self.loader.x_test))
        for i,predicted in enumerate(hx):
            #print self.loader.y_test[i],predicted
            mee = mee + np.sqrt(np.sum(np.square(np.array(self.loader.y_test[i])-predicted)))
        print mee / len(self.loader.x_test)
        #self.loader.plot_test_2D(self.loader.y_test,hx,name='Net_'+str(self.n_unit)+'_'+'try_'+str(c)+'_'+str(self.batch_size))
        c = c+1
for i in range(0,5):
    r = Regresser(eta=0.15,momentum=.3,lamb=0.0,n_unit=20)
    r.init_model()
    r.fit_model(epochs=300,batch_size=128 )#len(r.loader.x))
    r.evaluate_model(i)