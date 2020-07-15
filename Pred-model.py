from __future__ import print_function
import numpy as np
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.regularizers import L1L2
from datetime import datetime, timedelta
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import numba
        


def data():
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg
        
    dataset = pd.read_csv('formatted_128444.csv')
    days = []


    for i in range (1635):
        s = str(dataset.loc[i,'DATE'])
        # you could also import date instead of datetime and use that.
        date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
        days.append(date.weekday())
        dataset.loc[i,'DATE'] = date.date()
        
    dataset['DAY'] = days
    dataset = dataset[['DATE', 'DAY', 'CLS']]
    
    values = dataset.iloc[:, 2:4].values;
    # integer encode direction
    # ensure all data is float
    values = values.astype('float32')
    # normalize features

    scaler = MinMaxScaler(feature_range=(0, 1))
    #values = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(values, 4, 1)
    
    values = reframed.values
    train = values[:, :]
    val = values[1470:, :]
    test = values[1300:1470, :]
    # train_y = y[:150]
    # test_y = y[150:]
    # split into input and outputs
    train_X, train_y = train[:, :-2], train[:, -2]
    val_X, val_y = val[:, :-2], val[:, -2]
    test_X, test_y = test[:, :-2], test[:, -2]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    return train_X, train_y, val_X, val_y, test_X, test_y

@numba.jit 
def create_model(train_X, train_y, val_X, val_y):

    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
        
    # design network
    model = Sequential()
    model.add(LSTM(units = 128, return_sequences = True, input_shape=(train_X.shape[1], train_X.shape[2]), bias_regularizer=L1L2(l1=0.1, l2=0.05)))
    model.add(Dropout({{uniform(0, 1)}}))

    model.add(LSTM(units = 64))
    model.add(Dropout({{uniform(0, 1)}}))

    # model.add(Dense(16,init='uniform',activation='relu'))
    model.add(Dense({{choice([64, 32, 16, 8])}}))

    model.add(Dense(units = 1))
    model.compile(optimizer = {{choice(['adam'])}}, loss = root_mean_squared_error)
    # fit network
    history = model.fit(train_X, train_y, epochs = 500, batch_size={{choice([25])}}, verbose=2, shuffle=False,validation_split=0.1)
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amin(history.history['val_loss']) 
    print('Best validation loss of epoch:', validation_acc)
    return {'loss': validation_acc, 'status': STATUS_OK, 'model': model}
    #validation_loss = history.history['val_loss']
    #print('Best validation acc of epoch:', validation_loss)
    #return {'loss': validation_loss, 'status': STATUS_OK, 'model': model}


if __name__=="__main__": 
    best_run, best_model = optim.minimize(model=create_model,data=data,algo=tpe.suggest,max_evals=10, trials=Trials())
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)