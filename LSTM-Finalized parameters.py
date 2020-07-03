#!/usr/bin/env python
# coding: utf-8

# In[23]:


import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('formatted_128714-JKH.csv')
# print(dataset_train.head)
training_set = dataset_train.iloc[:, 5:6]
training_set.dropna(inplace=True)
training_set_open = dataset_train.iloc[:, 3:4].values

dataset_train = pd.read_csv('formatted_128714-JKH.csv')
copy = pd.read_csv('formatted_128714-JKH.csv')
dataset_train = dataset_train[['DATE','OP', 'CLS']]
days = []

from datetime import datetime, timedelta

for i in range (1773):
    s = str(dataset_train.loc[i,'DATE'])
    # you could also import date instead of datetime and use that.
    date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
    days.append(date.weekday())
    dataset_train.loc[i,'DATE'] = date.date()
    
dataset_train['DAY'] = days


# In[24]:


# load dataset
dataset = pd.read_csv('formatted_128714-JKH.csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
i = 1
# plot each column
plt.figure()
for group in groups:
    plt.subplot(len(groups), 1, i)
    plt.plot(values[:, group])
    plt.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
plt.show()


# In[25]:


from sklearn.preprocessing import MinMaxScaler

dataset = pd.read_csv('formatted_128714-JKH.csv')
copy = pd.read_csv('formatted_128714-JKH.csv')
dataset = dataset[['DATE','OP', 'CLS','TREND']]
days = []

from datetime import datetime, timedelta

for i in range (1773):
    s = str(dataset.loc[i,'DATE'])
    # you could also import date instead of datetime and use that.
    date = datetime(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))
    days.append(date.weekday())
    dataset.loc[i,'DATE'] = date.date()
    
dataset['DAY'] = days
dataset = dataset[['DATE', 'DAY', 'CLS','TREND']]

#values = dataset.iloc[:, 1:4].values;
values = dataset[['DAY','TREND', 'CLS']]
print(values.shape)
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
values[['DAY','TREND']] = scaler.fit_transform(values[['DAY','TREND']])
values[['CLS']] = scaler2.fit_transform(values[['CLS']])
print(values)


# convert series to supervised learning
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
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


reframed = series_to_supervised(values, 4, 1)
print(reframed)


# In[26]:


corr = reframed.corrwith(reframed['var1(t)'])
corr.sort_values().plot.barh(color = 'blue',title = 'Strength of Correlation')


# In[27]:


# split into train and test sets
values = reframed.values
train = values[:1400, :]
val = values[1400:1525, :]
test = values[1525:, :]
# train_y = y[:150]
# test_y = y[150:]
# split into input and outputs
train_X, train_y = train[:, :-2], train[:, -1]
val_X, val_y = val[:, :-2], val[:, -1]
test_X, test_y = test[:, :-2], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
# train_y = train_y.reshape((train_y.shape[0], 1, 1))
# test_y = test_y.reshape((test_y.shape[0], 1, 1))
print(train_X.shape, train_y.shape, val_X.shape, val_y.shape, test_X.shape, test_y.shape)


# In[28]:


train_X


# In[29]:


print(train_X)
train_y.shape


# In[30]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.regularizers import L1L2


# In[31]:


from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))


# In[32]:


model = Sequential() 
model.add(LSTM(units = 128, return_sequences = True, input_shape=(train_X.shape[1], train_X.shape[2]), bias_regularizer=L1L2(l1=0.001, l2=0.001))) 
model.add(Dropout(0.5))

model.add(LSTM(units = 64)) 
model.add(Dropout(0.5))

model.add(Dense(units=16,init='uniform',activation='relu'))

model.add(Dense(units = 1)) 
model.compile(optimizer = 'adam', loss = root_mean_squared_error)
history = model.fit(train_X, train_y, epochs = 700, batch_size=10, validation_data=(val_X, val_y), verbose=2, shuffle=False)

predicted_stock_price = model.predict(test_X, batch_size = 10)
plt.plot(test_y, color = 'black', label = 'Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.savefig('predicted_stock_priceday1.png')
plt.show()


# In[33]:


# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()


# In[34]:


rms = np.sqrt(np.mean(np.power((np.array(test_y)-np.array(predicted_stock_price)),2)))
mse = np.mean(np.power((np.array(test_y)-np.array(predicted_stock_price)),2))
print("Root Mean Squared Error:",rms)
print("Mean Squared Error:",mse)

test2 = values[1525:, :]
test_trend = test[:, -2]

test_prices = test[:, -4]

#print(test_prices)

trend = []
i=0

for n in predicted_stock_price:
    if((n-test_prices[i])>0):
        trend.append(1)
    else:
        trend.append(0)
    i = i+1
    
#print(trend)

from sklearn.metrics import accuracy_score

print("Accuracy for trend prediction:", accuracy_score(test_trend, trend))


# In[35]:


# plt.plot(val2, color = 'black', label = 'Stock Price')
# plt.plot(val1, color = 'green', label = 'Predicted Stock Price')
# plt.title('Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.savefig('predicted_stock_priceday1.png')
# plt.show()
predicted_stock_price = model.predict(test_X, batch_size = 25)
predicted_stock_price = scaler2.inverse_transform(predicted_stock_price)
test_y = scaler2.inverse_transform(test_y.reshape(-1, 1))
plt.plot(test_y, color = 'black', label = 'Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
#print(np.mean(np.power((np.array(test_y)-np.array(predicted_stock_price)),2)))


# In[ ]:




