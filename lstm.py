# importing required libraries
import tensorflow

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_data import k

data = k.sort_index(ascending=True, axis=0)
# Sort data in ascending order, original data in descending order

usedk = pd.DataFrame(index=range(0, len(k)), columns=['trade_date', 'low'])
# I am a conservative investor, so I tend to predict daily low, you can change to close or high
for i in range(0, len(k)):
    usedk['trade_date'][i] = data['trade_date'][i]
    usedk['low'][i] = data['low'][i]

    # setting index
    usedk.index = usedk.trade_date
    usedk.drop('trade_date', axis=1, inplace=True)

    # creating train and test sets
    dataset = usedk.values
    train = dataset[0:100, :]
    valid = dataset[100:, :]

    # converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    x_train, y_train = [], []
    for i in range(15, len(train)):
        x_train.append(scaled_data[i-15:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
                   input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

    # predicting 50 values, using past 15 from the train data

    inputs = usedk[len(usedk) - len(valid) - 15:].values

    inputs = inputs.reshape(-1, 1)

    inputs = scaler.transform(inputs)

    X_test = []

    for i in range(15, inputs.shape[0]):

        X_test.append(inputs[i-15:i, 0])

    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    closing_price = model.predict(X_test)

    closing_price = scaler.inverse_transform(closing_price)

rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
print(rms)

# for plotting
train = usedk[:100]
valid = usedk[100:]
valid['Predictions'] = closing_price
plt.plot(train['low'])
plt.plot(valid[['low', 'Predictions']])
