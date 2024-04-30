# Code based on https://www.youtube.com/watch?v=hpfQE0bTeA4
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM
from keras.layers import Dense
from keras import optimizers
from keras.models import Model
from keras.layers import Dense, Input, Activation

def build_table(data):

    # Adding indicators
    data['RSI']=ta.rsi(data.close, length=15) #relative strength index
    data['EMAF']=ta.ema(data.close, length=20) #fast
    data['EMAM']=ta.ema(data.close, length=100) #medium
    data['EMAS']=ta.ema(data.close, length=150) #slow

    data['Target'] = data['close_adj']-data['open']
    data['Target'] = data['Target'].shift(-1)

    data['TargetClass'] = [1 if data['Target'][i]>0 else 0 for i in range(len(data))]

    data['TargetNextClose'] = data['close_adj'].shift(-1) #closing price of next day

    data.dropna(inplace=True)
    data.reset_index(inplace = True)
    data.drop(['close', 'date', 'time', 'id'], axis=1, inplace=True)

    sc = MinMaxScaler(feature_range=(0,1))
    data_set_scaled = sc.fit_transform(data)
    return data_set_scaled

# splits training data into value and prediction
def pre_train(data_set_scaled, backcandles=10):

    # only want to feed model with first 12 columns in order to predict the target, targetClass and TargetNextClose
    # multiple feature from data provided to the model
    X = []

    for j in range(12): #data_set_scaled[0].size):#2 columns are target not X
        X.append([])
        for i in range(backcandles, data_set_scaled.shape[0]):#backcandles+2
            X[j].append(data_set_scaled[i-backcandles:i, j])

    X=np.moveaxis(X, [0], [2])
    X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-1])
    y=np.reshape(yi,(len(yi),1))

    return X, y

def train(X_train, Y_train, backcandles=10):

    lstm_input = Input(shape=(backcandles, 12), name='lstm_input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=Y_train, batch_size=32, epochs=10, shuffle=True, validation_split = 0.1)
    return model

# def action(model, prev_day, acutal_prev_close, threshold = 0.005):

#     prediction = model.predict(np.array([prev_day, ]))
#     prediction = prediction[0][0]

#     print('prev', acutal_prev_close)
#     print('predict', prediction)

#     diff = prediction - acutal_prev_close

#     if(diff > threshold):

#         return 'Buy'
    
#     elif(diff < -1 * threshold):

#         return 'Sell'
#     else:
#         return 'Hold'

def split_data(X, Y, percent_train):

    splitlimit = int(len(X)*percent_train)
    X_train, X_test = X[:splitlimit], X[splitlimit:]
    Y_train, Y_test = Y[:splitlimit], Y[splitlimit:]

    return X_train, X_test, Y_train, Y_test
    
def create_model(data, name):

    # data = yf.download(tickers = '^RUI', start = '2012-03-11', end = '2022-07-10')

    # data = pd.read_csv('datasets/abc.csv')
    data_set_scaled = build_table(data)

    backcandles = 10
    X, Y = pre_train(data_set_scaled, backcandles)
    X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.8)

    trained_model = train(X_train, Y_train, backcandles)
    
    Y_pred = trained_model.predict(X_test)
    for i in range(10):
        print(Y_pred[i], Y_test[i])

    loss = trained_model.evaluate(X_test, Y_test)
    print(loss)

    # fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16,8))
    # ax.plot(Y_test, color = 'black', label = 'Test')
    # ax.plot(Y_pred, color = 'green', label = 'pred')
    # ax.legend()

    # plt.show()

    trained_model.save('./models/' + name + '_lstm_model.keras')

    return trained_model

def test(trained_model):

    data = pd.read_csv('datasets/abc.csv')
    # model = lstm(data)
    data_set_scaled = build_table(data)

    # print(type(data_set_scaled))

    backcandles = 10
    X, Y = pre_train(data_set_scaled, backcandles)

    X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.95)

    Y_pred = trained_model.predict(X_test)
    for i in range(10):
        print(Y_pred[i], Y_test[i])

    # result = model.action(trained_model, X_test[2], Y_test[1][0])
    # print(result)

    loss = trained_model.evaluate(X_test, Y_test)
    print(loss)

    # fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(16,8))
    # ax.plot(Y_test, color = 'black', label = 'Test')
    # ax.plot(Y_pred, color = 'green', label = 'pred')
    # ax.legend()

    # plt.show()