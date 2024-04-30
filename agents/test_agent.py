import lstm_agent as lstm
import lstm2
import rnn_agent as rnn
import joblib
import keras
import pandas as pd
import numpy as np

# data = pd.read_csv('datasets/abc.csv')

# model = lstm.create_model(data, 'abc')

data = pd.read_csv('datasets/abc.csv')

model = keras.models.load_model('./models/abc_lstm_model.keras')
data = lstm.build_table(data)
X, Y = lstm.pre_train(data)

_, X_test, _, Y_test = lstm.split_data(X, Y, 0.9)

prediction = model.predict(X_test[[500]])
print('prediction', prediction)
print('actual', Y_test[500])
