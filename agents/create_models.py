import os
import pandas as pd
import lstm_agent as lstm

path = "datasets"

for file in os.listdir(path):
    
    stock = file[:-4]
    data = pd.read_csv(path + "/" + file)

    print('Creating model for', stock)

    lstm.create_model(data, stock)


