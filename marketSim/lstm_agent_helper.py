import keras
import lstm_agent as lstm
import tensorflow as tf
import pandas as pd

class agent_helper():

    # creates agent to make predictions for each stock
    def __init__(self, env, stock_list, train_test):

        self.env = env
        # self.stock_list = stock_list

        self.model_list = []
        self.lstm_dataX = []
        self.lstm_dataY = []
        self.total_correct_pred_daily = []
        self.num_days = 0

        for name, data in stock_list:

            print('Preparing LSTM for', name)
            
            # load all lstm models
            path = 'models/' + name + '_lstm_model.keras'
            curr = keras.models.load_model(path)
            self.model_list.append(curr)

            # preparing data to make predictions and compare to actual
            tmp = lstm.build_table(data)
            tmpx, tmpy = lstm.pre_train(tmp)
            _, X_test, _, Y_test = lstm.split_data(tmpx, tmpy, train_test)
            self.lstm_dataX.append(X_test)
            self.lstm_dataY.append(Y_test)

        self.total_correct_pred = [0] * len(self.model_list)
        
    # given a observation index, return the optimal action for each stock
    def action(self, observation_idx):

        self.num_days += 1

        correct_pred = 0

        threshold = 0.005

        action_list = []

        # action for every stock along with analysis
        for idx, model in enumerate(self.model_list):

            # get prediction
            prediction = model(self.lstm_dataX[idx][[observation_idx]])
            prev_day = self.lstm_dataY[idx][observation_idx - 1]

            diff = prediction[0][0] - prev_day[0]

            curr_action = 0

            # make decision
            if(diff > threshold): # buy
                curr_action = 1
            
            elif(diff < -1 * threshold): # sell
                curr_action = 2

            action_list.append(curr_action)

            actual_val = self.lstm_dataY[idx][observation_idx]
            actual_diff = actual_val[0] - prev_day[0]

            # # compare prediction to actual result
            # if(actual_diff > threshold): # should buy

            #     if(curr_action == 1):

            #         correct_pred += 1
            #         self.total_correct_pred[idx] += 1
                    
            
            # elif(actual_diff < -1 * threshold): # should sell

            #     if(curr_action == 2):

            #         correct_pred += 1
            #         self.total_correct_pred[idx] += 1

            # else: # should hold
                
            #     if(curr_action == 0):

            #         correct_pred += 1
            #         self.total_correct_pred[idx] += 1

            if(actual_diff > 0):

                if(curr_action != 2): # did not sell well price went up

                    correct_pred += 1
                    self.total_correct_pred[idx] += 1
            else:
            
                if(curr_action != 1): # did not buy well price went down

                    correct_pred += 1
                    self.total_correct_pred[idx] += 1

            # print(idx)
            # print(len(self.total_correct_pred))
            # self.total_correct_pred[idx] += correct_pred

            
        print('Current correct predictions:', correct_pred)
        self.total_correct_pred_daily.append(correct_pred)

        return action_list
    
    def get_total_correct_predictions_daily(self):

        return self.total_correct_pred_daily
    
    def get_total_correct_predictions(self):

        return self.total_correct_pred, self.num_days