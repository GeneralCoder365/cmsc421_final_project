import keras
import lstm_agent as lstm
class agent_helper():

    def __init__(self, env, stock_list):

        self.env = env
        # self.stock_list = stock_list

        self.model_list = []
        self.lstm_dataX = []
        self.lstm_dataY = []

        for name, data in stock_list:

            path = './models/' + name + '.keras'
            curr = keras.models.load_model(path)
            self.model_list.append(curr)

            # getting data to make predictions and compare to actual
            tmp = lstm.build_table(data)
            tmpx, tmpy = lstm.pre_train(tmp)
            _, X_test, _, Y_test = lstm.split_data(tmpx, tmpy, 0.8)
            self.lstm_dataX.append(X_test)
            self.lstm_dataY.append(Y_test)
        
    def action(self, observation_idx):

        threshold = 0.005

        action_list = []

        for idx, model in enumerate(self.model_list):

            prediction = model.predict(self.lstm_dataX[idx][[observation_idx]])
            prev_day = self.lstm_dataY[idx][observation_idx - 1]

            diff = prediction[0][0] - prev_day[0]

            if(diff > threshold):

                action_list.append(1) #buy
            
            elif(diff < -1 * threshold):

                action_list.append(2) #sell
            else:
                action_list.append(0) #hold

        return action_list
