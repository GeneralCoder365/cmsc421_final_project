import gymnasium as gym # reinforcement learning environment
from gymnasium import spaces
import numpy as np # matrices
import pandas # data processing
import os # read directories

class markEnv(gym.Env):
    # set up environmnet: read and process data (must be OHLCV and csv format), action and observation space
    def __init__(self, train_test_stop=0.8, operationCost=0, callback=None):
        # declare class variables
        self.episode = 1
        self.train_test_stop = train_test_stop # value between 0 and 1 (inclusive) for training agent
        self.operationCost = operationCost # cost to ensure optimal trading
        self.callback = callback 
        self.done = False # determines if episode is done
        self.reward = None # reward to return in step
        self.open = 0
        self.close = 0
        self.agentLocation = 0 # index of agent in the stock data
        self.terminated = False # whether one episode is done or not

        # data will contain a tuple of the stock name and stock data
        self.stocks = []
        self.unmodified_stocks = []

        # use pandas to read each individual stock data csv 
        path = "datasets/"
        for file in os.listdir(path):
            # declare a dataframe for the file and process any irrelevant data
            df = pandas.read_csv(path + "/" + file)
            df_unmodified = pandas.read_csv(path + "/" + file)

            # drop id column and convert date and time columns to relevant classes
            df.drop('id', axis='columns', inplace=True)
            df['date'] = pandas.to_datetime(df['date'], format="%Y-%m-%d")
            df['time'] = pandas.to_datetime(df['time'], format="%H:%M:%S").dt.time

            # append stock name and data to list
            self.stocks.append((file.replace(".csv", ""), df))
            self.unmodified_stocks.append((file.replace(".csv", ""), df_unmodified))

        # action space: 0 (hold), 1 (buy), 2 (sell)
        self.action_space = spaces.Discrete(3)

        # observation space: open, close, high, low, volume values will always be some natural number
        self.observation_space = spaces.Box(low=0, high=float("inf"), dtype=np.float32)

        # length of stocks list and limit
        self.numStocks = len(self.stocks)
        self.numObservations = len(self.stocks[0][1]['date']) # number of data rows, could use any column name
        # self.limit = round(self.train*self.numObservations)
        self.limit = round((1-self.train_test_stop)*self.numObservations)

        self.net = [0] * self.numStocks # used to see net profit

        self.results = pandas.DataFrame(columns=['date', 'stock', 'decision', 'profit'])



    def get_unmodified_stocks(self):

        return self.unmodified_stocks
    
    def get_train_test_stop(self):

        return self.train_test_stop

    # performs an action in the environment
    # action = an array of length length = stocks list with an integer between 0-2 for each stock
    # returns the new state, reward (array of length = stocks list), and terminated

    def step(self, action: list):
        # initiate reward
        self.reward = np.zeros(shape=self.numStocks, dtype=np.float32)

        # index for total stocks
        stockIndex = 0

        # calculate reward and possibleGain from actions
        for idx, choice in enumerate(action):
            self.open = self.stocks[stockIndex][1]['open'][self.agentLocation]
            self.close = self.stocks[stockIndex][1]['close'][self.agentLocation]
            percentChange = (self.close - self.open)/self.open

            curr_net_profit = 0

            if choice == 0: # hold action
                self.reward[stockIndex] = 0
            elif choice == 1: # buy action
                self.reward[stockIndex] = percentChange - self.operationCost
                curr_net_profit = self.close - self.open
            elif choice == 2: # sell action
                self.reward[stockIndex] = (-1*percentChange) - self.operationCost
                curr_net_profit = self.open - self.close

            stockIndex += 1

            self.net[idx] += curr_net_profit
        
            to_append_results = pandas.DataFrame([[self.stocks[stockIndex-1][1]['date'][self.agentLocation], idx, choice, curr_net_profit]], columns=self.results.columns)
            self.results = pandas.concat([to_append_results, self.results])
        # increment agent index
        self.agentLocation += 1

        # determine if agent has reached the limit
        if self.agentLocation == self.limit:
            self.terminated = True

        # print(self.results)

        return self.agentLocation, self.reward, self.terminated, {}
    
    def get_profits(self):
        
        return self.net


    # reset environment variables for next episode
    def reset(self):
        # increment episode
        self.episode += 1

        # reset necessary variables
        self.terminated = False
        self.reward = None
        self.open = 0
        self.close = 0
        self.agentLocation = 0

        # return agent location
        return self.agentLocation, {}
    

    # return current agent location
    # ***might need to tweak, just thought that returning the date and time might be more helpful than a counter
    # so the observationWindow (number of data in one observation i.e. a single day) could be better
    def getObservation(self):
        return self.agentLocation


        

