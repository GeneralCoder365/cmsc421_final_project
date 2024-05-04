import gymnasium as gym # reinforcement learning environment
from gymnasium import spaces
import numpy as np # matrices
import pandas # data processing
import os # read directories

class markEnv(gym.Env):
    # set up environmnet: read and process data (must be OHLCV and csv format), action and observation space
    def __init__(self, train=1, operationCost=0, callback=None):
        # declare class variables
        self.episode = 1
        self.train = train # value between 0 and 1 (inclusive) for training agent
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

        # use pandas to read each individual stock data csv 
        path = "datasets/"
        for file in os.listdir(path):
            # declare a dataframe for the file and process any irrelevant data
            df = pandas.read_csv(path + "/" + file)

            # drop id column and convert date and time columns to relevant classes
            df.drop('id', axis='columns', inplace=True)
            df['date'] = pandas.to_datetime(df['date'], format="%Y-%m-%d")
            df['time'] = pandas.to_datetime(df['time'], format="%H:%M:%S").dt.time

            # append stock name and data to list
            self.stocks.append((file.replace(".csv", ""), df))

        # action space: 0 (hold), 1 (buy), 2 (sell)
        self.action_space = spaces.Discrete(3)

        # observation space: open, close, high, low, volume values will always be some natural number
        self.observation_space = spaces.Box(low=0, high=float("inf"), dtype=np.float32)

        # length of stocks list and limit
        self.numStocks = len(self.stocks)
        self.numObservations = len(self.stocks[0][1]['date']) # number of data rows, could use any column name
        self.limit = round(self.train*self.numObservations)

    # performs an action in the environment
    # action = an array of length length = stocks list with an integer between 0-2 for each stock
    # returns the new state, reward (array of length = stocks list), and terminated
    def step(self, action: list):
        # initiate reward
        self.reward = np.zeros(shape=self.numStocks, dtype=np.float32)

        # index for total stocks
        stockIndex = 0

        # calculate reward and possibleGain from actions
        for choice in action:
            self.open = self.stocks[stockIndex][1]['open'][self.agentLocation]
            self.close = self.stocks[stockIndex][1]['close'][self.agentLocation]
            percentChange = (self.close - self.open)/self.open

            if choice == 0: # hold action
                self.reward[stockIndex] = 0
            elif choice == 1: # buy action
                self.reward[stockIndex] = percentChange - self.operationCost
            elif choice == 2: # sell action
                self.reward[stockIndex] = (-1*percentChange) - self.operationCost

            stockIndex += 1

        # increment agent index
        self.agentLocation += 1

        # determine if agent has reached the limit
        if self.agentLocation == self.limit:
            self.terminated = True

        return self.agentLocation, self.reward, self.terminated, {}


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


        

