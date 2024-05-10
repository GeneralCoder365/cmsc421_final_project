import gym
from gym import spaces
import numpy as np
from scipy.stats import norm

class OptionsTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data, r=0.01, sigma=0.2, T=1/12, volume=1000000, window_length=50):
        super().__init__()
        self.data = data
        self.r = r
        self.sigma = sigma
        self.T = T
        self.volume = volume
        self.window_length = window_length
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_length,), dtype=np.float32)
        self.current_step = window_length
        self.predicted_prices = np.zeros(len(data))

    def reset(self):
        self.current_step = self.window_length
        self.predicted_prices = self.data['actual'].values
        return self._next_observation()

    def _next_observation(self):
        if self.current_step >= len(self.data):
            return np.zeros(self.window_length)
        prices_window = self.data['actual'][self.current_step-self.window_length:self.current_step]
        return prices_window.values

    def step(self, action, predicted_price=None):
        if predicted_price is not None:
            self.predicted_prices[self.current_step] = predicted_price
        actual_price = self.data['actual'][self.current_step]
        predicted_price = self.predicted_prices[self.current_step]
        reward = self.calculate_profit(actual_price, predicted_price, action)
        self.current_step += 1
        done = self.current_step >= len(self.data)
        return self._next_observation(), reward, done, {}

    def calculate_profit(self, actual_price, predicted_price, action):
        if action == 1:  # Buy Call
            return max(0, (self.black_scholes(predicted_price, actual_price, self.T, self.r, self.sigma, 'call') - 
                        self.black_scholes(actual_price, actual_price, self.T, self.r, self.sigma, 'call')) * self.volume)
        elif action == 2:  # Buy Put
            return max(0, (self.black_scholes(predicted_price, actual_price, self.T, self.r, self.sigma, 'put') - 
                        self.black_scholes(actual_price, actual_price, self.T, self.r, self.sigma, 'put')) * self.volume)
        return 0

    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def render(self, mode='human', close=False):
        print(f"Step: {self.current_step}, Actual: {self.data['actual'][self.current_step]}, Predicted: {self.predicted_prices[self.current_step]}")