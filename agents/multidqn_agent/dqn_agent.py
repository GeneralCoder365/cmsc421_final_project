import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
Adam._name = 'hey' #! workaround for a bug in keras-rl
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

class DeepQTradingAgent:
    def __init__(self, input_shape, output_shape, window_length=1):
        print(f"Initializing agent with output shape {output_shape}")
        self.input_shape = (window_length,) + input_shape  # Adjusting input shape based on window_length
        self.output_shape = output_shape
        self.window_length = window_length
        self.model = self.build_model()
        self.agent = self.build_agent(self.model)

    def build_model(self):
        print("Building model...")
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))  # Ensuring the input shape is correctly defined
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.output_shape, activation='linear'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    def build_agent(self, model):
        print("Building DQN agent...")
        memory = SequentialMemory(limit=50000, window_length=self.window_length)
        policy = BoltzmannQPolicy()
        dqn_agent = DQNAgent(model=model, nb_actions=self.output_shape, memory=memory,
                             nb_steps_warmup=500, target_model_update=1e-2, policy=policy)
        dqn_agent.compile(Adam(learning_rate=0.001), metrics=['mae'])
        return dqn_agent

    def train(self, env, nb_steps, model_name):
        print("Training agent...")
        print(f"Pre-filling memory with {self.agent.nb_steps_warmup} steps...")

        while self.agent.memory.nb_entries < self.agent.nb_steps_warmup:
            observation = env.reset()  # Resets and gets the initial observation
            done = False
            while not done:
                action = np.random.randint(0, self.output_shape)  # Take random actions
                next_observation, reward, done, _ = env.step(action)
                # Ensure the observations are reshaped for memory consistency
                observation_reshaped = np.array(observation).reshape(1, -1)
                next_observation_reshaped = np.array(next_observation).reshape(1, -1)
                self.agent.memory.append(observation_reshaped, action, reward, next_observation_reshaped, done)
                observation = next_observation
                if self.agent.memory.nb_entries >= self.agent.nb_steps_warmup:
                    break

        print(f"Starting training for {nb_steps} steps...")
        self.agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=2)

        # Save the final weights
        project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_dir, 'agents', 'trained_models', 'multidqn_models', model_name + '.h5')
        self.save(model_path)

    def test(self, env, nb_episodes):
        print(f"Testing for {nb_episodes} episodes...")
        results = self.agent.test(env, nb_episodes=nb_episodes, visualize=False)
        predictions = []
        for episode in results.history['episode_reward']:
            predicted_price = env.predicted_prices[env.current_step - 1]  # Corrected indexing
            predictions.append(predicted_price)
            print(f"Predicted Price from test: {predicted_price}")
        return predictions

    def save(self, filepath):
        print(f"Saving model weights to {filepath}")
        self.agent.save_weights(filepath, overwrite=True)

    def load(self, filepath):
        print(f"Loading model weights from {filepath}")
        self.agent.load_weights(filepath)