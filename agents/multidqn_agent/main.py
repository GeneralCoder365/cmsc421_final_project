import os
import pandas as pd
import numpy as np

from data_manager import get_csvs_from_dir, load_data, feature_engineering
from stock_environment import OptionsTradingEnv
from dqn_agent import DeepQTradingAgent
from ensemble_manager import full_ensemble, perc_ensemble

def save_predictions(date_range, actuals, predictions, results_dir, model_name='multidqn'):
    results_path = os.path.join(results_dir, 'predictions.csv')
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
    else:
        results_df = pd.DataFrame(columns=['date', 'actual'])

    results_df[model_name] = predictions  # Update or create the column for DQN predictions
    results_df['date'] = date_range
    results_df['actual'] = actuals
    results_df.to_csv(results_path, index=False)
    print(f"Predictions saved in {results_path}")

def main():
    print("Starting the main process...")

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r, sigma, T = 0.01, 0.2, 1/12  # Configuration parameters
    volume = 1000000  # Large volume for institutional level trading
    model_names = ['multidqn1', 'multidqn2', 'multidqn3']  # Names of the models
    window_length = 50
    nb_steps = 10000
    nb_episodes_test = 10

    print("Loading data...")
    data_directory = os.path.join(project_dir, 'environment', 'datasets')
    csv_files, stock_names = get_csvs_from_dir(data_directory)

    for file_path in csv_files:
        print(f"Processing file: {file_path}")

        df = load_data(file_path)
        df = feature_engineering(df)

        # Initialize and set up the trading environment
        env = OptionsTradingEnv(df, r=r, sigma=sigma, T=T, volume=volume, window_length=window_length)

        # Initialize agents
        agents = []
        for model_name in model_names:
            input_shape = (2,)  # Adjust based on your actual environment setup
            output_shape = env.action_space.n
            agent = DeepQTradingAgent(input_shape=input_shape, output_shape=output_shape, window_length=window_length)
            agents.append(agent)

            # Train and save model
            print(f"Training and saving model: {model_name}")
            agent.train(env, nb_steps, model_name)

        # Collect and average predictions from each agent
        predictions = []
        for index, agent in enumerate(agents):
            # Load model weights
            model_path = os.path.join(project_dir, 'agents', 'trained_models', model_names[index] + '.h5')
            print(f"Loading model: {model_names[index]} for testing")
            agent.load(model_path)

            # Test the model
            predicted_prices = agent.test(env, nb_episodes_test)
            predictions.append(predicted_prices)

        # Employ ensemble decision-making
        if len(predictions) > 1:
            print("Applying ensemble decision making...")
            ensemble_results = perc_ensemble(predictions)  # Choose based on your preference
            ensemble_decisions = ensemble_results['Ensemble Decision'].tolist()
            print(f"Ensemble Decisions: {ensemble_decisions}")
        else:
            ensemble_decisions = predictions[0]

        # Update the environment with new predicted prices
        env.set_predicted_prices(ensemble_decisions)

        # Date range for predictions
        date_range = pd.date_range(start=df.index[-nb_episodes_test], periods=nb_episodes_test)

        # Save results
        actuals = df['actual'].iloc[-nb_episodes_test:].tolist()
        results_dir = os.path.join(project_dir, 'results', os.path.basename(file_path).split('.')[0])
        save_predictions(date_range, actuals, ensemble_decisions, results_dir)

    print("Process completed.")

if __name__ == '__main__':
    main()