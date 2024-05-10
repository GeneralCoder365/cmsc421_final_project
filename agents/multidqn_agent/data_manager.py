import os
import pandas as pd

# Retrieve paths to CSV files within the specified directory.
def get_csvs_from_dir(directory):
    datasets_dir = os.path.join(directory)
    try:
        csv_files = [os.path.join(datasets_dir, file) for file in os.listdir(datasets_dir) if file.endswith('.csv')]
        stock_names = [os.path.basename(file).split(".")[0] for file in csv_files]
        return csv_files, stock_names
    except FileNotFoundError:
        print(f"The directory {datasets_dir} does not exist.")
        return [], []

# Load data from a CSV file and preprocess it for DQN usage.
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df.set_index('datetime', inplace=True)
    df = df[['close']].resample('D').mean()
    df.interpolate(method='linear', inplace=True)
    df.rename(columns={'close': 'actual'}, inplace=True) # renamed to match the stock_environment.py
    return df


# Apply feature engineering to the dataframe for use as state inputs in DQN.
def feature_engineering(df):
    df['return'] = df['actual'].pct_change()
    df['moving_avg_5'] = df['actual'].rolling(window=5).mean()
    df['moving_avg_10'] = df['actual'].rolling(window=10).mean()
    df['rsi'] = compute_rsi(df['actual'], window=14)
    df.dropna(inplace=True)
    return df

# Compute the Relative Strength Index for a given Pandas Series.
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))