import numpy as np
import pandas as pd

def full_ensemble(predictions):
    # Aggregates predictions by requiring unanimous agreement among agents.
    # Arguments:
    #     predictions: A list of lists, where each sublist represents an agent's set of predictions.
    # Returns:
    #     A DataFrame with aggregated results based on unanimous agreement.

    # Convert list of predictions into a DataFrame
    prediction_df = pd.DataFrame(predictions).transpose()
    # Require unanimous agreement (all predictions in a row must be the same)
    unanimous = prediction_df.nunique(axis=1) == 1
    # If unanimous, take the first agent's prediction; otherwise, default to 'No Action'
    result = prediction_df[unanimous].iloc[:, 0].fillna('No Action')
    return pd.DataFrame({'Ensemble Decision': result})

def perc_ensemble(predictions, threshold=0.7):
    # Aggregates predictions based on a percentage threshold for agreement.
    # Arguments:
    #     predictions: A list of lists, where each sublist represents an agent's set of predictions.
    #     threshold: The percentage of agents that need to agree on a decision.
    # Returns:
    #     A DataFrame with aggregated results based on the threshold.
    
    prediction_df = pd.DataFrame(predictions).transpose()
    # Calculate the mode and its frequency for each row
    mode_count = prediction_df.mode(axis=1, numeric_only=True).count(axis=1)
    total_count = len(predictions)
    # Apply threshold: if the mode's frequency meets the threshold, use the mode; otherwise, 'No Action'
    valid_mode = mode_count / total_count >= threshold
    mode_values = prediction_df.mode(axis=1, numeric_only=True)[0]  # Assuming the mode is the first one listed
    decision = np.where(valid_mode, mode_values, 'No Action')
    return pd.DataFrame({'Ensemble Decision': decision})