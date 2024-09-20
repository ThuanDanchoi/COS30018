# File: predictor.py
# Purpose: This module is responsible for making predictions using the trained
# model. It processes the most recent data and outputs the predicted stock price.

import numpy as np


def predict_next_day(model, last_sequence, scaler, prediction_days):
    """
    Predict the stock price for the next day based on the last sequence of data.
    """
    real_data = last_sequence[-prediction_days:]
    prediction = model.predict(real_data)
    return scaler.inverse_transform(prediction)

def multistep_predict(model, last_sequence, scaler, steps):
    """
    Predict stock prices for multiple days into the future (multistep prediction).
    """
    predictions = []
    current_sequence = last_sequence

    for _ in range(steps):
        prediction = model.predict(current_sequence)
        prediction = scaler.inverse_transform(prediction)
        predictions.append(prediction[0, 0])

        # Update the current sequence with the new prediction
        new_data_point = np.zeros((1, 1, current_sequence.shape[2]))  # Create a placeholder for the next input
        new_data_point[0, 0, 0] = prediction[0, 0]  # Add the predicted value to the new sequence
        current_sequence = np.append(current_sequence[:, 1:, :], new_data_point, axis=1)

    return predictions

def multivariate_predict(model, data, feature_columns, scaler, prediction_days):
    """
    Predict stock price for a future day using multiple features (multivariate prediction).
    """
    last_sequence = data[-prediction_days:][feature_columns].values
    last_sequence = scaler.transform(last_sequence)
    last_sequence = last_sequence.reshape(1, prediction_days, len(feature_columns))

    prediction = model.predict(last_sequence)

    reshaped_prediction = np.zeros((1, len(feature_columns)))
    reshaped_prediction[0, 0] = prediction[0, 0]  # Assign the predicted closing price

    # Perform inverse scaling with all features (closing price will be used)
    return scaler.inverse_transform(reshaped_prediction)[0, 0]


def multivariate_multistep_predict(model, data, feature_columns, scaler, prediction_days, steps):
    """
    Predict stock prices for multiple days into the future using multiple features (multivariate and multistep prediction).
    """
    predictions = []
    last_sequence = data[-prediction_days:][feature_columns].values
    last_sequence = scaler.transform(last_sequence)
    current_sequence = last_sequence.reshape(1, prediction_days, len(feature_columns))

    for _ in range(steps):
        prediction = model.predict(current_sequence)

        reshaped_prediction = np.zeros((1, len(feature_columns)))
        reshaped_prediction[0, 3] = prediction[0, 0]


        prediction_rescaled = scaler.inverse_transform(reshaped_prediction)

        # Store the predicted closing price
        predictions.append(prediction_rescaled[0, 3])

        # Update the current sequence for the next step
        new_data_point = np.zeros((1, 1, current_sequence.shape[2]))
        new_data_point[0, 0, 3] = prediction[0, 0]
        current_sequence = np.append(current_sequence[:, 1:, :], new_data_point, axis=1)

    return predictions

