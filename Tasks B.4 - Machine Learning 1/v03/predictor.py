# File: predictor.py
# Purpose: This module is responsible for making predictions using the trained
# model. It processes the most recent data and outputs the predicted stock price
# for the next day.

import numpy as np


def predict_next_day(model, last_sequence, scaler, prediction_days):
    real_data = last_sequence[-prediction_days:]
    print("real_data shape before reshape:", real_data.shape)

    prediction = model.predict(real_data)

    prediction = scaler.inverse_transform(prediction)
    return prediction

