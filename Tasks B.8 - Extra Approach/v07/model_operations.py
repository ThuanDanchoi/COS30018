# File: model_operations.py
# Purpose: This module contains functions related to building, training,
# and testing the stock prediction model. It defines the LSTM model architecture,
# trains it on the processed data, and tests its performance on unseen data.

import statsmodels.api as sm
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
from sklearn.ensemble import RandomForestRegressor


def build_model(input_shape, num_layers=3, layer_type='LSTM', layer_size=50, dropout_rate=0.2, bidirectional=False):
    model = Sequential()

    # First RNN layer
    if bidirectional:
        if layer_type == 'LSTM':
            model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape)))
        elif layer_type == 'GRU':
            model.add(Bidirectional(GRU(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape)))
    else:
        if layer_type == 'LSTM':
            model.add(LSTM(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
        elif layer_type == 'GRU':
            model.add(GRU(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))

    model.add(Dropout(dropout_rate))

    # Add remaining layers
    for _ in range(1, num_layers):
        if bidirectional:
            if layer_type == 'LSTM':
                model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1))))
            elif layer_type == 'GRU':
                model.add(Bidirectional(GRU(units=layer_size, return_sequences=(_ < num_layers - 1))))
        else:
            if layer_type == 'LSTM':
                model.add(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1)))
            elif layer_type == 'GRU':
                model.add(GRU(units=layer_size, return_sequences=(_ < num_layers - 1)))

        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


def train_model(model, x_train, y_train, epochs=25, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    return model


def test_model(model, data, scaler, prediction_days, price_value):
    total_dataset = pd.concat((data[price_value]), axis=0)
    model_inputs = total_dataset[len(total_dataset) - len(data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    return predicted_prices


# ARIMA and Samira Model Functions
def train_arima_model(data, order=(6, 1, 0)):
    model = sm.tsa.ARIMA(data, order=order)
    arima_result = model.fit()
    return arima_result


def train_sarima_model(data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = sm.tsa.statespace.SARIMAX(data, order=order, seasonal_order=seasonal_order)
    sarima_result = model.fit()
    return sarima_result


def predict_arima(arima_model, steps):
    arima_predictions = arima_model.forecast(steps=steps)
    return arima_predictions


def predict_sarima(sarima_model, steps):
    sarima_predictions = sarima_model.forecast(steps=steps)
    return sarima_predictions


# Random Forest Model Functions with Hyperparameter Tuning
def train_random_forest_model(x_train, y_train, n_estimators=100, max_depth=10, min_samples_split=5):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                     min_samples_split=min_samples_split)
    rf_model.fit(x_train, y_train)
    return rf_model


def test_random_forest_model(rf_model, x_test):
    rf_predictions = rf_model.predict(x_test)
    return rf_predictions
