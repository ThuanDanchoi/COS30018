# File: model_operations.py
# Purpose: This module contains functions related to building, training,
# and testing the stock prediction model. It defines the LSTM model architecture,
# trains it on the processed data, and tests its performance on unseen data.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU, SimpleRNN, Bidirectional
import numpy as np
import pandas as pd

def build_model(input_shape, num_layers=3, layer_type='LSTM', layer_size=50, dropout_rate=0.2):
    model = Sequential()

    # Add the first RNN layer (LSTM, GRU, or SimpleRNN)
    if layer_type == 'LSTM':
        model.add(LSTM(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'GRU':
        model.add(GRU(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'RNN':
        model.add(SimpleRNN(units=layer_size, return_sequences=(num_layers > 1), input_shape=input_shape))
    elif layer_type == 'BiLSTM':
        model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(num_layers > 1)), input_shape=input_shape))
    elif layer_type == 'BiGRU':
        model.add(Bidirectional(GRU(units=layer_size, return_sequences=(num_layers > 1)), input_shape=input_shape))
    else:
        raise ValueError(f"Unsupported layer_type: {layer_type}")

    model.add(Dropout(dropout_rate))

    # Add the remaining RNN layers
    for _ in range(1, num_layers):
        if layer_type == 'LSTM':
            model.add(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'GRU':
            model.add(GRU(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'RNN':
            model.add(SimpleRNN(units=layer_size, return_sequences=(_ < num_layers - 1)))
        elif layer_type == 'BiLSTM':
            model.add(Bidirectional(LSTM(units=layer_size, return_sequences=(_ < num_layers - 1))))
        elif layer_type == 'BiGRU':
            model.add(Bidirectional(GRU(units=layer_size, return_sequences=(_ < num_layers - 1))))

        model.add(Dropout(dropout_rate))

    # Add the output layer
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

