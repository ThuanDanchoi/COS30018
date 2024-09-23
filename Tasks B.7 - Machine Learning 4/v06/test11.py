from test1 import load_data, prepare_data, load_macro_data
from model_operations import (build_model, train_model, test_model, train_arima_model, predict_arima,
                              train_sarima_model, predict_sarima, train_random_forest_model, test_random_forest_model)
from predictor import predict_next_day, multistep_predict, multivariate_predict, multivariate_multistep_predict
from visualization import plot_candlestick_chart, plot_boxplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# COMPANY: The stock ticker symbol of the company to analyze.
COMPANY = 'CBA.AX'

# TRAIN_START, TRAIN_END: The start and end dates for the training data.
TRAIN_START, TRAIN_END = '2020-01-01', '2023-08-01'

# TEST_START, TEST_END: The start and end dates for the testing data.
TEST_START, TEST_END = '2023-08-02', '2024-07-02'

# PREDICTION_DAYS: Number of days to look back when creating input sequences for prediction.
PREDICTION_DAYS = 60

# FEATURE_COLUMNS: List of features (columns) from the dataset to be scaled and used for training.
FEATURE_COLUMNS = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]

# NAN_METHOD: Method for handling missing data (NaN values). Options: 'drop', 'fill', 'ffill', 'bfill'.
NAN_METHOD, FILL_VALUE = 'ffill', 0

# SPLIT_METHOD: Method for splitting the dataset into training and testing sets. Options: 'ratio', 'date'.
SPLIT_METHOD = 'ratio'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2023-01-01'

# RANDOM_SPLIT: Whether to split the data randomly (True) or sequentially (False).
RANDOM_SPLIT = False

# USE_CACHE: Whether to load data from a local cache if available, to save time on downloading.
USE_CACHE = True

# CACHE_DIR: Directory where cached data will be stored and loaded from.
CACHE_DIR = 'data_cache'

# Load stock data
data = load_data(COMPANY, TRAIN_START, TRAIN_END, nan_handling=NAN_METHOD, fill_value=FILL_VALUE,
                 cache_dir=CACHE_DIR, use_cache=USE_CACHE)

# Load macroeconomic data
macro_data = load_macro_data()

# Resample macroeconomic data to daily frequency and forward-fill missing values
macro_data = macro_data.resample('D').ffill()

# Combine stock data with macroeconomic data
data = data.join(macro_data, how='left')

# Debug: Print columns to verify that GDP and other macroeconomic data are present
print("Columns in data after joining macroeconomic data:", data.columns)

# Fill any remaining missing values
data = data.fillna(method='ffill')

# Chuẩn bị dữ liệu, bao gồm cả các chỉ báo kỹ thuật và các cột kinh tế vĩ mô (GDP, Inflation, Unemployment)
x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                         split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO,
                                                         split_date=SPLIT_DATE, random_split=RANDOM_SPLIT)

# Ensure the date index is in datetime format
data.index = pd.to_datetime(data.index)

# Reindex the data to ensure it has a continuous daily date range and fill missing dates
full_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
data = data.reindex(full_range).interpolate(method='linear')  # Use interpolation to fill missing values

# Debug: Print sample data to check if everything is correct
print(data.head())

# Now the 'Close' column has a regular daily frequency
arima_model = train_arima_model(data['Close'])
sarima_model = train_sarima_model(data['Close'])

# ARIMA and SARIMA Predictions
arima_predictions = predict_arima(arima_model, len(y_test))
sarima_predictions = predict_sarima(sarima_model, len(y_test))

# Build and train LSTM/GRU model
model = build_model((x_train.shape[1], len(FEATURE_COLUMNS)), num_layers=4, layer_type='GRU', layer_size=100, dropout_rate=0.3, bidirectional=True)
train_model(model, x_train, y_train)
predicted_prices = model.predict(x_test)
predicted_prices = scalers["Close"].inverse_transform(predicted_prices)

# Assuming y_test is not inverse transformed and predicted_prices is:
y_test_unscaled = scalers["Close"].inverse_transform(y_test.reshape(-1, 1))

# Flatten the LSTM/GRU predicted prices (if not already 1D)
predicted_prices_flat = predicted_prices.flatten()

# Convert ARIMA and SARIMA predictions to NumPy arrays (since they're Series)
arima_predictions_flat = arima_predictions.values
sarima_predictions_flat = sarima_predictions.values

# Calculate model errors to determine weights
arima_mse = np.mean((arima_predictions_flat - y_test_unscaled.flatten())**2)
sarima_mse = np.mean((sarima_predictions_flat - y_test_unscaled.flatten())**2)
lstm_mse = np.mean((predicted_prices_flat - y_test_unscaled.flatten())**2)
total_mse = arima_mse + sarima_mse + lstm_mse

# Assign weights inversely proportional to MSE: from lower MSE to higher weight
arima_weight = (1 - arima_mse / total_mse)
sarima_weight = (1 - sarima_mse / total_mse)
lstm_weight = (1 - lstm_mse / total_mse)

# Normalize weights
arima_weight /= (arima_weight + sarima_weight + lstm_weight)
sarima_weight /= (arima_weight + sarima_weight + lstm_weight)
lstm_weight /= (arima_weight + sarima_weight + lstm_weight)

# Combine ARIMA, SARIMA, and LSTM/GRU predictions
combined_predictions = (arima_weight * arima_predictions_flat) + \
                       (sarima_weight * sarima_predictions_flat) + \
                       (lstm_weight * predicted_prices_flat)

# Train Random Forest Model
rf_model = train_random_forest_model(x_train.reshape(x_train.shape[0], x_train.shape[1] * len(FEATURE_COLUMNS)), y_train)
rf_predictions = test_random_forest_model(rf_model, x_test.reshape(x_test.shape[0], x_test.shape[1] * len(FEATURE_COLUMNS)))

# Inverse transform Random Forest predictions
rf_predictions = scalers["Close"].inverse_transform(rf_predictions.reshape(-1, 1))

# Plot combined predictions and actual prices
plt.figure(figsize=(12, 7))
plt.plot(y_test_unscaled, color="black", linewidth=2, label=f"Actual {COMPANY} Price")
plt.plot(combined_predictions, color="blue", linewidth=2, linestyle='--', label="Combined (ARIMA, SARIMA, LSTM/GRU)")
plt.plot(predicted_prices_flat, color="green", linestyle='--', label="LSTM/GRU")
plt.plot(arima_predictions_flat, color="orange", linestyle=':', label="ARIMA")
plt.plot(sarima_predictions_flat, color="purple", linestyle='-.', label="SARIMA")
plt.plot(rf_predictions, color="red", linestyle='-', label="Random Forest")

plt.title(f"{COMPANY} Share Price Prediction", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel(f"{COMPANY} Share Price", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
