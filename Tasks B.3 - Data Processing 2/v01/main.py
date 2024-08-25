# File: main.py
# Purpose: This is the main script that orchestrates the entire stock prediction workflow.
# It integrates various components, including data loading, preprocessing,
# model building, training, testing, and prediction. The script is responsible for
# coordinating these tasks, generating visualizations, and outputting predictions.

from data_processing import load_data, prepare_data
from model_operations import build_model, train_model, test_model
from predictor import predict_next_day
from visualization import plot_candlestick_chart, plot_boxplot

import matplotlib.pyplot as plt

# COMPANY: The stock ticker symbol of the company to analyze.
COMPANY = 'CBA.AX'

# TRAIN_START, TRAIN_END: The start and end dates for the training data.
TRAIN_START, TRAIN_END = '2020-01-01', '2023-08-01'

# TEST_START, TEST_END: The start and end dates for the testing data.
TEST_START, TEST_END = '2023-08-02', '2024-07-02'

# PREDICTION_DAYS: Number of days to look back when creating input sequences for prediction.
PREDICTION_DAYS = 60

# FEATURE_COLUMNS: List of features (columns) from the dataset to be scaled and used for training.
FEATURE_COLUMNS = ["Close", "Volume"]

# NAN_METHOD: Method for handling missing data (NaN values). Options: 'drop', 'fill', 'ffill', 'bfill'.
# FILL_VALUE: Value to use for filling NaNs if 'fill' method is selected.
NAN_METHOD, FILL_VALUE = 'ffill', 0

# SPLIT_METHOD: Method for splitting the dataset into training and testing sets. Options: 'ratio', 'date'.
# SPLIT_RATIO: Ratio of the data to be used for training if 'ratio' method is selected.
# SPLIT_DATE: Specific date to split the data if 'date' method is selected.
SPLIT_METHOD = 'ratio'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2023-01-01'

# RANDOM_SPLIT: Whether to split the data randomly (True) or sequentially (False).
RANDOM_SPLIT = False

# USE_CACHE: Whether to load data from a local cache if available, to save time on downloading.
USE_CACHE = True

# CACHE_DIR: Directory where cached data will be stored and loaded from.
CACHE_DIR = 'data_cache'

# Load and prepare data with caching, scaling, and splitting
data = load_data(COMPANY, TRAIN_START, TRAIN_END, nan_handling=NAN_METHOD, fill_value=FILL_VALUE,
                 cache_dir=CACHE_DIR, use_cache=USE_CACHE)
x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                         split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO,
                                                         split_date=SPLIT_DATE, random_split=RANDOM_SPLIT)

# Build, train, and test model
model = build_model((x_train.shape[1], len(FEATURE_COLUMNS)))
train_model(model, x_train, y_train)
predicted_prices = model.predict(x_test)
predicted_prices = scalers["Close"].inverse_transform(predicted_prices)

# Assuming y_test is not inverse transformed and predicted_prices is:
y_test_unscaled = scalers["Close"].inverse_transform(y_test.reshape(-1, 1))

# Plot the candlestick chart for the data
plot_candlestick_chart(data, n_days=5, title=f"{COMPANY} Candlestick Chart")

# Plot the boxplot for the 'Close' feature
plot_boxplot(data, feature='Close', n_days=20, title=f"{COMPANY} Boxplot Chart")

# Plot results
plt.plot(y_test_unscaled, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

# Predict next day
last_sequence = x_test[-1].reshape(1, PREDICTION_DAYS, len(FEATURE_COLUMNS))
prediction = predict_next_day(model, last_sequence, scalers["Close"], PREDICTION_DAYS)
print(f"Prediction: {prediction}")
