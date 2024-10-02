from data_processing_B7 import load_data, prepare_data, load_macro_data
from model_operations import build_model, train_model, test_model, train_random_forest_model, test_random_forest_model
from xgboost import XGBRegressor
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

# Prepare data for training
x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                         split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO,
                                                         split_date=SPLIT_DATE, random_split=RANDOM_SPLIT)

# Ensure the date index is in datetime format
data.index = pd.to_datetime(data.index)

# Reindex the data to ensure it has a continuous daily date range and fill missing dates
full_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
data = data.reindex(full_range).interpolate(method='linear')

# Train XGBoost model
xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6)
xgb_model.fit(x_train.reshape(x_train.shape[0], x_train.shape[1] * len(FEATURE_COLUMNS)), y_train)

# Make predictions with XGBoost
xgb_predictions = xgb_model.predict(x_test.reshape(x_test.shape[0], x_test.shape[1] * len(FEATURE_COLUMNS)))

# Inverse transform XGBoost predictions
xgb_predictions = scalers["Close"].inverse_transform(xgb_predictions.reshape(-1, 1))

# Train Random Forest Model
rf_model = train_random_forest_model(x_train.reshape(x_train.shape[0], x_train.shape[1] * len(FEATURE_COLUMNS)), y_train)
rf_predictions = test_random_forest_model(rf_model, x_test.reshape(x_test.shape[0], x_test.shape[1] * len(FEATURE_COLUMNS)))
rf_predictions = scalers["Close"].inverse_transform(rf_predictions.reshape(-1, 1))

# Inverse transform actual values
y_test_unscaled = scalers['Close'].inverse_transform(y_test.reshape(-1, 1))

# Plot XGBoost predictions and actual prices
plt.figure(figsize=(12, 7))
plt.plot(y_test_unscaled, color="black", linewidth=2, label=f"Actual {COMPANY} Price")
plt.plot(xgb_predictions, color="blue", linestyle='--', label="XGBoost Predictions")
plt.plot(rf_predictions, color="red", linestyle='-', label="Random Forest")
plt.title(f"{COMPANY} Share Price Prediction", fontsize=14)
plt.xlabel("Time", fontsize=12)
plt.ylabel(f"{COMPANY} Share Price", fontsize=12)
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

# Assuming RSI and MACD are already calculated in prepare_data function
# Extract RSI and MACD values for plotting
rsi = data['RSI']
macd = data['MACD']

# Plot RSI and MACD using a polar chart
plt.figure(figsize=(12, 7))
ax = plt.subplot(111, projection='polar')

# Polar chart for RSI
ax.plot(np.linspace(0, 2*np.pi, len(rsi)), rsi, color="green", linestyle='-', label="RSI")

# Polar chart for MACD
ax.plot(np.linspace(0, 2*np.pi, len(macd)), macd, color="orange", linestyle='-', label="MACD")

plt.title(f"RSI & MACD Indicators (Polar Chart)", fontsize=14)
plt.legend(loc="upper right")
plt.show()

rsi_values = data['RSI'].values
macd_values = data['MACD'].values

# Tạo DataFrame để hiển thị gọn gàng hơn
rsi_macd_df = pd.DataFrame({
    'Date': data.index,
    'RSI': rsi_values,
    'MACD': macd_values
})

# In ra bảng giá trị RSI và MACD
print("Bảng giá trị RSI và MACD:")
print(rsi_macd_df)




