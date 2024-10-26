import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
from xgboost import XGBRegressor

from data_processing_B7 import load_data, prepare_data, load_macro_data
from model_operations import train_random_forest_model, test_random_forest_model

# === CONFIGURATION PARAMETERS ===
COMPANY = 'CBA.AX'
TRAIN_START, TRAIN_END = '2020-01-01', '2023-08-01'
TEST_START, TEST_END = '2023-08-02', '2024-07-02'
PREDICTION_DAYS = 60
FEATURE_COLUMNS = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
NAN_METHOD, FILL_VALUE = 'ffill', 0
SPLIT_METHOD = 'ratio'
SPLIT_RATIO = 0.8
SPLIT_DATE = '2023-01-01'
RANDOM_SPLIT = False
USE_CACHE = True
CACHE_DIR = 'data_cache'

warnings.filterwarnings("ignore", message="X does not have valid feature names")


# === DATA LOADING AND PREPARATION ===
def load_and_prepare_data():
    """
    Load stock data, macroeconomic data, and prepare the dataset for model training.
    """
    data = load_data(COMPANY, TRAIN_START, TRAIN_END, nan_handling=NAN_METHOD, fill_value=FILL_VALUE,
                     cache_dir=CACHE_DIR, use_cache=USE_CACHE)

    macro_data = load_macro_data()

    # Resample macroeconomic data to daily frequency and forward-fill missing values
    macro_data = macro_data.resample('D').ffill()

    # Combine stock data with macroeconomic data
    data = data.join(macro_data, how='left')
    data = data.fillna(method='ffill')

    print(f"Columns in data after joining macroeconomic data: {data.columns}")

    # Prepare data for training
    x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                             split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO,
                                                             split_date=SPLIT_DATE, random_split=RANDOM_SPLIT)
    return x_train, y_train, x_test, y_test, scalers, data


# === MODEL TRAINING FUNCTIONS ===
def train_xgboost(x_train, y_train):
    """
    Train the XGBoost model.
    """
    print("Training XGBoost model...")
    xgb_model = XGBRegressor(n_estimators=500, learning_rate=0.01, max_depth=6, verbosity=1)
    xgb_model.fit(x_train.reshape(x_train.shape[0], x_train.shape[1] * len(FEATURE_COLUMNS)), y_train)
    print("XGBoost training completed.")
    return xgb_model


def train_random_forest(x_train, y_train):
    """
    Train the Random Forest model.
    """
    print("Training Random Forest model...")
    rf_model = train_random_forest_model(x_train.reshape(x_train.shape[0], x_train.shape[1] * len(FEATURE_COLUMNS)),
                                         y_train)
    print("Random Forest training completed.")
    return rf_model


# === PLOTTING FUNCTION ===
def plot_predictions(y_test_unscaled, xgb_predictions, rf_predictions):
    """
    Plot the actual prices vs. predicted prices from XGBoost and Random Forest models.
    """
    plt.figure(figsize=(12, 7))

    # Actual prices
    plt.plot(y_test_unscaled, color="black", linewidth=2, label=f"Actual {COMPANY} Price")

    # XGBoost predictions
    plt.plot(xgb_predictions, color="blue", linestyle='--', label="XGBoost Predictions")

    # Random Forest predictions
    plt.plot(rf_predictions, color="red", linestyle='-', label="Random Forest")

    # Add title and labels
    plt.title(f"{COMPANY} Share Price Prediction", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel(f"{COMPANY} Share Price", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True)

    # Adjust y-axis
    plt.ylim([min(y_test_unscaled.min(), xgb_predictions.min(), rf_predictions.min()) - 10,
              max(y_test_unscaled.max(), xgb_predictions.max(), rf_predictions.max()) + 10])

    # Show plot
    plt.show()


# === POLAR CHART FUNCTION FOR RSI & MACD ===
def plot_rsi_macd(data):
    """
    Plot a polar chart for RSI and MACD indicators.
    """
    rsi_values = data['RSI'].values
    macd_values = data['MACD'].values
    theta = np.linspace(0, 2 * np.pi, len(rsi_values))

    plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)
    ax.plot(theta, rsi_values, linestyle='--', color='green', label="RSI")
    ax.plot(theta, macd_values, linestyle='-', color='orange', label="MACD")

    plt.title("RSI & MACD Indicators (Polar Chart)")
    plt.legend(loc="upper right")
    plt.show()


# === EXPORT RSI & MACD TO CSV ===
def export_csv(data):
    """
    Export RSI and MACD values to a CSV file.
    """
    rsi_values = data['RSI'].values
    macd_values = data['MACD'].values

    # Create DataFrame to store RSI and MACD values
    rsi_macd_df = pd.DataFrame({
        'Date': data.index,
        'RSI': rsi_values,
        'MACD': macd_values
    })

    # Save to CSV
    rsi_macd_df.to_csv('rsi_macd_values.csv', index=False)
    print("RSI and MACD values have been saved to 'rsi_macd_values.csv'.")


# === MAIN FUNCTION ===
if __name__ == "__main__":
    x_train, y_train, x_test, y_test, scalers, data = load_and_prepare_data()

    xgb_model = train_xgboost(x_train, y_train)
    rf_model = train_random_forest(x_train, y_train)

    print("Making predictions with XGBoost...")
    xgb_predictions = xgb_model.predict(x_test.reshape(x_test.shape[0], x_test.shape[1] * len(FEATURE_COLUMNS)))
    xgb_predictions = scalers["Close"].inverse_transform(xgb_predictions.reshape(-1, 1))

    print("Making predictions with Random Forest...")
    rf_predictions = test_random_forest_model(rf_model,
                                              x_test.reshape(x_test.shape[0], x_test.shape[1] * len(FEATURE_COLUMNS)))
    rf_predictions = scalers["Close"].inverse_transform(rf_predictions.reshape(-1, 1))

    # Inverse transform y_test to match the unscaled predictions
    y_test_unscaled = scalers['Close'].inverse_transform(y_test.reshape(-1, 1))

    plot_predictions(y_test_unscaled, xgb_predictions, rf_predictions)

    # Export RSI & MACD values to CSV
    export_csv(data)

    # Plot RSI and MACD Polar Chart
    plot_rsi_macd(data)


