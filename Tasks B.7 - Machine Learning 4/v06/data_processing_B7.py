from fredapi import Fred
import pandas as pd
import yfinance as yf
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(company, start_date, end_date, nan_handling='drop', fill_value=0,
              cache_dir='data_cache', use_cache=True):
    """
    Load stock data, handle NaN values, and optionally cache the data locally.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = f"{cache_dir}/{company}_{start_date}_{end_date}.csv"

    # Check if cached data exists
    if use_cache and os.path.exists(cache_file):
        data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        print(f"Loaded data from cache: {cache_file}")
    else:
        # Download the stock data
        data = yf.download(company, start_date, end_date)

        # Load macroeconomic data
        macro_data = load_macro_data()

        # Join the macroeconomic data with stock data based on the index (date)
        data = data.join(macro_data, how='left')

        # Handle NaN values
        if nan_handling == 'drop':
            data.dropna(inplace=True)
        elif nan_handling == 'fill':
            data.fillna(fill_value, inplace=True)
        elif nan_handling == 'ffill':
            data.ffill(inplace=True)
        elif nan_handling == 'bfill':
            data.bfill(inplace=True)
        else:
            raise ValueError("Invalid NaN handling method.")

        # Save data to cache
        if use_cache:
            data.to_csv(cache_file)
            print(f"Saved data to cache: {cache_file}")

    return data



def prepare_data(data, feature_columns, prediction_days, split_method='ratio',
                 split_ratio=0.8, split_date=None, random_split=False):
    """
    Prepare, scale, and split stock data for model training.
    """
    feature_columns += ['RSI', 'MACD', 'GDP', 'Inflation', 'Unemployment']


    data['RSI'] = calculate_rsi(data)
    data['RSI'] = data['RSI'].bfill()
    data['MACD'], data['MACD_Signal'] = calculate_macd(data)

    scalers = {}
    for feature in feature_columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data[feature] = scaler.fit_transform(data[feature].values.reshape(-1, 1))
        scalers[feature] = scaler

    x_data, y_data = [], []
    for x in range(prediction_days, len(data)):
        x_data.append(data[feature_columns].iloc[x - prediction_days:x].values)
        y_data.append(data['Close'].iloc[x])  # Assuming 'Close' is the target

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    if split_method == 'date' and split_date:
        split_index = data.index.get_loc(split_date)
        x_train, x_test = x_data[:split_index], x_data[split_index:]
        y_train, y_test = y_data[:split_index], y_data[split_index:]
    elif split_method == 'ratio':
        if random_split:
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=split_ratio, random_state=42)
        else:
            split_index = int(len(x_data) * split_ratio)
            x_train, x_test = x_data[:split_index], x_data[split_index:]
            y_train, y_test = y_data[:split_index], y_data[split_index:]
    else:
        raise ValueError("Invalid split method.")

    return x_train, y_train, x_test, y_test, scalers

def load_macro_data():
    fred = Fred(api_key='6101c716c394ca6a417d3e2923c66f31')

    # Download GDP, inflation, and unemployment rate
    gdp = fred.get_series('GDP', start_date='2020-01-01', end_date='2023-08-01')
    inflation = fred.get_series('CPIAUCSL', start_date='2020-01-01',
                                end_date='2023-08-01')  # CPI as proxy for inflation
    unemployment = fred.get_series('UNRATE', start_date='2020-01-01', end_date='2023-08-01')

    # Create a DataFrame
    macro_data = pd.DataFrame({
        'GDP': gdp,
        'Inflation': inflation,
        'Unemployment': unemployment
    })

    return macro_data


def calculate_rsi(data, column='Close', period=14):
    delta = data[column].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(data, column='Close', short_window=12, long_window=26, signal_window=9):
    short_ema = data[column].ewm(span=short_window, adjust=False).mean()
    long_ema = data[column].ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()

    return macd, signal
