# File: visualization.py
# Purpose: This module introduces advanced data visualization techniques,
# including candlestick and boxplot charts, to enhance the analysis of stock market data.
# These visualizations complement the predictive model by providing deeper insights into
# historical price movements and distributions.

import mplfinance as mpf
def plot_candlestick_chart(data, n_days=1, title="Candlestick Chart"):
    """
    Plots a candlestick chart of the stock data.
    """
    # Resampling data to aggregate into n_days candlesticks
    data_resampled = data.resample(f'{n_days}D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    # Plotting the candlestick chart
    mpf.plot(data_resampled, type='candle', volume=True, title=title, style='charles')

def plot_boxplot(data, feature='Close', n_days=20, title="Boxplot Chart"):
    """
    Plots a boxplot chart of the stock data.
    """
    # Rolling window for the specified feature
    rolling_data = data[feature].rolling(window=n_days)

    # Plotting the boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot(rolling_data, notch=True, patch_artist=True)
    plt.title(title)
    plt.ylabel(f'{feature} Price')
    plt.show()


