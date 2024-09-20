import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

def plot_candlestick_chart(data, company, n_days=1):
    """
    Plot a candlestick chart for the given stock market data.
    Each candlestick represents 'n_days' of trading data.
    """
    if n_days > 1:
        data_resampled = data.resample(f'{n_days}D').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    else:
        data_resampled = data

    mpf.plot(data_resampled, type='candle', title=f'{company} Candlestick Chart', style='charles', volume=True)
    plt.show()

def plot_boxplot(data, company, column='Close', n_days=1):
    """
    Plot multiple boxplot charts for the given stock market data.
    Each boxplot shows the distribution of data over a moving window of 'n_days'.
    """
    rolling_data = data[column].rolling(window=n_days).mean().dropna()
    boxplot_data = [rolling_data[i:i + n_days] for i in range(0, len(rolling_data), n_days)]

    plt.figure(figsize=(12, 6))
    plt.title(f'{company} Boxplot Chart')
    plt.boxplot(boxplot_data, patch_artist=True, showmeans=True)
    plt.xlabel(f'Rolling {n_days}-Day Period')
    plt.ylabel('Closing Price')
    plt.xticks(ticks=range(1, len(boxplot_data) + 1), labels=[f'{i*n_days+1}-{(i+1)*n_days}' for i in range(len(boxplot_data))])
    plt.grid(True)
    plt.show()

