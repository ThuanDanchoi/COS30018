import yfinance as yf
import backtrader as bt
import matplotlib.pyplot as plt
from model_operations import build_model, train_model
from predictor import predict_next_day
from data_processing import prepare_data
import numpy as np


# Create a Backtrader strategy class
class StockPredictionStrategy(bt.Strategy):
    params = (
        ('prediction_days', 60),
        ('layer_type', 'GRU'),
        ('layer_size', 100),
        ('dropout_rate', 0.3),
        ('num_layers', 4),
    )

    def __init__(self):
        self.data_close = self.data.close
        self.model = None
        self.x_train = None
        self.y_train = None
        self.scalers = None
        self.predicted_price = None
        self.FEATURE_COLUMNS = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]

    def start(self):
        # Fetch historical data of Apple from 2020 to 2024
        data = yf.download('AAPL', start='2020-01-01', end='2024-12-31')
        x_train, y_train, _, _, scalers = prepare_data(data, self.FEATURE_COLUMNS, self.params.prediction_days)

        # Build and train the model
        self.model = build_model((x_train.shape[1], len(self.FEATURE_COLUMNS)),
                                 num_layers=self.params.num_layers,
                                 layer_type=self.params.layer_type,
                                 layer_size=self.params.layer_size,
                                 dropout_rate=self.params.dropout_rate)
        train_model(self.model, x_train, y_train)

        self.x_train = x_train
        self.y_train = y_train
        self.scalers = scalers

    def next(self):
        if not self.position:  # Not in position
            # Predict the next day price
            last_sequence = self.x_train[-1].reshape(1, self.params.prediction_days, len(self.FEATURE_COLUMNS))
            self.predicted_price = predict_next_day(self.model, last_sequence, self.scalers["Close"],
                                                    self.params.prediction_days)

            # Buy if predicted price is greater than the current close price
            if self.predicted_price > self.data_close[0]:
                self.buy()
        elif self.predicted_price < self.data_close[0]:  # Sell if predicted price is less than the current close price
            self.sell()

cerebro = bt.Cerebro()
cerebro.addstrategy(StockPredictionStrategy)

# Convert data to Backtrader format
apple_data = yf.download('AAPL', start='2020-01-01', end='2024-12-31')
data_feed = bt.feeds.PandasData(dataname=apple_data)


cerebro.adddata(data_feed)


cerebro.run()


cerebro.plot(style='candlestick')