import time
import numpy as np
import datetime as dt
import yfinance as yf

def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))

    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    rsi = []
    for i in range(period, len(prices)):
        avg_gain = (avg_gain * (period - 1) + gain[i - period]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - period]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi_value = 100 - (100 / (1 + rs))
        rsi.append(rsi_value)

    return rsi

def trade_finance():
    ticker = 'AAPL'
    data = yf.download(ticker, period='1mo', interval='1h')  # Download 1 month of hourly data

    # Columns to use
    columns = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
    selected_data = data[columns].to_numpy()  # Convert selected columns to NumPy array

    # Extract individual columns
    high_prices = selected_data[:, 0]  # High prices
    low_prices = selected_data[:, 1]  # Low prices
    open_prices = selected_data[:, 2]  # Open prices
    close_prices = selected_data[:, 3]  # Close prices
    volume = selected_data[:, 4]  # Volume
    adj_close_prices = selected_data[:, 5]  # Adjusted Close prices

    # Parameters for SMA calculation
    sma_short_period = 10
    sma_long_period = 20
    rsi_period = 14

    # Stop-loss and take-profit parameters (percentage)
    stop_loss_pct = 0.02  # 2% stop loss
    take_profit_pct = 0.04  # 4% take profit

    # Variables for simulation
    current_position = None
    cash = 100000  # Initial virtual cash
    stock_qty = 0
    entry_price = 0
    start_time = dt.datetime.now()
    max_duration = dt.timedelta(minutes=5)  # Simulate for 5 minutes
    index = 0
    total_profit = 0  # Track total profit or loss

    while True:
        # Check if the script has run for longer than the max duration
        if dt.datetime.now() - start_time > max_duration:
            print("Stopping the script after 5 minutes.")
            break

        # Get the current price
        if index < len(close_prices):
            current_price = close_prices[index]
            index += 1
        else:
            print("End of historical price data.")
            break

        # Extract Close Prices for SMA Calculation (take the last `sma_long_period` prices)
        if index >= sma_long_period:
            recent_prices = close_prices[index - sma_long_period:index]

            # Calculate Moving Averages
            sma_short = np.mean(recent_prices[-sma_short_period:])
            sma_long = np.mean(recent_prices)

            # Calculate RSI
            if index >= sma_long_period + rsi_period:
                rsi_values = calculate_rsi(close_prices[index - (sma_long_period + rsi_period):index], rsi_period)
                rsi = rsi_values[-1]
            else:
                continue

            # Debug information for moving averages and RSI
            print(f"SMA Short: {sma_short}, SMA Long: {sma_long}, RSI: {rsi}")

            # Buy Signal with Relaxed RSI Confirmation
            if sma_short > sma_long and current_position is None and rsi < 40:
                qty = int(cash // current_price)
                if qty > 0:
                    stock_qty = qty
                    cash -= stock_qty * current_price
                    entry_price = current_price
                    current_position = 'long'
                    print(f"Bought {stock_qty} shares at {current_price}, Remaining Cash: {cash}")

            # Stop-loss or Take-profit check
            if current_position == 'long':
                # Stop-loss triggered
                if current_price <= entry_price * (1 - stop_loss_pct):
                    cash += stock_qty * current_price
                    profit = (current_price - entry_price) * stock_qty
                    total_profit += profit
                    print(f"Stop-Loss Triggered: Sold {stock_qty} shares at {current_price}, Profit: {profit}, New Cash Balance: {cash}")
                    stock_qty = 0
                    current_position = None

                # Take-profit triggered
                elif current_price >= entry_price * (1 + take_profit_pct):
                    cash += stock_qty * current_price
                    profit = (current_price - entry_price) * stock_qty
                    total_profit += profit
                    print(f"Take-Profit Triggered: Sold {stock_qty} shares at {current_price}, Profit: {profit}, New Cash Balance: {cash}")
                    stock_qty = 0
                    current_position = None

            # Sell Signal with Relaxed RSI Confirmation
            elif sma_short < sma_long and current_position == 'long' and rsi > 60:
                cash += stock_qty * current_price
                profit = (current_price - entry_price) * stock_qty
                total_profit += profit
                print(f"Sold {stock_qty} shares at {current_price}, Profit: {profit}, New Cash Balance: {cash}")
                stock_qty = 0
                current_position = None

            # Debug profit/loss tracking
            print(f"Total Profit/Loss so far: {total_profit}")

        time.sleep(1)

    # Final evaluation of performance
    if current_position == 'long':
        # Sell any remaining stock at the last available price
        cash += stock_qty * current_price
        profit = (current_price - entry_price) * stock_qty
        total_profit += profit
        print(f"Sold {stock_qty} shares at {current_price}, Profit: {profit}, New Cash Balance: {cash}")
        stock_qty = 0
        current_position = None

    print(f"Total Profit/Loss at the end of simulation: {total_profit}")

trade_finance()
