# Extending the predict_next_day function to predict multiple steps for backtesting
import numpy as np
def predict_next_days(model, last_sequence, scaler, prediction_days, num_steps):
    """
    Predict stock prices for multiple future days based on the last sequence of data.

    Parameters:
    - model: Trained model to use for predictions.
    - last_sequence: The last sequence of data used for prediction.
    - scaler: Scaler object used for scaling the data.
    - prediction_days: The number of days used in each prediction.
    - num_steps: The number of future steps (days) to predict.

    Returns:
    - predictions: A list of predicted prices for the next 'num_steps' days.
    """
    predictions = []
    current_sequence = last_sequence[-prediction_days:]  # Start with the last known sequence

    for _ in range(num_steps):
        # Predict the next day based on the current sequence
        prediction = model.predict(current_sequence.reshape(1, prediction_days, -1))
        predicted_price = scaler.inverse_transform(prediction)[0, 0]
        predictions.append(predicted_price)

        # Update the current sequence by appending the predicted value and removing the oldest value
        current_sequence = np.append(current_sequence[1:], prediction)

    return predictions

# Example usage:
# predictions = predict_next_days(model, last_sequence, scaler, prediction_days=30, num_steps=10)
# This will predict stock prices for the next 10 days based on a 30-day window.