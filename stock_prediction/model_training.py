# File: model_training.py
# Purpose: This module is responsible for building, compiling, and training
# the machine learning models, particularly LSTM models, to predict stock prices.
# It also includes functions for scaling the data and saving the trained model.

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler
