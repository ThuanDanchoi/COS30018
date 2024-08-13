# File: main.py
# Purpose: This is the main script that orchestrates the entire workflow.
# It calls the necessary functions from other modules to load data,
# train the model, make predictions, and visualize the results.

from data_loading import load_data
from model_training import build_model, train_model
from prediction import predict_next_day
from visualization import plot_results
