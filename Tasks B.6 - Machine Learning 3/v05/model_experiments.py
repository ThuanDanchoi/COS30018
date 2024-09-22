# File: model_experiments.py
# Purpose: Test LSTM, GRU, and RNN models with various hyperparameters for stock prediction.
# Saves results (loss, time) to `model_experiment_results.csv`.

from data_processing import load_data, prepare_data
from model_operations import build_model, train_model
import time
import pandas as pd

# Model configurations
model_types = ['LSTM', 'GRU', 'RNN']
layers_config = [2, 3, 4]  # Number of layers
units_config = [50, 100, 150]  # Number of units in each layer
epochs_config = [25, 50]  # Number of epochs
batch_size_config = [32, 64]  # Batch size

# Results to store the outcome of each experiment
results = []

# Load and prepare data
COMPANY = 'CBA.AX'
TRAIN_START, TRAIN_END = '2020-01-01', '2023-08-01'
TEST_START, TEST_END = '2023-08-02', '2024-07-02'
FEATURE_COLUMNS = ["Close", "Volume"]
PREDICTION_DAYS = 60
NAN_METHOD, FILL_VALUE = 'ffill', 0
SPLIT_METHOD = 'ratio'
SPLIT_RATIO = 0.8
RANDOM_SPLIT = False
USE_CACHE = True
CACHE_DIR = 'data_cache'

data = load_data(COMPANY, TRAIN_START, TRAIN_END, nan_handling=NAN_METHOD, fill_value=FILL_VALUE,
                 cache_dir=CACHE_DIR, use_cache=USE_CACHE)
x_train, y_train, x_test, y_test, scalers = prepare_data(data, FEATURE_COLUMNS, PREDICTION_DAYS,
                                                         split_method=SPLIT_METHOD, split_ratio=SPLIT_RATIO,
                                                         random_split=RANDOM_SPLIT)

# Loop over configurations
for model_type in model_types:
    for num_layers in layers_config:
        for layer_size in units_config:
            for epochs in epochs_config:
                for batch_size in batch_size_config:
                    print(f"Training {model_type} with {num_layers} layers, {layer_size} units, {epochs} epochs, batch size {batch_size}")

                    # Build and train model
                    input_shape = (x_train.shape[1], x_train.shape[2])
                    model = build_model(input_shape=input_shape, num_layers=num_layers, layer_type=model_type, layer_size=layer_size)

                    # Measure training time
                    start_time = time.time()
                    trained_model = train_model(model, x_train, y_train, epochs=epochs, batch_size=batch_size)
                    training_time = time.time() - start_time

                    # Evaluate model on validation data
                    train_loss = trained_model.evaluate(x_train, y_train, verbose=0)
                    val_loss = trained_model.evaluate(x_test, y_test, verbose=0)

                    # Save results
                    results.append({
                        'Model Type': model_type,
                        'Layers': num_layers,
                        'Units per Layer': layer_size,
                        'Epochs': epochs,
                        'Batch Size': batch_size,
                        'Training Loss': train_loss,
                        'Validation Loss': val_loss,
                        'Time Taken (s)': training_time
                    })

data = pd.DataFrame(results)

data.to_csv("model_experiment_results.csv", index=False)

print(data)


