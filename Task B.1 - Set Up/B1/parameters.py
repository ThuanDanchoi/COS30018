import os
import time
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import Huber

# Window size or the sequence length (number of previous days to use for prediction)
N_STEPS = 50

# Lookup step, 1 is the next day (number of days in the future to predict)
LOOKUP_STEP = 15

# whether to scale feature columns & output price as well
SCALE = True
scale_str = f"sc-{int(SCALE)}"

# whether to shuffle the dataset
SHUFFLE = True
shuffle_str = f"sh-{int(SHUFFLE)}"

# whether to split the training/testing set by date
SPLIT_BY_DATE = False
split_by_date_str = f"sbd-{int(SPLIT_BY_DATE)}"

# test ratio size, 0.2 means 20% of the data will be used for testing
TEST_SIZE = 0.2

# features to use, "adjclose" is the adjusted closing price, "volume", "open", "high", and "low" are self-explanatory
FEATURE_COLUMNS = ["adjclose", "volume", "open", "high", "low"]

# date now (used for naming files)
date_now = time.strftime("%Y-%m-%d")

### Model parameters
# Number of LSTM layers
N_LAYERS = 2

# LSTM cell
CELL = LSTM

# Number of LSTM units (neurons) per layer
UNITS = 256

# Dropout rate to prevent overfitting (percentage of neurons to drop)
DROPOUT = 0.4

# whether to use bidirectional RNNs
BIDIRECTIONAL = False

### Training parameters
# Loss function

LOSS = Huber()


# Optimizer
OPTIMIZER = "adam"

# Batch size (number of samples per gradient update)
BATCH_SIZE = 64

# Number of training epochs (how many times the model will see the entire dataset)
EPOCHS = 100

# The stock ticker symbol you want to predict (Amazon stock in this case)
ticker = "AMZN"

# The filename where the data for the selected ticker will be saved
ticker_data_filename = os.path.join("data", f"{ticker}_{date_now}.csv")

# Model name to save, making it as unique as possible based on parameters
model_name = f"{date_now}_{ticker}-{shuffle_str}-{scale_str}-{split_by_date_str}-\
{LOSS}-{OPTIMIZER}-{CELL.__name__}-seq-{N_STEPS}-step-{LOOKUP_STEP}-layers-{N_LAYERS}-units-{UNITS}"

# Add "-b" to the model name if using bidirectional LSTMs
if BIDIRECTIONAL:
    model_name += "-b"
