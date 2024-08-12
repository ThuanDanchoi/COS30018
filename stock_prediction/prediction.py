# File: prediction.py
# Purpose: This module focuses on making predictions using the trained model.
# It includes functions to transform input data, make predictions,
# and inverse-transform the results to their original scale.

import numpy as np
from sklearn.preprocessing import MinMaxScaler
