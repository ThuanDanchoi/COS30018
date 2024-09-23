# **Stock Price Prediction System**

## Overview

This project implements a stock price prediction system using machine learning techniques, specifically LSTM neural networks. The system fetches historical stock data, processes it, trains a predictive model, and visualizes the predicted vs. actual stock prices.

## Prerequisites

Before running the code, ensure you have the following installed:

+ Python 3.12
+ Virtual environment (recommended)
+ Required Python libraries listed in requirements.txt

## Setup Instructions
1. Clone the Repository:
  ```bash
git clone <repository_url>
```
2. Create and Activate a Virtual Environment:
```bash
python -m venv stock_prediction_env
source stock_prediction_env/bin/activate   # On Windows: stock_prediction_env\Scripts\activate

```
3. Install Required Libraries:
 ```bash
pip install -r requirements.txt
```
## Running the Code
1. Run the Main Script:
 ```bash
python main.py

 ```
This script will:
+ Load and preprocess stock data.
+ Train the LSTM model.
+ Make predictions on test data.
+ Visualize the predicted and actual stock prices.

2. Customize Parameters:
+ Modify data_loading.py, model_training.py, or other modules to change stock symbols, training periods, model architecture, etc.


## Project Structure
+ `data_loading.py` Handles data fetching and preprocessing.
+ `model_training.py`: Builds and trains the LSTM model.
+ `prediction.py`: Generates predictions using the trained model.
+ `visualization.py`: Visualizes the prediction results.
+ `main.py`: Orchestrates the full workflow.