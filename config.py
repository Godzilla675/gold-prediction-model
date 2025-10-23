"""
Configuration file for gold price prediction model.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PLOTS_DIR = BASE_DIR / "plots"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data configuration
GOLD_TICKER = "GC=F"  # Gold Futures (Yahoo Finance)
ALTERNATIVE_TICKER = "GLD"  # SPDR Gold Shares ETF (more stable data)
START_DATE = "2015-01-01"  # Get 9+ years of historical data
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# Model configuration
SEQUENCE_LENGTH = 60  # Use 60 days of data to predict next day
FORECAST_HORIZON = 1  # Predict 1 day ahead
ACCURACY_THRESHOLD = 0.90  # 90% accuracy requirement

# Training configuration
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15

# Feature engineering
TECHNICAL_INDICATORS = [
    'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_10', 'EMA_20',
    'RSI_14',
    'MACD', 'MACD_signal',
    'BB_upper', 'BB_middle', 'BB_lower',
    'ATR_14',
    'OBV'
]

# Model types to evaluate
MODELS_TO_EVALUATE = ['LSTM', 'GRU', 'XGBoost', 'Prophet', 'Ensemble']

# Evaluation metrics
METRICS = ['MAE', 'RMSE', 'MAPE', 'R2', 'Directional_Accuracy']
