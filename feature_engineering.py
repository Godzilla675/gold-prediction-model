"""
Feature engineering module for gold price prediction.
Creates technical indicators and derived features.
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features from raw gold price data."""
    
    def __init__(self, data):
        """
        Initialize with gold price data.
        
        Args:
            data: pandas.DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.features_df = None
        
    def add_technical_indicators(self):
        """Add technical indicators as features."""
        logger.info("Adding technical indicators...")
        
        df = self.data.copy()
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Simple Moving Averages
        df['SMA_10'] = SMAIndicator(close=close, window=10).sma_indicator()
        df['SMA_20'] = SMAIndicator(close=close, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=close, window=50).sma_indicator()
        
        # Exponential Moving Averages
        df['EMA_10'] = EMAIndicator(close=close, window=10).ema_indicator()
        df['EMA_20'] = EMAIndicator(close=close, window=20).ema_indicator()
        
        # Relative Strength Index
        df['RSI_14'] = RSIIndicator(close=close, window=14).rsi()
        
        # MACD
        macd = MACD(close=close)
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_middle'] = bb.bollinger_mavg()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = bb.bollinger_wband()
        
        # Average True Range (Volatility)
        df['ATR_14'] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
        
        # On Balance Volume
        df['OBV'] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
        
        self.features_df = df
        logger.info(f"Added {len(df.columns) - len(self.data.columns)} technical indicators")
        return df
        
    def add_price_features(self):
        """Add price-based features."""
        logger.info("Adding price-based features...")
        
        if self.features_df is None:
            self.features_df = self.data.copy()
            
        df = self.features_df
        
        # Price changes
        df['Price_Change'] = df['Close'].diff()
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100
        
        # High-Low range
        df['Daily_Range'] = df['High'] - df['Low']
        df['Daily_Range_Pct'] = (df['Daily_Range'] / df['Close']) * 100
        
        # Gap (difference between open and previous close)
        df['Gap'] = df['Open'] - df['Close'].shift(1)
        df['Gap_Pct'] = (df['Gap'] / df['Close'].shift(1)) * 100
        
        # Position within day's range
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        self.features_df = df
        logger.info("Added price-based features")
        return df
        
    def add_lagged_features(self, lags=[1, 2, 3, 5, 7, 14, 21, 30]):
        """Add lagged features for time series."""
        logger.info(f"Adding lagged features for {len(lags)} lags...")
        
        if self.features_df is None:
            self.features_df = self.data.copy()
            
        df = self.features_df
        
        for lag in lags:
            df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
            df[f'Return_Lag_{lag}'] = df['Close'].pct_change(lag)
            
        self.features_df = df
        logger.info(f"Added {len(lags) * 3} lagged features")
        return df
        
    def add_rolling_statistics(self, windows=[5, 10, 20, 30]):
        """Add rolling statistics."""
        logger.info(f"Adding rolling statistics for windows: {windows}")
        
        if self.features_df is None:
            self.features_df = self.data.copy()
            
        df = self.features_df
        
        for window in windows:
            # Rolling mean
            df[f'Rolling_Mean_{window}'] = df['Close'].rolling(window=window).mean()
            
            # Rolling std (volatility)
            df[f'Rolling_Std_{window}'] = df['Close'].rolling(window=window).std()
            
            # Rolling min/max
            df[f'Rolling_Min_{window}'] = df['Close'].rolling(window=window).min()
            df[f'Rolling_Max_{window}'] = df['Close'].rolling(window=window).max()
            
        self.features_df = df
        logger.info(f"Added rolling statistics")
        return df
        
    def add_time_features(self):
        """Add time-based features."""
        logger.info("Adding time-based features...")
        
        if self.features_df is None:
            self.features_df = self.data.copy()
            
        df = self.features_df
        
        # Extract date components
        df['Day_of_Week'] = df.index.dayofweek
        df['Day_of_Month'] = df.index.day
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        df['Year'] = df.index.year
        
        # Cyclical encoding for day of week and month
        df['Day_of_Week_Sin'] = np.sin(2 * np.pi * df['Day_of_Week'] / 7)
        df['Day_of_Week_Cos'] = np.cos(2 * np.pi * df['Day_of_Week'] / 7)
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        self.features_df = df
        logger.info("Added time-based features")
        return df
        
    def create_all_features(self):
        """Create all features."""
        logger.info("Creating all features...")
        
        self.add_technical_indicators()
        self.add_price_features()
        self.add_lagged_features()
        self.add_rolling_statistics()
        self.add_time_features()
        
        # Drop rows with NaN values (from indicators and lags)
        initial_rows = len(self.features_df)
        self.features_df = self.features_df.dropna()
        rows_dropped = initial_rows - len(self.features_df)
        
        logger.info(f"Feature engineering complete. Dropped {rows_dropped} rows with NaN values")
        logger.info(f"Final dataset shape: {self.features_df.shape}")
        logger.info(f"Total features: {len(self.features_df.columns)}")
        
        return self.features_df
        
    def get_feature_names(self):
        """Get list of all feature names."""
        if self.features_df is None:
            return []
        return self.features_df.columns.tolist()


def main():
    """Test feature engineering."""
    from data_fetcher import GoldDataFetcher
    
    # Fetch data
    fetcher = GoldDataFetcher()
    data = fetcher.fetch_data()
    
    # Create features
    engineer = FeatureEngineer(data)
    features_df = engineer.create_all_features()
    
    print("\n=== Feature Engineering Summary ===")
    print(f"Original features: {len(data.columns)}")
    print(f"Total features: {len(features_df.columns)}")
    print(f"Dataset shape: {features_df.shape}")
    print(f"\nFeature names:\n{engineer.get_feature_names()}")
    print(f"\nSample data:\n{features_df.tail()}")


if __name__ == "__main__":
    main()
