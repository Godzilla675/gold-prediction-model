"""
Mock data generator for testing when internet is not available.
Generates realistic gold price data based on historical patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_mock_gold_data(start_date='2015-01-01', end_date=None, base_price=1200, trend=0.0001, volatility=0.02):
    """
    Generate realistic mock gold price data.
    
    Args:
        start_date: Start date for data
        end_date: End date for data (default: today)
        base_price: Starting gold price in USD
        trend: Daily trend (0.0001 = 0.01% daily increase)
        volatility: Daily volatility (0.02 = 2%)
        
    Returns:
        pandas.DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    logger.info(f"Generating mock gold price data from {start_date} to {end_date}")
    
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Generate prices with trend and noise
    np.random.seed(42)  # For reproducibility
    
    # Simulate price movement
    returns = np.random.normal(trend, volatility, n_days)
    price_multipliers = np.exp(np.cumsum(returns))
    close_prices = base_price * price_multipliers
    
    # Generate OHLV data
    data = []
    for i, date in enumerate(date_range):
        close = close_prices[i]
        
        # Generate intraday variation
        daily_volatility = np.random.uniform(0.005, 0.02)
        high = close * (1 + daily_volatility * np.random.uniform(0.3, 1.0))
        low = close * (1 - daily_volatility * np.random.uniform(0.3, 1.0))
        
        # Open price influenced by previous close
        if i == 0:
            open_price = base_price
        else:
            gap = np.random.normal(0, 0.003)
            open_price = close_prices[i-1] * (1 + gap)
        
        # Volume (higher volume on more volatile days)
        base_volume = 100000000
        volume = int(base_volume * (1 + np.random.uniform(-0.3, 0.5)))
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    logger.info(f"Generated {len(df)} days of mock data")
    logger.info(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    return df


def add_realistic_patterns(df):
    """Add more realistic patterns to mock data."""
    # Add some seasonal patterns
    df_copy = df.copy()
    
    # Simulate market cycles
    n_points = len(df_copy)
    cycle_period = 365  # Annual cycle
    
    # Add cyclical component
    cycle_effect = 0.05 * np.sin(2 * np.pi * np.arange(n_points) / cycle_period)
    df_copy['Close'] = df_copy['Close'] * (1 + cycle_effect)
    
    # Recalculate OHLV based on adjusted close
    for i in range(len(df_copy)):
        close = df_copy.iloc[i]['Close']
        orig_high = df.iloc[i]['High']
        orig_low = df.iloc[i]['Low']
        orig_close = df.iloc[i]['Close']
        
        # Adjust proportionally
        ratio = close / orig_close
        df_copy.iloc[i, df_copy.columns.get_loc('High')] = orig_high * ratio
        df_copy.iloc[i, df_copy.columns.get_loc('Low')] = orig_low * ratio
    
    return df_copy


def generate_comprehensive_mock_data():
    """Generate comprehensive mock data with realistic patterns."""
    # Generate base data
    df = generate_mock_gold_data(
        start_date='2015-01-01',
        base_price=1200,
        trend=0.00015,  # Slight upward trend
        volatility=0.015
    )
    
    # Add realistic patterns
    df = add_realistic_patterns(df)
    
    # Add some major events (price jumps/drops)
    event_dates = [
        ('2016-06-24', 0.05),   # Brexit
        ('2020-03-16', -0.08),  # COVID crash
        ('2020-08-06', 0.12),   # COVID recovery
        ('2022-03-08', 0.06),   # Ukraine conflict
    ]
    
    for event_date, impact in event_dates:
        if event_date in df.index:
            idx = df.index.get_loc(event_date)
            # Apply impact to subsequent days with decay
            for i in range(idx, min(idx + 10, len(df))):
                decay = 0.8 ** (i - idx)
                df.iloc[i, df.columns.get_loc('Close')] *= (1 + impact * decay)
    
    return df


if __name__ == "__main__":
    df = generate_comprehensive_mock_data()
    print("\n=== Mock Gold Price Data ===")
    print(f"Records: {len(df)}")
    print(f"\nFirst 5 records:\n{df.head()}")
    print(f"\nLast 5 records:\n{df.tail()}")
    print(f"\nStatistics:\n{df.describe()}")
