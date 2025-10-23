"""
Module to fetch gold price data from free APIs.
Using Yahoo Finance (yfinance) as it's free and has reliable historical data.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from config import GOLD_TICKER, ALTERNATIVE_TICKER, START_DATE, DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoldDataFetcher:
    """Fetches gold price data from Yahoo Finance."""
    
    def __init__(self, ticker=GOLD_TICKER, start_date=START_DATE):
        self.ticker = ticker
        self.start_date = start_date
        self.data = None
        
    def fetch_data(self, end_date=None, use_mock=False):
        """
        Fetch gold price data from Yahoo Finance.
        
        Args:
            end_date: End date for data fetching. If None, uses today.
            use_mock: If True, use mock data instead of real API
            
        Returns:
            pandas.DataFrame: Gold price data
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching gold price data from {self.start_date} to {end_date}")
        
        try:
            if use_mock:
                raise ValueError("Using mock data as requested")
                
            # Try primary ticker (Gold Futures)
            gold_data = yf.download(self.ticker, start=self.start_date, end=end_date, progress=False)
            
            if gold_data.empty or len(gold_data) < 100:
                logger.warning(f"Insufficient data from {self.ticker}, trying alternative ticker {ALTERNATIVE_TICKER}")
                gold_data = yf.download(ALTERNATIVE_TICKER, start=self.start_date, end=end_date, progress=False)
            
            if gold_data.empty:
                raise ValueError("No data retrieved from Yahoo Finance")
                
            # Flatten multi-level columns if present
            if isinstance(gold_data.columns, pd.MultiIndex):
                gold_data.columns = gold_data.columns.get_level_values(0)
                
            # Sort by date
            gold_data = gold_data.sort_index()
            
            logger.info(f"Successfully fetched {len(gold_data)} days of data")
            logger.info(f"Date range: {gold_data.index[0]} to {gold_data.index[-1]}")
            
            self.data = gold_data
            return gold_data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            logger.warning("Falling back to mock data for testing/demo purposes")
            
            # Use mock data as fallback
            try:
                from mock_data import generate_comprehensive_mock_data
                gold_data = generate_comprehensive_mock_data()
                
                # Filter by date range if needed
                gold_data = gold_data[gold_data.index >= self.start_date]
                if end_date:
                    gold_data = gold_data[gold_data.index <= end_date]
                
                logger.info(f"Successfully generated {len(gold_data)} days of mock data")
                logger.info(f"Date range: {gold_data.index[0]} to {gold_data.index[-1]}")
                
                self.data = gold_data
                return gold_data
                
            except Exception as mock_error:
                logger.error(f"Error generating mock data: {str(mock_error)}")
                raise
            
    def save_data(self, filename='gold_prices.csv'):
        """Save fetched data to CSV file."""
        if self.data is None:
            raise ValueError("No data to save. Fetch data first.")
            
        filepath = DATA_DIR / filename
        self.data.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
        
    def load_data(self, filename='gold_prices.csv'):
        """Load data from CSV file."""
        filepath = DATA_DIR / filename
        self.data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Data loaded from {filepath}")
        return self.data
        
    def get_latest_price(self):
        """Get the most recent gold price."""
        if self.data is None:
            raise ValueError("No data available. Fetch data first.")
            
        latest_data = self.data.iloc[-1]
        return {
            'date': self.data.index[-1],
            'close': latest_data['Close'],
            'open': latest_data['Open'],
            'high': latest_data['High'],
            'low': latest_data['Low'],
            'volume': latest_data['Volume']
        }


def main():
    """Test data fetching functionality."""
    fetcher = GoldDataFetcher()
    data = fetcher.fetch_data()
    
    print("\n=== Gold Price Data Summary ===")
    print(f"Total records: {len(data)}")
    print(f"\nFirst 5 records:\n{data.head()}")
    print(f"\nLast 5 records:\n{data.tail()}")
    print(f"\nData statistics:\n{data.describe()}")
    
    # Save data
    fetcher.save_data()
    
    # Get latest price
    latest = fetcher.get_latest_price()
    print(f"\n=== Latest Gold Price ===")
    print(f"Date: {latest['date']}")
    print(f"Close: ${latest['close']:.2f}")
    print(f"High: ${latest['high']:.2f}")
    print(f"Low: ${latest['low']:.2f}")


if __name__ == "__main__":
    main()
