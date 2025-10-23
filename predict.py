"""
Gold price prediction script.
Uses trained models to predict future gold prices.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import joblib

from config import MODELS_DIR, SEQUENCE_LENGTH
from data_fetcher import GoldDataFetcher
from feature_engineering import FeatureEngineer
from models import LSTMModel, GRUModel, XGBoostModel
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class GoldPricePredictor:
    """Predict future gold prices using trained models."""
    
    def __init__(self):
        self.models = {}
        self.scaler_X = None
        self.scaler_y = None
        self.feature_columns = None
        
    def load_models(self):
        """Load all trained models."""
        logger.info("Loading trained models...")
        
        try:
            # Try to load XGBoost model
            xgb_model = XGBoostModel()
            xgb_model.load()
            self.models['XGBoost'] = xgb_model
            logger.info("Loaded XGBoost model")
        except Exception as e:
            logger.warning(f"Could not load XGBoost model: {e}")
            
        try:
            # Try to load LSTM model
            lstm_model = LSTMModel(input_shape=(SEQUENCE_LENGTH, 1))
            lstm_model.load()
            self.models['LSTM'] = lstm_model
            logger.info("Loaded LSTM model")
        except Exception as e:
            logger.warning(f"Could not load LSTM model: {e}")
            
        try:
            # Try to load GRU model
            gru_model = GRUModel(input_shape=(SEQUENCE_LENGTH, 1))
            gru_model.load()
            self.models['GRU'] = gru_model
            logger.info("Loaded GRU model")
        except Exception as e:
            logger.warning(f"Could not load GRU model: {e}")
            
        if not self.models:
            raise ValueError("No models could be loaded. Train models first using trainer.py")
            
    def load_scalers(self):
        """Load saved scalers."""
        try:
            scaler_path_X = MODELS_DIR / "scaler_X.joblib"
            scaler_path_y = MODELS_DIR / "scaler_y.joblib"
            
            if scaler_path_X.exists() and scaler_path_y.exists():
                self.scaler_X = joblib.load(scaler_path_X)
                self.scaler_y = joblib.load(scaler_path_y)
                logger.info("Loaded scalers")
            else:
                logger.warning("Scalers not found. Will create new ones.")
                self.scaler_X = MinMaxScaler()
                self.scaler_y = MinMaxScaler()
        except Exception as e:
            logger.warning(f"Could not load scalers: {e}")
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            
    def prepare_latest_data(self):
        """Fetch and prepare the latest gold price data."""
        logger.info("Fetching latest gold price data...")
        
        # Fetch data
        fetcher = GoldDataFetcher()
        raw_data = fetcher.fetch_data()
        
        # Create features
        engineer = FeatureEngineer(raw_data)
        features_df = engineer.create_all_features()
        
        self.feature_columns = [col for col in features_df.columns if col != 'Close']
        
        return features_df
        
    def predict_next_price(self, model_name='XGBoost'):
        """
        Predict the next gold price.
        
        Args:
            model_name: Name of model to use ('LSTM', 'GRU', 'XGBoost')
            
        Returns:
            dict: Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
            
        # Prepare data
        features_df = self.prepare_latest_data()
        
        # Get latest data point
        X_latest = features_df[self.feature_columns].iloc[-SEQUENCE_LENGTH:].values
        
        # Scale data
        if self.scaler_X is not None:
            X_scaled = self.scaler_X.fit_transform(X_latest)
        else:
            X_scaled = X_latest
            
        model = self.models[model_name]
        
        # Make prediction based on model type
        if model_name in ['LSTM', 'GRU']:
            X_seq = X_scaled.reshape(1, SEQUENCE_LENGTH, -1)
            y_pred_scaled = model.predict(X_seq)
        else:  # XGBoost
            X_seq = X_scaled[-1].reshape(1, -1)
            y_pred_scaled = model.predict(X_seq).reshape(-1, 1)
            
        # Inverse transform
        if self.scaler_y is not None:
            # Fit scaler on historical close prices
            self.scaler_y.fit(features_df['Close'].values.reshape(-1, 1))
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)[0][0]
        else:
            y_pred = y_pred_scaled[0][0]
            
        # Get current price
        current_price = features_df['Close'].iloc[-1]
        
        # Calculate change
        price_change = y_pred - current_price
        price_change_pct = (price_change / current_price) * 100
        
        result = {
            'model': model_name,
            'current_price': current_price,
            'predicted_price': y_pred,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'prediction_date': datetime.now() + timedelta(days=1),
            'current_date': features_df.index[-1]
        }
        
        return result
        
    def predict_all_models(self):
        """Get predictions from all loaded models."""
        logger.info("\n" + "="*60)
        logger.info("GOLD PRICE PREDICTIONS")
        logger.info("="*60)
        
        results = {}
        for model_name in self.models.keys():
            try:
                result = self.predict_next_price(model_name)
                results[model_name] = result
                
                logger.info(f"\n{model_name} Model:")
                logger.info(f"  Current Price: ${result['current_price']:.2f}")
                logger.info(f"  Predicted Price: ${result['predicted_price']:.2f}")
                logger.info(f"  Change: ${result['price_change']:.2f} ({result['price_change_pct']:+.2f}%)")
                logger.info(f"  Prediction for: {result['prediction_date'].strftime('%Y-%m-%d')}")
                
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                
        # Ensemble prediction (average of all models)
        if len(results) > 1:
            avg_prediction = np.mean([r['predicted_price'] for r in results.values()])
            current_price = list(results.values())[0]['current_price']
            
            logger.info(f"\nEnsemble (Average) Prediction:")
            logger.info(f"  Predicted Price: ${avg_prediction:.2f}")
            logger.info(f"  Change: ${avg_prediction - current_price:.2f} "
                       f"({((avg_prediction - current_price) / current_price * 100):+.2f}%)")
                       
        logger.info("="*60)
        
        return results


def main():
    """Main prediction script."""
    try:
        predictor = GoldPricePredictor()
        predictor.load_models()
        predictor.load_scalers()
        results = predictor.predict_all_models()
        
        logger.info("\n✓ Prediction complete!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.info("\nPlease train models first using: python trainer.py")


if __name__ == "__main__":
    main()
