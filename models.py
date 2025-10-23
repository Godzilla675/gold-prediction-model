"""
Machine learning models for gold price prediction.
Implements LSTM, GRU, XGBoost, Prophet, and Ensemble methods.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import xgboost as xgb
from prophet import Prophet
import logging
import joblib
from config import (SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE, LEARNING_RATE,
                    EARLY_STOPPING_PATIENCE, MODELS_DIR)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all prediction models."""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = MinMaxScaler()
        self.trained = False
        
    def prepare_sequences(self, X, y, sequence_length=SEQUENCE_LENGTH):
        """Prepare sequences for time series models."""
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)
        
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        r2 = r2_score(y_true, y_pred)
        
        # Directional accuracy
        direction_true = np.diff(y_true) > 0
        direction_pred = np.diff(y_pred) > 0
        directional_accuracy = np.mean(direction_true == direction_pred) * 100
        
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2,
            'Directional_Accuracy': directional_accuracy
        }
        
    def save(self, filepath):
        """Save model to disk."""
        raise NotImplementedError
        
    def load(self, filepath):
        """Load model from disk."""
        raise NotImplementedError


class LSTMModel(BaseModel):
    """LSTM-based model for gold price prediction."""
    
    def __init__(self, input_shape, name="LSTM"):
        super().__init__(name)
        self.input_shape = input_shape
        self.build_model()
        
    def build_model(self):
        """Build LSTM architecture."""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=self.input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='huber',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Built {self.name} model")
        return model
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train the LSTM model."""
        logger.info(f"Training {self.name} model...")
        
        callbacks = [
            EarlyStopping(patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        self.trained = True
        logger.info(f"{self.name} training complete")
        return history
        
    def predict(self, X):
        """Make predictions."""
        if not self.trained and self.model is None:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X, verbose=0)
        
    def save(self, filepath=None):
        """Save model."""
        if filepath is None:
            filepath = MODELS_DIR / f"{self.name.lower()}_model.h5"
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath=None):
        """Load model."""
        if filepath is None:
            filepath = MODELS_DIR / f"{self.name.lower()}_model.h5"
        self.model = keras.models.load_model(filepath)
        self.trained = True
        logger.info(f"Model loaded from {filepath}")


class GRUModel(LSTMModel):
    """GRU-based model for gold price prediction."""
    
    def __init__(self, input_shape, name="GRU"):
        self.input_shape = input_shape
        BaseModel.__init__(self, name)
        self.build_model()
        
    def build_model(self):
        """Build GRU architecture."""
        model = Sequential([
            Bidirectional(GRU(128, return_sequences=True), input_shape=self.input_shape),
            Dropout(0.2),
            Bidirectional(GRU(64, return_sequences=True)),
            Dropout(0.2),
            GRU(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='huber',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Built {self.name} model")
        return model


class XGBoostModel(BaseModel):
    """XGBoost model for gold price prediction."""
    
    def __init__(self, name="XGBoost"):
        super().__init__(name)
        self.model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=7,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model."""
        logger.info(f"Training {self.name} model...")
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        self.trained = True
        logger.info(f"{self.name} training complete")
        
    def predict(self, X):
        """Make predictions."""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)
        
    def save(self, filepath=None):
        """Save model."""
        if filepath is None:
            filepath = MODELS_DIR / f"{self.name.lower()}_model.joblib"
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath=None):
        """Load model."""
        if filepath is None:
            filepath = MODELS_DIR / f"{self.name.lower()}_model.joblib"
        self.model = joblib.load(filepath)
        self.trained = True
        logger.info(f"Model loaded from {filepath}")


class ProphetModel(BaseModel):
    """Facebook Prophet model for gold price prediction."""
    
    def __init__(self, name="Prophet"):
        super().__init__(name)
        self.model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
    def train(self, df, target_col='Close'):
        """Train Prophet model."""
        logger.info(f"Training {self.name} model...")
        
        # Prepare data in Prophet format
        prophet_df = pd.DataFrame({
            'ds': df.index,
            'y': df[target_col].values
        })
        
        self.model.fit(prophet_df)
        self.trained = True
        logger.info(f"{self.name} training complete")
        
    def predict(self, periods=1):
        """Make predictions."""
        if not self.trained:
            raise ValueError("Model must be trained before prediction")
            
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast['yhat'].values
        
    def save(self, filepath=None):
        """Save model."""
        if filepath is None:
            filepath = MODELS_DIR / f"{self.name.lower()}_model.joblib"
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
        
    def load(self, filepath=None):
        """Load model."""
        if filepath is None:
            filepath = MODELS_DIR / f"{self.name.lower()}_model.joblib"
        self.model = joblib.load(filepath)
        self.trained = True
        logger.info(f"Model loaded from {filepath}")


class EnsembleModel(BaseModel):
    """Ensemble of multiple models."""
    
    def __init__(self, models, weights=None, name="Ensemble"):
        super().__init__(name)
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)
        
    def train(self, *args, **kwargs):
        """Train all models in ensemble."""
        logger.info(f"Training {self.name} model (all sub-models)...")
        for model in self.models:
            if hasattr(model, 'train'):
                model.train(*args, **kwargs)
        self.trained = True
        
    def predict(self, X):
        """Make ensemble predictions."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
            
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return ensemble_pred


def main():
    """Test models."""
    logger.info("Models module loaded successfully")
    logger.info("Available models: LSTM, GRU, XGBoost, Prophet, Ensemble")


if __name__ == "__main__":
    main()
