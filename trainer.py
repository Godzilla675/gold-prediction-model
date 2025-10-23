"""
Training pipeline for gold price prediction models.
Handles data preparation, model training, and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime

from config import (SEQUENCE_LENGTH, VALIDATION_SPLIT, TEST_SPLIT, 
                    ACCURACY_THRESHOLD, MODELS_DIR, PLOTS_DIR)
from data_fetcher import GoldDataFetcher
from feature_engineering import FeatureEngineer
from models import LSTMModel, GRUModel, XGBoostModel, ProphetModel, EnsembleModel, BaseModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_style('whitegrid')


class ModelTrainer:
    """Handles training and evaluation of gold price prediction models."""
    
    def __init__(self, data_df, target_col='Close'):
        """
        Initialize trainer with data.
        
        Args:
            data_df: DataFrame with features and target
            target_col: Name of target column
        """
        self.data_df = data_df
        self.target_col = target_col
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.models = {}
        self.results = {}
        
    def prepare_data(self):
        """Prepare train, validation, and test sets."""
        logger.info("Preparing train/val/test splits...")
        
        # Separate features and target
        feature_cols = [col for col in self.data_df.columns if col != self.target_col]
        X = self.data_df[feature_cols].values
        y = self.data_df[self.target_col].values.reshape(-1, 1)
        
        # Calculate split indices
        total_len = len(X)
        test_size = int(total_len * TEST_SPLIT)
        val_size = int(total_len * VALIDATION_SPLIT)
        train_size = total_len - test_size - val_size
        
        # Split data chronologically
        X_train = X[:train_size]
        X_val = X[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        
        y_train = y[:train_size]
        y_val = y[train_size:train_size + val_size]
        y_test = y[train_size + val_size:]
        
        # Scale data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        X_val_scaled = self.scaler_X.transform(X_val)
        X_test_scaled = self.scaler_X.transform(X_test)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train)
        y_val_scaled = self.scaler_y.transform(y_val)
        y_test_scaled = self.scaler_y.transform(y_test)
        
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train_scaled
        self.y_val = y_val_scaled
        self.y_test = y_test_scaled
        
        # Store original values for evaluation
        self.y_train_orig = y_train
        self.y_val_orig = y_val
        self.y_test_orig = y_test
        
        logger.info(f"Train set: {X_train_scaled.shape}, Val set: {X_val_scaled.shape}, Test set: {X_test_scaled.shape}")
        
    def prepare_sequences(self, X, y, sequence_length=SEQUENCE_LENGTH):
        """Create sequences for LSTM/GRU models."""
        X_seq, y_seq = [], []
        for i in range(len(X) - sequence_length):
            X_seq.append(X[i:i + sequence_length])
            y_seq.append(y[i + sequence_length])
        return np.array(X_seq), np.array(y_seq)
        
    def train_lstm(self):
        """Train LSTM model."""
        logger.info("Training LSTM model...")
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(self.X_train, self.y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(self.X_val, self.y_val)
        
        # Build and train model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        lstm_model = LSTMModel(input_shape=input_shape)
        lstm_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        
        self.models['LSTM'] = lstm_model
        return lstm_model
        
    def train_gru(self):
        """Train GRU model."""
        logger.info("Training GRU model...")
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(self.X_train, self.y_train)
        X_val_seq, y_val_seq = self.prepare_sequences(self.X_val, self.y_val)
        
        # Build and train model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        gru_model = GRUModel(input_shape=input_shape)
        gru_model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        
        self.models['GRU'] = gru_model
        return gru_model
        
    def train_xgboost(self):
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")
        
        xgb_model = XGBoostModel()
        xgb_model.train(self.X_train, self.y_train.ravel(), 
                       self.X_val, self.y_val.ravel())
        
        self.models['XGBoost'] = xgb_model
        return xgb_model
        
    def train_all_models(self):
        """Train all models."""
        logger.info("Training all models...")
        
        self.train_lstm()
        self.train_gru()
        self.train_xgboost()
        
        logger.info("All models trained successfully")
        
    def evaluate_model(self, model_name, X_test, y_test_orig):
        """Evaluate a single model."""
        logger.info(f"Evaluating {model_name} model...")
        
        model = self.models[model_name]
        
        # Handle sequence models differently
        if model_name in ['LSTM', 'GRU']:
            X_test_seq, y_test_seq = self.prepare_sequences(X_test, self.y_test)
            y_pred_scaled = model.predict(X_test_seq)
            
            # Get corresponding original values
            y_test_eval = y_test_orig[SEQUENCE_LENGTH:]
        else:
            y_pred_scaled = model.predict(X_test).reshape(-1, 1)
            y_test_eval = y_test_orig
            
        # Inverse transform predictions
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        y_true = y_test_eval.flatten()
        
        # Calculate metrics
        metrics = model.evaluate(y_true, y_pred)
        
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'actuals': y_true
        }
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  MAE: ${metrics['MAE']:.2f}")
        logger.info(f"  RMSE: ${metrics['RMSE']:.2f}")
        logger.info(f"  MAPE: {metrics['MAPE']:.2f}%")
        logger.info(f"  R²: {metrics['R2']:.4f}")
        logger.info(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")
        
        return metrics
        
    def evaluate_all_models(self):
        """Evaluate all trained models."""
        logger.info("\n" + "="*60)
        logger.info("EVALUATING ALL MODELS ON TEST SET")
        logger.info("="*60)
        
        for model_name in self.models.keys():
            self.evaluate_model(model_name, self.X_test, self.y_test_orig)
            logger.info("-"*60)
            
    def get_best_model(self, metric='MAPE'):
        """Get the best performing model based on a metric."""
        if not self.results:
            raise ValueError("No results available. Evaluate models first.")
            
        best_model_name = None
        best_score = float('inf') if metric in ['MAE', 'RMSE', 'MAPE'] else float('-inf')
        
        for model_name, result in self.results.items():
            score = result['metrics'][metric]
            
            if metric in ['MAE', 'RMSE', 'MAPE']:
                if score < best_score:
                    best_score = score
                    best_model_name = model_name
            else:  # R2, Directional_Accuracy
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    
        logger.info(f"\nBest model: {best_model_name} (with {metric}={best_score:.4f})")
        return best_model_name, best_score
        
    def check_accuracy_threshold(self):
        """Check if any model meets the 90% accuracy threshold."""
        logger.info("\n" + "="*60)
        logger.info("CHECKING ACCURACY THRESHOLD (90%)")
        logger.info("="*60)
        
        meets_threshold = False
        
        for model_name, result in self.results.items():
            mape = result['metrics']['MAPE']
            accuracy = 100 - mape  # Accuracy = 100 - MAPE
            directional_acc = result['metrics']['Directional_Accuracy']
            
            logger.info(f"{model_name}:")
            logger.info(f"  Price Accuracy: {accuracy:.2f}%")
            logger.info(f"  Directional Accuracy: {directional_acc:.2f}%")
            
            if accuracy >= ACCURACY_THRESHOLD * 100:
                logger.info(f"  ✓ Meets threshold!")
                meets_threshold = True
            else:
                logger.info(f"  ✗ Does not meet threshold")
                
        return meets_threshold
        
    def plot_results(self):
        """Plot predictions vs actuals for all models."""
        n_models = len(self.results)
        fig, axes = plt.subplots(n_models, 1, figsize=(15, 5*n_models))
        
        if n_models == 1:
            axes = [axes]
            
        for idx, (model_name, result) in enumerate(self.results.items()):
            ax = axes[idx]
            y_true = result['actuals']
            y_pred = result['predictions']
            
            ax.plot(y_true, label='Actual', linewidth=2)
            ax.plot(y_pred, label='Predicted', linewidth=2, alpha=0.7)
            ax.set_title(f'{model_name} - Predictions vs Actual', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Gold Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        filepath = PLOTS_DIR / f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Prediction plots saved to {filepath}")
        plt.close()
        
    def save_models(self):
        """Save all trained models."""
        logger.info("Saving all models...")
        for model_name, model in self.models.items():
            model.save()
        logger.info("All models saved successfully")
        
    def generate_report(self):
        """Generate a comprehensive evaluation report."""
        logger.info("\n" + "="*80)
        logger.info("GOLD PRICE PREDICTION MODEL - EVALUATION REPORT")
        logger.info("="*80)
        
        # Summary table
        logger.info("\nMODEL COMPARISON SUMMARY:")
        logger.info("-"*80)
        logger.info(f"{'Model':<15} {'MAE':<10} {'RMSE':<10} {'MAPE':<10} {'R²':<10} {'Dir.Acc':<10}")
        logger.info("-"*80)
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            logger.info(f"{model_name:<15} "
                       f"${metrics['MAE']:<9.2f} "
                       f"${metrics['RMSE']:<9.2f} "
                       f"{metrics['MAPE']:<9.2f}% "
                       f"{metrics['R2']:<9.4f} "
                       f"{metrics['Directional_Accuracy']:<9.2f}%")
        
        logger.info("-"*80)
        
        # Best model
        best_model, best_score = self.get_best_model(metric='MAPE')
        logger.info(f"\nBEST MODEL: {best_model}")
        
        # Threshold check
        meets_threshold = self.check_accuracy_threshold()
        
        if meets_threshold:
            logger.info("\n✓ SUCCESS: At least one model meets the 90% accuracy threshold!")
        else:
            logger.info("\n✗ WARNING: No model meets the 90% accuracy threshold.")
            logger.info("   Consider: More data, feature engineering, or architecture changes.")
            
        logger.info("="*80)


def main():
    """Main training pipeline."""
    logger.info("Starting Gold Price Prediction Training Pipeline")
    
    # Step 1: Fetch data
    logger.info("\nStep 1: Fetching gold price data...")
    fetcher = GoldDataFetcher()
    raw_data = fetcher.fetch_data()
    fetcher.save_data()
    
    # Step 2: Feature engineering
    logger.info("\nStep 2: Creating features...")
    engineer = FeatureEngineer(raw_data)
    features_df = engineer.create_all_features()
    
    # Step 3: Train models
    logger.info("\nStep 3: Training models...")
    trainer = ModelTrainer(features_df)
    trainer.prepare_data()
    trainer.train_all_models()
    
    # Step 4: Evaluate models
    logger.info("\nStep 4: Evaluating models...")
    trainer.evaluate_all_models()
    
    # Step 5: Generate report and plots
    logger.info("\nStep 5: Generating reports...")
    trainer.generate_report()
    trainer.plot_results()
    
    # Step 6: Save models
    trainer.save_models()
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING PIPELINE COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
