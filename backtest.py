#!/usr/bin/env python3
"""
Backtesting script to validate model accuracy on historical data.
Tests if models can predict current prices from old data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

from data_fetcher import GoldDataFetcher
from feature_engineering import FeatureEngineer
from trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def backtest_predictions(cutoff_date='2024-01-01', prediction_days=30):
    """
    Test model predictions by training on data up to cutoff_date
    and predicting the next prediction_days.
    
    Args:
        cutoff_date: Date to split train/test
        prediction_days: Number of days to predict
    """
    logger.info("="*80)
    logger.info("BACKTESTING GOLD PRICE PREDICTIONS")
    logger.info("="*80)
    
    # Fetch all data
    logger.info("\n1. Fetching historical data...")
    fetcher = GoldDataFetcher()
    all_data = fetcher.fetch_data()
    
    # Split at cutoff date
    train_data = all_data[all_data.index < cutoff_date]
    test_data = all_data[all_data.index >= cutoff_date].head(prediction_days)
    
    logger.info(f"   Train period: {train_data.index[0]} to {train_data.index[-1]}")
    logger.info(f"   Test period:  {test_data.index[0]} to {test_data.index[-1]}")
    logger.info(f"   Test days: {len(test_data)}")
    
    # Create features on training data
    logger.info("\n2. Engineering features...")
    engineer = FeatureEngineer(train_data)
    features_df = engineer.create_all_features()
    
    # Train models
    logger.info("\n3. Training models on historical data...")
    trainer = ModelTrainer(features_df)
    trainer.prepare_data()
    
    # Train just XGBoost for speed (best performing model)
    trainer.train_xgboost()
    
    logger.info("\n4. Making predictions on test period...")
    
    # Get actual prices
    actual_prices = test_data['Close'].values
    
    # Make predictions using the trained model
    # For simplicity, we'll predict each day using the features from that day
    engineer_test = FeatureEngineer(all_data)
    features_test = engineer_test.create_all_features()
    
    # Get test features
    test_features = features_test[features_test.index >= cutoff_date].head(prediction_days)
    feature_cols = [col for col in test_features.columns if col != 'Close']
    X_test = test_features[feature_cols].values
    
    # Scale and predict
    X_test_scaled = trainer.scaler_X.transform(X_test)
    predictions_scaled = trainer.models['XGBoost'].predict(X_test_scaled)
    predictions = trainer.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(actual_prices - predictions))
    rmse = np.sqrt(np.mean((actual_prices - predictions)**2))
    mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
    accuracy = 100 - mape
    
    # Directional accuracy
    if len(actual_prices) > 1:
        direction_actual = np.diff(actual_prices) > 0
        direction_pred = np.diff(predictions) > 0
        directional_acc = np.mean(direction_actual == direction_pred) * 100
    else:
        directional_acc = 0
    
    logger.info("\n" + "="*80)
    logger.info("BACKTEST RESULTS")
    logger.info("="*80)
    logger.info(f"\nTrained on data until: {cutoff_date}")
    logger.info(f"Predicted next {len(actual_prices)} days")
    logger.info(f"\nPerformance Metrics:")
    logger.info(f"  Accuracy:              {accuracy:.2f}%")
    logger.info(f"  MAPE:                  {mape:.2f}%")
    logger.info(f"  MAE:                   ${mae:.2f}")
    logger.info(f"  RMSE:                  ${rmse:.2f}")
    logger.info(f"  Directional Accuracy:  {directional_acc:.2f}%")
    
    logger.info(f"\nSample Predictions:")
    logger.info(f"  {'Date':<12} {'Actual':<12} {'Predicted':<12} {'Error':<12}")
    logger.info(f"  {'-'*50}")
    
    for i in range(min(10, len(actual_prices))):
        date = test_data.index[i].strftime('%Y-%m-%d')
        actual = actual_prices[i]
        pred = predictions[i]
        error = pred - actual
        logger.info(f"  {date:<12} ${actual:<11.2f} ${pred:<11.2f} ${error:+11.2f}")
    
    if accuracy >= 90:
        logger.info(f"\n✓ SUCCESS: Model achieves {accuracy:.2f}% accuracy on backtest!")
    else:
        logger.info(f"\n⚠ WARNING: Model accuracy {accuracy:.2f}% is below 90% threshold.")
    
    logger.info("="*80)
    
    return {
        'accuracy': accuracy,
        'mape': mape,
        'mae': mae,
        'rmse': rmse,
        'directional_accuracy': directional_acc,
        'predictions': predictions,
        'actuals': actual_prices
    }


def main():
    """Run backtest."""
    import sys
    
    # Use different cutoff dates for testing
    cutoff_dates = ['2024-01-01', '2024-06-01', '2024-09-01']
    
    results = []
    
    for cutoff in cutoff_dates:
        try:
            result = backtest_predictions(cutoff_date=cutoff, prediction_days=20)
            results.append((cutoff, result))
        except Exception as e:
            logger.error(f"Backtest for {cutoff} failed: {e}")
    
    if results:
        logger.info("\n" + "="*80)
        logger.info("BACKTEST SUMMARY")
        logger.info("="*80)
        logger.info(f"\n{'Cutoff Date':<15} {'Accuracy':<12} {'MAPE':<10} {'MAE':<10}")
        logger.info("-"*50)
        
        for cutoff, result in results:
            logger.info(f"{cutoff:<15} {result['accuracy']:<11.2f}% "
                       f"{result['mape']:<9.2f}% ${result['mae']:<9.2f}")
        
        avg_accuracy = np.mean([r['accuracy'] for _, r in results])
        logger.info(f"\nAverage Accuracy: {avg_accuracy:.2f}%")
        
        if avg_accuracy >= 90:
            logger.info("\n✓ Models consistently achieve >90% accuracy across different periods!")
        
        logger.info("="*80)


if __name__ == "__main__":
    main()
