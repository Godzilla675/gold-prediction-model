#!/usr/bin/env python3
"""
Real benchmark test using live data.
Tests if the model can predict today's actual gold price from yesterday's data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys

from data_fetcher import GoldDataFetcher
from feature_engineering import FeatureEngineer
from trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def real_benchmark_test():
    """
    Real benchmark: Train on data up to yesterday, predict today's price.
    Compare with actual today's price.
    """
    logger.info("="*80)
    logger.info("REAL BENCHMARK TEST - LIVE DATA")
    logger.info("="*80)
    
    # Get today's date
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    # Go back further to get trading days (account for weekends)
    train_until_date = (today - timedelta(days=5)).strftime('%Y-%m-%d')
    
    logger.info(f"\nToday's date: {today.strftime('%Y-%m-%d')}")
    logger.info(f"Training data up to: {train_until_date}")
    
    # Fetch ALL available data (including most recent)
    logger.info("\n1. Fetching live gold price data from Yahoo Finance...")
    fetcher = GoldDataFetcher()
    
    try:
        # Try to fetch REAL data (not mock)
        all_data = fetcher.fetch_data(use_mock=False)
        
        if all_data is None or len(all_data) == 0:
            logger.error("Failed to fetch live data from Yahoo Finance")
            logger.info("\nIMPORTANT: Cannot perform real benchmark without live data access.")
            logger.info("The system would need internet access to yahoo.com/finance to fetch real data.")
            return None
            
        logger.info(f"   ✓ Fetched {len(all_data)} days of LIVE data")
        logger.info(f"   Date range: {all_data.index[0]} to {all_data.index[-1]}")
        
        # Get the most recent date in the data
        most_recent_date = all_data.index[-1]
        logger.info(f"\n   Most recent data available: {most_recent_date.strftime('%Y-%m-%d')}")
        
        # Check if we have today's data
        has_today = most_recent_date.date() >= (today - timedelta(days=1)).date()
        
        if not has_today:
            logger.warning(f"\n   ⚠ Most recent data is from {most_recent_date.strftime('%Y-%m-%d')}")
            logger.warning(f"   This may be due to market hours or weekend/holiday")
        
        # Split data: train on everything except last day, test on last day
        train_data = all_data.iloc[:-1]
        test_data = all_data.iloc[-1:]
        
        logger.info(f"\n2. Splitting data...")
        logger.info(f"   Training: {len(train_data)} days (up to {train_data.index[-1].strftime('%Y-%m-%d')})")
        logger.info(f"   Testing: {test_data.index[0].strftime('%Y-%m-%d')} (ACTUAL PRICE)")
        
        # Get actual price we're trying to predict
        actual_price = test_data['Close'].values[0]
        logger.info(f"   ACTUAL PRICE to predict: ${actual_price:.2f}")
        
        # Create features from training data
        logger.info(f"\n3. Engineering features from training data...")
        engineer_train = FeatureEngineer(train_data)
        features_train = engineer_train.create_all_features()
        
        # Train model (using only training data)
        logger.info(f"\n4. Training XGBoost model on historical data...")
        trainer = ModelTrainer(features_train)
        trainer.prepare_data()
        trainer.train_xgboost()
        
        logger.info(f"   ✓ Model trained successfully")
        
        # Now prepare the test data with features
        logger.info(f"\n5. Preparing test data for prediction...")
        
        # To make a prediction, we need features for the test date
        # Create features from ALL data (including test) to get proper features
        engineer_all = FeatureEngineer(all_data)
        features_all = engineer_all.create_all_features()
        
        # Get features for the last date (test date)
        test_features = features_all.iloc[-1:]
        feature_cols = [col for col in test_features.columns if col != 'Close']
        X_test = test_features[feature_cols].values
        
        # Scale using the training scaler
        X_test_scaled = trainer.scaler_X.transform(X_test)
        
        # Make prediction
        logger.info(f"\n6. Making prediction for {test_data.index[0].strftime('%Y-%m-%d')}...")
        prediction_scaled = trainer.models['XGBoost'].predict(X_test_scaled)
        predicted_price = trainer.scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        
        # Calculate error
        error = predicted_price - actual_price
        error_pct = abs(error / actual_price) * 100
        accuracy = 100 - error_pct
        
        # Results
        logger.info("\n" + "="*80)
        logger.info("REAL BENCHMARK RESULTS - LIVE DATA")
        logger.info("="*80)
        logger.info(f"\nTest Date: {test_data.index[0].strftime('%Y-%m-%d')}")
        logger.info(f"\nACTUAL PRICE:     ${actual_price:.2f}")
        logger.info(f"PREDICTED PRICE:  ${predicted_price:.2f}")
        logger.info(f"\nERROR:            ${error:+.2f}")
        logger.info(f"ERROR %:          {error_pct:.2f}%")
        logger.info(f"ACCURACY:         {accuracy:.2f}%")
        
        if accuracy >= 90:
            logger.info(f"\n✓ SUCCESS: Model achieves {accuracy:.2f}% accuracy on LIVE data!")
            logger.info("   The model meets the 90% accuracy threshold on real-world data.")
        else:
            logger.info(f"\n⚠ Model accuracy {accuracy:.2f}% on this single day")
            logger.info("   Note: Single-day accuracy can vary. Multiple days needed for robust evaluation.")
        
        logger.info("\n" + "="*80)
        
        return {
            'test_date': test_data.index[0],
            'actual_price': actual_price,
            'predicted_price': predicted_price,
            'error': error,
            'error_pct': error_pct,
            'accuracy': accuracy,
            'using_live_data': True
        }
        
    except Exception as e:
        logger.error(f"\nError during live data test: {str(e)}")
        logger.info("\nNote: This benchmark requires live internet access to Yahoo Finance.")
        logger.info("Without access, we cannot fetch real-time data for validation.")
        import traceback
        traceback.print_exc()
        return None


def multi_day_benchmark(num_days=5):
    """
    Test on multiple recent days for more robust evaluation.
    """
    logger.info("\n" + "="*80)
    logger.info(f"MULTI-DAY BENCHMARK - Testing on last {num_days} days")
    logger.info("="*80)
    
    try:
        # Fetch all data
        fetcher = GoldDataFetcher()
        all_data = fetcher.fetch_data(use_mock=False)
        
        if all_data is None or len(all_data) < 100:
            logger.error("Insufficient live data for multi-day benchmark")
            return None
        
        results = []
        
        for i in range(1, min(num_days + 1, len(all_data) - 50)):
            # Split: train on everything before day i, predict day i
            train_data = all_data.iloc[:-i]
            test_data = all_data.iloc[-i:-i+1]
            
            # Skip if test data is empty
            if len(test_data) == 0:
                continue
            
            test_date = test_data.index[0]
            actual_price = test_data['Close'].values[0]
            
            # Create features and train
            engineer = FeatureEngineer(train_data)
            features = engineer.create_all_features()
            
            trainer = ModelTrainer(features)
            trainer.prepare_data()
            trainer.train_xgboost()
            
            # Prepare test features
            engineer_all = FeatureEngineer(all_data.iloc[:-i+1])
            features_all = engineer_all.create_all_features()
            test_features = features_all.iloc[-1:]
            feature_cols = [col for col in test_features.columns if col != 'Close']
            X_test = test_features[feature_cols].values
            X_test_scaled = trainer.scaler_X.transform(X_test)
            
            # Predict
            prediction_scaled = trainer.models['XGBoost'].predict(X_test_scaled)
            predicted_price = trainer.scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
            
            error = predicted_price - actual_price
            error_pct = abs(error / actual_price) * 100
            accuracy = 100 - error_pct
            
            results.append({
                'date': test_date,
                'actual': actual_price,
                'predicted': predicted_price,
                'accuracy': accuracy
            })
            
            logger.info(f"\n{test_date.strftime('%Y-%m-%d')}: Actual=${actual_price:.2f}, "
                       f"Predicted=${predicted_price:.2f}, Accuracy={accuracy:.2f}%")
        
        if results:
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            
            logger.info("\n" + "="*80)
            logger.info(f"MULTI-DAY BENCHMARK SUMMARY ({len(results)} days)")
            logger.info("="*80)
            logger.info(f"\nAverage Accuracy: {avg_accuracy:.2f}%")
            
            if avg_accuracy >= 90:
                logger.info(f"\n✓ SUCCESS: Average accuracy {avg_accuracy:.2f}% meets 90% threshold!")
            else:
                logger.info(f"\n⚠ Average accuracy {avg_accuracy:.2f}% on tested days")
            
            logger.info("="*80)
            
            return results
        
    except Exception as e:
        logger.error(f"Multi-day benchmark failed: {str(e)}")
        return None


def main():
    """Run real benchmark tests."""
    logger.info("\n" + "="*80)
    logger.info("HONEST BENCHMARK - USING LIVE DATA")
    logger.info("="*80)
    logger.info("\nThis test will:")
    logger.info("1. Fetch REAL gold prices from Yahoo Finance (not mock data)")
    logger.info("2. Train model on historical data (excluding most recent day)")
    logger.info("3. Predict the most recent day's actual price")
    logger.info("4. Compare prediction with real actual price")
    logger.info("\nNote: Requires internet access to Yahoo Finance API")
    logger.info("="*80)
    
    # Single day benchmark
    result = real_benchmark_test()
    
    if result is None:
        logger.error("\n" + "="*80)
        logger.error("BENCHMARK CANNOT RUN - NO LIVE DATA ACCESS")
        logger.error("="*80)
        logger.info("\nThe benchmark requires live internet access to fetch real gold prices.")
        logger.info("Without this access, we cannot verify predictions against actual prices.")
        logger.info("\nIn a real environment with internet access:")
        logger.info("  1. The script would fetch live gold prices from Yahoo Finance")
        logger.info("  2. Train on all data except the most recent day")
        logger.info("  3. Predict the most recent day's price")
        logger.info("  4. Compare with the actual price")
        logger.info("\nThe model architecture and training process are sound,")
        logger.info("but validation requires real-time data access.")
        logger.info("="*80)
        return 1
    
    # If single day worked, try multi-day
    if result and result['accuracy'] >= 85:
        logger.info("\n\nAttempting multi-day benchmark for more robust evaluation...")
        multi_results = multi_day_benchmark(num_days=5)
    
    return 0


if __name__ == "__main__":
    exit(main())
