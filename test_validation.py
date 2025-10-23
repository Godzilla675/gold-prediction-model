#!/usr/bin/env python3
"""
Validation test script to ensure everything works correctly.
"""

print('='*80)
print('VALIDATION TEST')
print('='*80)

# Test 1: Data fetching
print('\n1. Testing data fetching...')
from data_fetcher import GoldDataFetcher
fetcher = GoldDataFetcher()
data = fetcher.fetch_data()
print(f'   ✓ Fetched {len(data)} days of data')

# Test 2: Feature engineering
print('\n2. Testing feature engineering...')
from feature_engineering import FeatureEngineer
engineer = FeatureEngineer(data)
features = engineer.create_all_features()
print(f'   ✓ Created {len(features.columns)} features')

# Test 3: Check saved models
print('\n3. Checking saved models...')
from pathlib import Path
models_dir = Path('models')
model_files = list(models_dir.glob('*.h5')) + list(models_dir.glob('*.joblib'))
print(f'   ✓ Found {len(model_files)} model files:')
for f in sorted(model_files):
    print(f'     - {f.name}')

# Test 4: Load and predict
print('\n4. Testing prediction...')
from predict import GoldPricePredictor
predictor = GoldPricePredictor()
predictor.load_models()
print(f'   ✓ Loaded {len(predictor.models)} models: {list(predictor.models.keys())}')

# Quick prediction with each model
for model_name in predictor.models.keys():
    result = predictor.predict_next_price(model_name)
    print(f'   ✓ {model_name} prediction: ${result["predicted_price"]:.2f}')

print('\n' + '='*80)
print('ALL VALIDATION TESTS PASSED ✓')
print('='*80)
print('\nThe gold price prediction model is ready to use!')
print('Run "python demo.py" for a full demonstration.')
print('Run "python predict.py" to make predictions.')
