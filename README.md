# Gold Price Prediction Model

A comprehensive machine learning system for predicting gold prices using multiple advanced models including LSTM, GRU, XGBoost, and ensemble methods. The system fetches real-time gold price data, performs extensive feature engineering, and trains multiple models to achieve high accuracy predictions.

## 🎯 Features

- **Multiple ML Models**: LSTM, GRU, XGBoost, Prophet, and Ensemble methods
- **Real-time Data**: Fetches up-to-date gold price data using Yahoo Finance (free API)
- **GUI Application**: User-friendly graphical interface for easy predictions
- **Executable Builds**: Pre-built executables for Windows, Linux, and macOS
- **Advanced Feature Engineering**: 
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV)
  - Price-based features (changes, ranges, gaps)
  - Lagged features for time series analysis
  - Rolling statistics (mean, std, min, max)
  - Time-based features (cyclical encoding)
- **Comprehensive Evaluation**: MAE, RMSE, MAPE, R², and Directional Accuracy metrics
- **Automatic Model Selection**: Identifies best performing model
- **Accuracy Validation**: Ensures >90% accuracy threshold
- **Backtesting**: Tests models on historical data
- **Visualization**: Plots predictions vs actual prices

## 📋 Requirements

- Python 3.8 or higher
- See `requirements.txt` for all dependencies

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/Godzilla675/gold-prediction-model.git
cd gold-prediction-model
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📖 Usage

### GUI Application (Recommended)

The easiest way to use the gold price prediction model is through the graphical user interface:

```bash
python gui_app.py
```

The GUI provides:
- **Current Price Display**: View the latest gold price
- **Model Loading**: Easily load trained prediction models
- **Price Predictions**: Get predictions from all models with one click
- **Visual Results**: See predictions in an organized table
- **Activity Log**: Monitor operations in real-time

**Using the Executable:**

If you download the pre-built executable from the releases or GitHub Actions artifacts:
- Windows: Double-click `GoldPredictionModel.exe`
- Linux/macOS: Run `./GoldPredictionModel` in terminal

### Training Models

Train all models on historical gold price data:

```bash
python trainer.py
```

This will:
1. Fetch 9+ years of historical gold price data
2. Perform feature engineering
3. Train LSTM, GRU, and XGBoost models
4. Evaluate each model on test data
5. Generate comprehensive reports
6. Save trained models and plots

### Making Predictions

Predict future gold prices using trained models:

```bash
python predict.py
```

This will:
1. Load trained models
2. Fetch the latest gold price data
3. Make predictions using all available models
4. Display predicted prices with confidence metrics

### Individual Modules

Test individual components:

```bash
# Test data fetching
python data_fetcher.py

# Test feature engineering
python feature_engineering.py

# Test models
python models.py
```

## 📊 Model Architecture

### LSTM Model
- Bidirectional LSTM layers (128, 64, 32 units)
- Dropout layers (0.2) for regularization
- Dense output layer
- Huber loss function
- Adam optimizer with learning rate scheduling

### GRU Model
- Bidirectional GRU layers (128, 64, 32 units)
- Dropout layers (0.2) for regularization
- Dense output layer
- Huber loss function
- Adam optimizer with learning rate scheduling

### XGBoost Model
- 1000 estimators with early stopping
- Max depth: 7
- Learning rate: 0.01
- Subsample: 0.8
- Optimized for time series regression

### Ensemble Model
- Weighted average of all models
- Dynamic weight adjustment based on performance
- Improves prediction stability

## 📈 Performance Metrics

The system evaluates models using:

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of average squared differences
- **MAPE (Mean Absolute Percentage Error)**: Average percentage error
- **R² Score**: Coefficient of determination (goodness of fit)
- **Directional Accuracy**: Percentage of correct directional predictions (up/down)

### Accuracy Threshold

The system requires models to achieve >90% accuracy (100 - MAPE) on test data. If this threshold is not met, the system suggests improvements.

## 🔧 Configuration

Edit `config.py` to customize:

- Data sources and date ranges
- Model hyperparameters
- Training configuration
- Feature engineering parameters
- Evaluation metrics

## 🏗️ Building Executables

### Automated Builds (GitHub Actions)

The project includes a GitHub Actions workflow that automatically builds executables for Windows, Linux, and macOS on every push or pull request.

**Accessing Built Executables:**
1. Go to the "Actions" tab in the GitHub repository
2. Click on the latest "Build Executable with UI" workflow run
3. Download the artifacts for your platform:
   - `GoldPredictionModel-Windows` for Windows
   - `GoldPredictionModel-Linux` for Linux
   - `GoldPredictionModel-macOS` for macOS

### Manual Build

To build the executable manually on your local machine:

```bash
# Install PyInstaller if not already installed
pip install pyinstaller

# Build the executable
pyinstaller gold_prediction_gui.spec

# The executable will be created in the dist/ directory
```

**Platform-specific notes:**
- **Windows**: Creates `dist/GoldPredictionModel.exe`
- **Linux**: Creates `dist/GoldPredictionModel` (requires `python3-tk` package)
- **macOS**: Creates `dist/GoldPredictionModel`

The built executable is standalone and includes all necessary dependencies.

## 📂 Project Structure

```
gold-prediction-model/
├── config.py                 # Configuration settings
├── data_fetcher.py          # Data fetching module
├── feature_engineering.py   # Feature creation
├── models.py                # ML model implementations
├── trainer.py               # Training pipeline
├── predict.py               # Prediction script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Data storage (created automatically)
├── models/                 # Trained models (created automatically)
├── plots/                  # Visualization outputs (created automatically)
└── logs/                   # Log files (created automatically)
```

## 🔬 Research & Methodology

### Data Source
- **Yahoo Finance (yfinance)**: Free, reliable API with extensive historical data
- **Gold Futures (GC=F)** and **SPDR Gold Shares ETF (GLD)** as data sources
- 9+ years of historical data for robust training

### Feature Engineering Research
Based on extensive research in time series forecasting and financial prediction:
- **Technical Analysis**: Industry-standard indicators used by traders
- **Lagged Features**: Capture temporal dependencies
- **Rolling Statistics**: Identify trends and volatility patterns
- **Cyclical Encoding**: Handle periodic patterns in financial data

### Model Selection
Models chosen based on proven performance in time series prediction:
- **LSTM/GRU**: Excellent for capturing long-term dependencies in time series
- **XGBoost**: State-of-the-art gradient boosting, robust to noise
- **Ensemble**: Combines strengths of multiple models for better stability

### Validation Strategy
- **Time Series Split**: Maintains temporal order of data
- **Walk-forward Validation**: Simulates real-world prediction scenario
- **Multiple Metrics**: Comprehensive evaluation from different perspectives

## 🎓 Best Practices

1. **Retrain Regularly**: Financial markets change; retrain models monthly
2. **Monitor Performance**: Track prediction accuracy over time
3. **Ensemble Predictions**: Use multiple models for more reliable predictions
4. **Consider Context**: External factors (geopolitical events, inflation) affect gold prices
5. **Risk Management**: Use predictions as one input in investment decisions

## ⚠️ Disclaimer

This model is for educational and research purposes only. Financial predictions are inherently uncertain. Do not use this as the sole basis for investment decisions. Always consult with financial professionals and conduct your own research.

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Yahoo Finance for providing free financial data
- TensorFlow and scikit-learn communities
- Technical analysis library (ta) developers
- Open source community for amazing tools

---

**Built with ❤️ for accurate gold price predictions**