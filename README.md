# QuantumPredict — AI Market Prediction Platform

A full-stack Django application with Deep Learning (LSTM, GRU, Transformer) 
and Machine Learning (Random Forest, XGBoost) models for market price forecasting.

## 🚀 Quick Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Initialize the database
```bash
python manage.py makemigrations market_app
python manage.py migrate
```

### 3. Create admin user (optional)
```bash
python manage.py createsuperuser
```

### 4. Run the development server
```bash
python manage.py runserver
```

### 5. Open browser
```
http://127.0.0.1:8000
```

---

## 📊 How to Use

### Step 1 — Upload Data
- Go to **Data** tab
- Upload a CSV file with historical market data
- Required columns: `Date` + at least one price column (`Close`, `Open`, etc.)
- Optional but powerful: `Open`, `High`, `Low`, `Volume`

**Example CSV format:**
```
Date,Open,High,Low,Close,Volume
2020-01-02,300.35,300.58,295.19,298.39,33870100
2020-01-03,297.15,300.58,296.50,299.80,36580700
...
```

You can download historical data from:
- [Yahoo Finance](https://finance.yahoo.com) → Search ticker → Historical Data → Download
- [Kaggle Datasets](https://www.kaggle.com/datasets?search=stock+price)

### Step 2 — Train a Model
- Go to **Train** tab
- Select your uploaded dataset
- Choose an algorithm:
  - **LSTM** — Best for long sequences, captures temporal dependencies
  - **GRU** — Faster training than LSTM, good for medium datasets
  - **Transformer** — Attention-based, best for complex patterns
  - **Random Forest** — Fast, no GPU needed, good baseline
  - **XGBoost** — High accuracy ML, works without TensorFlow
- Adjust hyperparameters and click **Start Training**
- Training runs in background; watch the progress bar

### Step 3 — Generate Forecast
- Go to **Predict** tab
- Select a trained (Ready) model
- Choose forecast horizon (7–90 days)
- Click **Generate Forecast**
- View:
  - Buy/Sell/Hold signal
  - Bullish/Bearish/Neutral trend
  - Confidence score
  - Interactive price chart with confidence intervals
  - Detailed forecast table

---

## 🧠 Technical Architecture

### Backend (Django + Python)
- **Django 4.x** — Web framework, ORM, routing
- **SQLite** — Database for datasets, models, predictions
- **Background threading** — Non-blocking model training
- **REST API** — JSON endpoints for all operations

### Deep Learning Models (TensorFlow/Keras)
- **Stacked LSTM** — 3-layer LSTM with dropout regularization
- **Stacked GRU** — 3-layer GRU, 40% faster than LSTM
- **Transformer** — Multi-head self-attention with positional encoding
- All DL models use: Early stopping, LR scheduling, Huber loss

### Machine Learning Models (scikit-learn)
- **Random Forest** — 200 estimators, feature importance analysis
- **XGBoost** — 300 estimators with L1/L2 regularization
- **Gradient Boosting** — Fallback if XGBoost unavailable

### Feature Engineering
Automatically computed from OHLCV data:
- Moving Averages (MA 7, 21, 50)
- Exponential Moving Averages (EMA 12, 26)
- MACD + Signal Line
- RSI (14-period)
- Bollinger Bands + Width
- Price Momentum (5d, 10d)
- Volume Ratio
- Volatility

### Uncertainty Quantification
- Monte Carlo Dropout (20 samples)
- 95% confidence intervals for all forecasts
- Calibrated confidence score per prediction

---

## 📁 Project Structure
```
market_predictor/
├── manage.py
├── requirements.txt
├── db.sqlite3 (auto-created)
├── models_saved/ (trained model files)
├── media/datasets/ (uploaded CSV files)
├── market_predictor/
│   ├── settings.py
│   └── urls.py
├── market_app/
│   ├── models.py      # Database models
│   ├── views.py       # API endpoints + page views
│   ├── urls.py        # URL routing
│   └── templates/
│       └── market_app/
│           └── index.html  # Full frontend
└── predictor/
    └── engine.py      # ML/DL training & inference
```

---

## ⚙️ Configuration

In `market_predictor/settings.py`:
- Change `SECRET_KEY` for production
- Set `DEBUG = False` for production  
- Configure `ALLOWED_HOSTS` for your domain
- Switch to PostgreSQL for larger datasets

## 🔧 Without TensorFlow (CPU-only / lightweight)
If TensorFlow is unavailable, the app automatically falls back to:
- Random Forest (scikit-learn)
- Gradient Boosting (scikit-learn)
- XGBoost (if installed)

All ML models work without GPU and train much faster.

---

## 📈 Performance Tips
- **More data = better predictions** (2+ years recommended)
- Use daily OHLCV data for best feature extraction
- LSTM/GRU: 50+ epochs usually needed for convergence
- Lookback window: 30–90 days works well for most assets
- For crypto: use shorter lookbacks (30d) due to high volatility
- For stocks: 60d lookback with LSTM typically performs well
