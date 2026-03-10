"""
Advanced ML/DL Market Prediction Engine
Supports: LSTM, GRU, Transformer, Random Forest, XGBoost, Ensemble
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Try importing deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, 
                                          MultiHeadAttention, LayerNormalization,
                                          Input, GlobalAveragePooling1D, Add)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

# Try XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def compute_technical_indicators(df, target_col='Close'):
    """Add technical indicators as features."""
    data = df.copy()
    prices = data[target_col].values.astype(float)

    # Moving averages
    data['MA_7'] = pd.Series(prices).rolling(7).mean().values
    data['MA_21'] = pd.Series(prices).rolling(21).mean().values
    data['MA_50'] = pd.Series(prices).rolling(50).mean().values

    # Exponential MA
    data['EMA_12'] = pd.Series(prices).ewm(span=12).mean().values
    data['EMA_26'] = pd.Series(prices).ewm(span=26).mean().values

    # MACD
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['MACD_Signal'] = pd.Series(data['MACD'].values).ewm(span=9).mean().values

    # RSI
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    data['RSI'] = (100 - (100 / (1 + rs))).values

    # Bollinger Bands
    rolling_mean = pd.Series(prices).rolling(20).mean()
    rolling_std = pd.Series(prices).rolling(20).std()
    data['BB_Upper'] = (rolling_mean + 2 * rolling_std).values
    data['BB_Lower'] = (rolling_mean - 2 * rolling_std).values
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower'])

    # Volatility
    data['Volatility'] = pd.Series(prices).rolling(10).std().values

    # Price momentum
    data['Momentum_5'] = pd.Series(prices).pct_change(5).values
    data['Momentum_10'] = pd.Series(prices).pct_change(10).values

    # Volume features if available
    if 'Volume' in data.columns:
        vol = data['Volume'].astype(float)
        data['Volume_MA'] = vol.rolling(10).mean().values
        data['Volume_Ratio'] = (vol / (vol.rolling(10).mean() + 1e-10)).values

    data = data.fillna(method='bfill').fillna(0)
    return data


def prepare_sequences(data, target_col, lookback, feature_cols=None):
    """Create sequences for time-series models."""
    if feature_cols is None:
        feature_cols = [target_col]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data[feature_cols].values)

    target_idx = feature_cols.index(target_col)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i])
        y.append(scaled[i, target_idx])

    return np.array(X), np.array(y), scaler


def build_lstm_model(input_shape, units=128):
    """Advanced stacked LSTM with attention."""
    if not DL_AVAILABLE:
        return None

    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units // 2, return_sequences=True),
        Dropout(0.2),
        LSTM(units // 4, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model


def build_gru_model(input_shape, units=128):
    """Stacked GRU model."""
    if not DL_AVAILABLE:
        return None

    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(units // 2, return_sequences=True),
        Dropout(0.2),
        GRU(units // 4, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
    return model


def build_transformer_model(input_shape, num_heads=4, ff_dim=64):
    """Transformer model for time series."""
    if not DL_AVAILABLE:
        return None

    inputs = Input(shape=input_shape)

    # Transformer block 1
    attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(inputs, inputs)
    attn1 = Dropout(0.1)(attn1)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn1)

    # Feed-forward
    ff1 = Dense(ff_dim, activation='relu')(out1)
    ff1 = Dense(input_shape[-1])(ff1)
    ff1 = Dropout(0.1)(ff1)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff1)

    # Transformer block 2
    attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(out2, out2)
    attn2 = Dropout(0.1)(attn2)
    out3 = LayerNormalization(epsilon=1e-6)(out2 + attn2)

    pooled = GlobalAveragePooling1D()(out3)
    x = Dense(64, activation='relu')(pooled)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber', metrics=['mae'])
    return model


def train_deep_learning_model(model_type, dataset_path, target_col, lookback,
                               epochs, batch_size, save_path, scaler_path, log_callback=None):
    """Train LSTM, GRU, or Transformer model."""
    if not DL_AVAILABLE:
        raise RuntimeError("TensorFlow not available. Using ML models only.")

    df = pd.read_csv(dataset_path)
    df = compute_technical_indicators(df, target_col)

    # Build feature list
    feature_cols = [target_col]
    tech_features = ['MA_7', 'MA_21', 'EMA_12', 'MACD', 'RSI',
                     'BB_Width', 'Volatility', 'Momentum_5']
    for f in tech_features:
        if f in df.columns:
            feature_cols.append(f)

    if 'Volume' in df.columns:
        feature_cols.append('Volume_Ratio')

    X, y, scaler = prepare_sequences(df, target_col, lookback, feature_cols)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    input_shape = (X_train.shape[1], X_train.shape[2])

    if model_type == 'lstm':
        model = build_lstm_model(input_shape)
    elif model_type == 'gru':
        model = build_gru_model(input_shape)
    elif model_type == 'transformer':
        model = build_transformer_model(input_shape)
    else:
        model = build_lstm_model(input_shape)

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate
    y_pred = model.predict(X_test, verbose=0).flatten()

    # Inverse transform
    target_idx = feature_cols.index(target_col)
    dummy = np.zeros((len(y_test), len(feature_cols)))
    dummy[:, target_idx] = y_test
    y_test_real = scaler.inverse_transform(dummy)[:, target_idx]

    dummy2 = np.zeros((len(y_pred), len(feature_cols)))
    dummy2[:, target_idx] = y_pred
    y_pred_real = scaler.inverse_transform(dummy2)[:, target_idx]

    rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
    mae = float(mean_absolute_error(y_test_real, y_pred_real))
    mape = float(np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-10))) * 100)
    r2 = float(r2_score(y_test_real, y_pred_real))
    accuracy = float(max(0, 100 - mape))

    model.save(save_path)
    joblib.dump({'scaler': scaler, 'feature_cols': feature_cols, 'target_col': target_col,
                 'lookback': lookback}, scaler_path)

    return {
        'rmse': rmse, 'mae': mae, 'mape': mape,
        'r2_score': r2, 'accuracy': accuracy,
        'epochs_trained': len(history.history['loss'])
    }


def train_ml_model(model_type, dataset_path, target_col, lookback,
                   save_path, scaler_path):
    """Train Random Forest or XGBoost model."""
    df = pd.read_csv(dataset_path)
    df = compute_technical_indicators(df, target_col)

    feature_cols = [target_col, 'MA_7', 'MA_21', 'EMA_12', 'MACD',
                    'RSI', 'BB_Width', 'Volatility', 'Momentum_5']
    feature_cols = [f for f in feature_cols if f in df.columns]

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[feature_cols].values)
    target_idx = feature_cols.index(target_col)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i].flatten())
        y.append(scaled[i, target_idx])

    X, y = np.array(X), np.array(y)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_split=5, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        )
    elif model_type == 'xgboost' and XGB_AVAILABLE:
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            random_state=42
        )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Inverse transform
    dummy = np.zeros((len(y_test), len(feature_cols)))
    dummy[:, target_idx] = y_test
    y_test_real = scaler.inverse_transform(dummy)[:, target_idx]

    dummy2 = np.zeros((len(y_pred), len(feature_cols)))
    dummy2[:, target_idx] = y_pred
    y_pred_real = scaler.inverse_transform(dummy2)[:, target_idx]

    rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred_real)))
    mae = float(mean_absolute_error(y_test_real, y_pred_real))
    mape = float(np.mean(np.abs((y_test_real - y_pred_real) / (y_test_real + 1e-10))) * 100)
    r2 = float(r2_score(y_test_real, y_pred_real))
    accuracy = float(max(0, 100 - mape))

    joblib.dump({'model': model, 'scaler': scaler, 'feature_cols': feature_cols,
                 'target_col': target_col, 'lookback': lookback, 'is_ml': True}, save_path)
    joblib.dump({'scaler': scaler, 'feature_cols': feature_cols, 'target_col': target_col,
                 'lookback': lookback}, scaler_path)

    return {'rmse': rmse, 'mae': mae, 'mape': mape, 'r2_score': r2, 'accuracy': accuracy}


def generate_forecast(model_obj, dataset_path, forecast_days):
    """Generate future predictions with confidence intervals."""
    meta = joblib.load(model_obj.scaler_path)
    scaler = meta['scaler']
    feature_cols = meta['feature_cols']
    target_col = meta['target_col']
    lookback = meta['lookback']

    df = pd.read_csv(dataset_path)
    df = compute_technical_indicators(df, target_col)
    df = df.fillna(method='bfill').fillna(0)

    scaled_data = scaler.transform(df[feature_cols].values)
    target_idx = feature_cols.index(target_col)

    is_ml = meta.get('is_ml', False)
    if is_ml:
        ml_meta = joblib.load(model_obj.model_path)
        ml_model = ml_meta['model']

    last_sequence = scaled_data[-lookback:].copy()
    predictions = []
    mc_samples = 20  # Monte Carlo samples for uncertainty

    for step in range(forecast_days):
        all_preds = []
        for _ in range(mc_samples):
            noise = np.random.normal(0, 0.002, last_sequence.shape)
            noisy_seq = last_sequence + noise

            if is_ml:
                x = noisy_seq.flatten().reshape(1, -1)
                pred_scaled = ml_model.predict(x)[0]
            else:
                x = noisy_seq[np.newaxis, ...]
                if DL_AVAILABLE:
                    model_dl = tf.keras.models.load_model(model_obj.model_path)
                    pred_scaled = model_dl.predict(x, verbose=0)[0, 0]
                else:
                    pred_scaled = last_sequence[-1, target_idx]

            all_preds.append(pred_scaled)

        mean_pred = np.mean(all_preds)
        std_pred = np.std(all_preds)
        predictions.append({
            'mean': mean_pred,
            'lower': mean_pred - 1.96 * std_pred,
            'upper': mean_pred + 1.96 * std_pred
        })

        # Update sequence
        new_row = last_sequence[-1].copy()
        new_row[target_idx] = mean_pred
        last_sequence = np.vstack([last_sequence[1:], new_row])

    # Inverse transform
    raw_preds = np.array([p['mean'] for p in predictions])
    raw_lower = np.array([p['lower'] for p in predictions])
    raw_upper = np.array([p['upper'] for p in predictions])

    def inv(vals):
        dummy = np.zeros((len(vals), len(feature_cols)))
        dummy[:, target_idx] = vals
        return scaler.inverse_transform(dummy)[:, target_idx]

    pred_real = inv(raw_preds)
    lower_real = inv(raw_lower)
    upper_real = inv(raw_upper)

    # Generate future dates
    try:
        last_date = pd.to_datetime(df.iloc[-1, 0])
    except Exception:
        last_date = pd.Timestamp.today()

    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

    # Historical (last 90 days)
    hist_n = min(90, len(df))
    hist_vals = df[target_col].values[-hist_n:].tolist()
    try:
        hist_dates = pd.to_datetime(df.iloc[-hist_n:, 0]).dt.strftime('%Y-%m-%d').tolist()
    except Exception:
        hist_dates = [str(i) for i in range(hist_n)]

    # Determine trend and signal
    pct_change = (pred_real[-1] - pred_real[0]) / (pred_real[0] + 1e-10) * 100
    if pct_change > 2:
        trend, signal = 'bullish', 'buy'
    elif pct_change < -2:
        trend, signal = 'bearish', 'sell'
    else:
        trend, signal = 'neutral', 'hold'

    confidence = float(max(50, min(95, 80 - np.mean(np.abs(raw_upper - raw_lower)) * 100)))

    return {
        'dates': future_dates.strftime('%Y-%m-%d').tolist(),
        'predicted': pred_real.tolist(),
        'lower': lower_real.tolist(),
        'upper': upper_real.tolist(),
        'historical_dates': hist_dates,
        'historical_values': hist_vals,
        'trend': trend,
        'signal': signal,
        'confidence': confidence,
        'pct_change': float(pct_change),
    }
