"""
Generate a sample stock-like dataset for testing QuantumPredict.
Run: python generate_sample_data.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate 3 years of daily data
start = datetime(2021, 1, 1)
dates = pd.bdate_range(start=start, periods=756)  # ~3 years of business days

# Simulate price with trend + seasonality + noise
n = len(dates)
t = np.arange(n)

# Base trend (upward)
trend = 150 + t * 0.15

# Seasonal component
seasonal = 10 * np.sin(2 * np.pi * t / 252)

# Random walk component
returns = np.random.normal(0.0003, 0.018, n)
random_walk = np.cumprod(1 + returns) * 150

# Combine
close = trend * 0.3 + random_walk * 0.7 + seasonal + np.random.normal(0, 2, n)
close = np.maximum(close, 10)  # Ensure positive

# Generate OHLV from close
open_p = close * (1 + np.random.normal(0, 0.005, n))
high = np.maximum(close, open_p) * (1 + np.abs(np.random.normal(0, 0.008, n)))
low = np.minimum(close, open_p) * (1 - np.abs(np.random.normal(0, 0.008, n)))
volume = np.random.lognormal(14, 0.5, n).astype(int)

df = pd.DataFrame({
    'Date': dates.strftime('%Y-%m-%d'),
    'Open': open_p.round(2),
    'High': high.round(2),
    'Low': low.round(2),
    'Close': close.round(2),
    'Volume': volume,
})

df.to_csv('sample_AAPL.csv', index=False)
print(f"✓ Generated sample_AAPL.csv with {len(df)} rows")
print(f"  Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
print(f"\nNow upload sample_AAPL.csv in QuantumPredict!")
