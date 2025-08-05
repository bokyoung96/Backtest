# Crypto Signal Generator

This module provides a clean implementation of the crypto trading signal generation logic, separated from database dependencies.

## Features

- Calculates VWAP (Volume Weighted Average Price)
- Computes price movement metrics
- Generates trading signals based on price bands and VWAP crossovers
- Tracks exposure and price band touches
- Works with intraday price data

## Usage

```python
from crypto_signal_generator import CryptoSignalGenerator
import pandas as pd

# Load your price data (must have OHLCV columns)
# Option 1: Data with TIMEFRAME column
data = pd.read_csv('your_data.csv')
data['TIMEFRAME'] = pd.to_datetime(data['TIMEFRAME'])

# Option 2: Data with DatetimeIndex already set
data = pd.read_csv('your_data.csv', parse_dates=['date'])
data.set_index('date', inplace=True)

# Initialize the signal generator
signal_generator = CryptoSignalGenerator(
    price_data=data,
    rolling_vol=14,       # Rolling window for volatility calculation
    rolling_move=14,      # Rolling window for price movement calculation
    band_multiplier=1,    # Multiplier for price bands
    trade_freq=15         # Trading frequency in minutes
)

# Run the signal generation process
result = signal_generator.run()

# Analyze the results
print(result[['CLOSE', 'vwap', 'UB', 'LB', 'signals', 'exposure']].tail())
```

## Parameters

- `price_data`: DataFrame with OHLCV data (must have OPEN, HIGH, LOW, CLOSE, VOL columns)
- `rolling_vol`: Window size for volatility calculation (days)
- `rolling_move`: Window size for price movement calculation (days)
- `band_multiplier`: Multiplier for upper and lower price bands
- `trade_freq`: How often to generate trading signals (in minutes)

## Output Columns

The result DataFrame includes the following calculated columns:

- `vwap`: Volume Weighted Average Price
- `move_open`: Absolute price movement from the open price
- `d_vol`: Daily volatility
- `min_from_open`: Minutes from market open (adjusted -525 min)
- `min_of_day`: Rounded minutes of the day
- `move_open_rolling_mean`: Rolling average of price movements
- `sigma_open`: Shifted rolling average (used for bands)
- `UB`: Upper price band
- `LB`: Lower price band
- `signals`: Raw signal values (1 for buy, -1 for sell, 0 for no action)
- `exposure`: Actual trade exposure considering trade frequency
- `touch_ub`: Whether price touched the upper band
- `touch_lb`: Whether price touched the lower band
- `touch_vwap`: Whether price touched VWAP after hitting a band 