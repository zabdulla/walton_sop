"""Core technical indicators.

All indicators are pure functions: Series/DataFrame in, Series out.
No state, no side effects, no look-ahead.

Each indicator returns a Series aligned with the input index, with NaN
for the warmup period where insufficient data exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving Averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range - measures volatility."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def realized_volatility(series: pd.Series, period: int = 20) -> pd.Series:
    """Annualized realized volatility from returns."""
    returns = series.pct_change()
    return returns.rolling(window=period, min_periods=period).std() * np.sqrt(252)


def bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (upper, middle, lower)."""
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Momentum / Trend
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (0-100)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD. Returns (macd_line, signal_line, histogram)."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def rate_of_change(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change (percentage)."""
    return series.pct_change(periods=period) * 100


# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------

def vwap_ratio(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Price relative to rolling VWAP. >1 means price above VWAP."""
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_tp_vol = (typical_price * df["Volume"]).rolling(period, min_periods=period).sum()
    cum_vol = df["Volume"].rolling(period, min_periods=period).sum()
    rolling_vwap = cum_tp_vol / cum_vol
    return df["Close"] / rolling_vwap


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    sign = np.sign(df["Close"].diff())
    return (sign * df["Volume"]).cumsum()


# ---------------------------------------------------------------------------
# Regime detection (simple)
# ---------------------------------------------------------------------------

def trend_strength(series: pd.Series, period: int = 50) -> pd.Series:
    """Measures trend strength as R-squared of linear regression.

    Returns values 0-1. High values = strong trend, low = choppy/mean-reverting.
    This is useful for regime detection: apply trend-following in high-R2 periods,
    mean-reversion in low-R2 periods.
    """
    def _r2(window):
        if len(window) < period:
            return np.nan
        x = np.arange(len(window))
        y = window.values
        if np.std(y) == 0:
            return 0.0
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation ** 2

    return series.rolling(window=period, min_periods=period).apply(_r2, raw=False)
