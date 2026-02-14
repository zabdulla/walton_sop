"""Data fetching, storage, and split management.

Design principles:
- All data splits are explicit and enforced - no accidental look-ahead bias
- Raw data is immutable once fetched
- Indicators are computed per-split to prevent information leakage
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


@dataclass(frozen=True)
class DataSplit:
    """An immutable view into a specific time range of market data.

    This prevents accidental look-ahead bias by making the boundaries explicit.
    """

    df: pd.DataFrame
    name: str  # e.g. "train", "test", "oos_2020"
    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self):
        if self.df.empty:
            raise ValueError(f"DataSplit '{self.name}' has no data between {self.start} and {self.end}")

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"DataSplit(name='{self.name}', rows={len(self.df)}, "
            f"range={self.start.date()}..{self.end.date()})"
        )


def fetch_ohlcv(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance with optional local caching.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    Index is DatetimeIndex named 'Date'.
    """
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{symbol}_{start}_{end}_{interval}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    if yf is None:
        raise ImportError(
            "yfinance is required for fetching data. Install with: pip install yfinance"
        )

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {symbol} from {start} to {end}")

    # Normalize columns
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index).tz_localize(None)

    if cache_path is not None:
        df.to_parquet(cache_path)

    return df


def make_splits(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
) -> dict[str, DataSplit]:
    """Split data chronologically into train/validation/test sets.

    This is NOT the same as walk-forward analysis (see walkforward.py).
    This is a simple initial sanity check split. Walk-forward is the real test.

    Args:
        df: OHLCV DataFrame with DatetimeIndex
        train_frac: Fraction for in-sample training
        val_frac: Fraction for parameter selection / validation
        test_frac: Fraction for final out-of-sample test

    Returns:
        Dict mapping split name to DataSplit
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("Fractions must sum to 1.0")

    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    splits = {}

    train_df = df.iloc[:train_end]
    splits["train"] = DataSplit(
        df=train_df, name="train",
        start=train_df.index[0], end=train_df.index[-1],
    )

    if val_frac > 0:
        val_df = df.iloc[train_end:val_end]
        splits["val"] = DataSplit(
            df=val_df, name="val",
            start=val_df.index[0], end=val_df.index[-1],
        )

    test_df = df.iloc[val_end:]
    splits["test"] = DataSplit(
        df=test_df, name="test",
        start=test_df.index[0], end=test_df.index[-1],
    )

    return splits


def load_csv(path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Expects columns: Open, High, Low, Close, Volume (case-insensitive).
    """
    df = pd.read_csv(path, parse_dates=[date_col], index_col=date_col)
    df.columns = [c.strip().title() for c in df.columns]
    expected = {"Open", "High", "Low", "Close", "Volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"
    return df


def make_synthetic(
    n_bars: int = 2000,
    drift: float = 0.0002,
    volatility: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    This is useful for framework smoke testing, but NEVER use synthetic
    data to validate a strategy thesis - it has no real market microstructure.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-01", periods=n_bars)
    log_returns = rng.normal(drift, volatility, n_bars)
    close = 100 * np.exp(np.cumsum(log_returns))
    intraday_vol = abs(rng.normal(0, 0.005, n_bars))

    df = pd.DataFrame({
        "Open": close * (1 + rng.normal(0, 0.002, n_bars)),
        "High": close * (1 + intraday_vol),
        "Low": close * (1 - intraday_vol),
        "Close": close,
        "Volume": rng.integers(1_000_000, 10_000_000, n_bars),
    }, index=dates)
    df.index.name = "Date"
    return df


def compute_returns(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """Compute simple returns from a price column."""
    return df[col].pct_change()


def compute_log_returns(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """Compute log returns from a price column."""
    return np.log(df[col] / df[col].shift(1))
