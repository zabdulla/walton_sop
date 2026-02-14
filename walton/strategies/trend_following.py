"""Trend Following Strategy - Dual Moving Average Crossover.

THESIS: Markets exhibit momentum - assets that have been going up tend to
continue going up (and vice versa). This is one of the most studied effects
in finance, supported by:
- Behavioral: herding, anchoring, slow information diffusion
- Mechanical: portfolio rebalancing flows, stop-loss cascades
- Academic: Jegadeesh & Titman (1993), Asness et al. (2013)

CRITICAL ASSUMPTIONS:
1. Trends persist long enough to overcome transaction costs and whipsaws
2. The entry/exit lag from moving averages doesn't eat too much of the move
3. Markets aren't dominated by choppy, range-bound action

KNOWN WEAKNESSES:
- Gets whipsawed badly in sideways markets (many false signals)
- Moving averages are lagging indicators - late entry, late exit
- Performance heavily dependent on the few big trends per year
- If everyone uses the same MA crossover, the edge arbitrages away
"""

from __future__ import annotations

import pandas as pd

from walton.indicators.core import atr, ema, sma
from walton.strategy import Strategy, StrategyParams


class DualMACrossover(Strategy):
    """Classic dual moving average crossover with ATR-based trend filter.

    Goes long when fast MA > slow MA, flat otherwise.
    Uses ATR expansion as a confirmation that a real trend is forming
    (not just noise crossing).
    """

    @property
    def name(self) -> str:
        return "Dual MA Crossover"

    def default_params(self) -> StrategyParams:
        params = StrategyParams()
        params.set("fast_period", 10, lo=3, hi=50)
        params.set("slow_period", 40, lo=20, hi=200)
        params.set("atr_period", 14, lo=5, hi=30)
        params.set("atr_threshold", 1.0, lo=0.5, hi=2.0)
        return params

    def generate_signals(self, df: pd.DataFrame, params: StrategyParams) -> pd.Series:
        close = df["Close"]

        fast_period = int(params.get("fast_period"))
        slow_period = int(params.get("slow_period"))
        atr_period = int(params.get("atr_period"))
        atr_mult = params.get("atr_threshold")

        fast_ma = ema(close, fast_period)
        slow_ma = sma(close, slow_period)
        current_atr = atr(df, atr_period)
        atr_ma = sma(current_atr, slow_period)  # ATR moving average for comparison

        signals = pd.Series(0.0, index=df.index)

        # Long when fast > slow AND volatility is expanding (trend confirmation)
        # The ATR filter helps avoid whipsaws in low-volatility chop
        long_condition = (fast_ma > slow_ma) & (current_atr > atr_ma * atr_mult)

        # Short when fast < slow AND volatility expanding
        short_condition = (fast_ma < slow_ma) & (current_atr > atr_ma * atr_mult)

        signals[long_condition] = 1.0
        signals[short_condition] = -1.0

        return signals


class DonchianBreakout(Strategy):
    """Donchian channel breakout - a purer trend-following approach.

    Buy on N-period high, sell on M-period low. This is closer to the
    original Turtle Trading approach and has fewer parameters to overfit.

    The structural logic: new highs represent genuine price discovery
    and tend to lead to continued movement. The asymmetric exit (shorter
    lookback) captures the trend while exiting faster on reversals.
    """

    @property
    def name(self) -> str:
        return "Donchian Breakout"

    def default_params(self) -> StrategyParams:
        params = StrategyParams()
        params.set("entry_period", 20, lo=10, hi=60)
        params.set("exit_period", 10, lo=5, hi=30)
        return params

    def generate_signals(self, df: pd.DataFrame, params: StrategyParams) -> pd.Series:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        entry_period = int(params.get("entry_period"))
        exit_period = int(params.get("exit_period"))

        entry_high = high.rolling(entry_period, min_periods=entry_period).max()
        entry_low = low.rolling(entry_period, min_periods=entry_period).min()
        exit_high = high.rolling(exit_period, min_periods=exit_period).max()
        exit_low = low.rolling(exit_period, min_periods=exit_period).min()

        signals = pd.Series(0.0, index=df.index)
        position = 0.0

        for i in range(entry_period, len(df)):
            if position == 0:
                if close.iloc[i] >= entry_high.iloc[i - 1]:
                    position = 1.0
                elif close.iloc[i] <= entry_low.iloc[i - 1]:
                    position = -1.0
            elif position == 1.0:
                if close.iloc[i] <= exit_low.iloc[i - 1]:
                    position = 0.0
            elif position == -1.0:
                if close.iloc[i] >= exit_high.iloc[i - 1]:
                    position = 0.0

            signals.iloc[i] = position

        return signals
