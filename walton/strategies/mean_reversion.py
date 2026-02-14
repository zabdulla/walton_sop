"""Mean Reversion Strategy - Bollinger Band bounce.

THESIS: Prices that deviate significantly from their moving average tend to
revert. This is a well-documented effect in equities at short time horizons
(3-10 days), driven by:
- Overreaction to news
- Liquidity provision (market makers buying dips)
- Mean-reverting earnings surprises

CRITICAL ASSUMPTIONS (each is a potential failure mode):
1. The asset IS mean-reverting in the testing period
2. Transaction costs don't eat the small edge
3. Regime doesn't shift to trending (where this strategy hemorrhages money)

KNOWN WEAKNESSES:
- Loses badly in strong trends / momentum regimes
- Sensitive to the lookback period and threshold
- Can catch falling knives in crashes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from walton.indicators.core import bollinger_bands, rsi
from walton.strategy import Strategy, StrategyParams


class BollingerMeanReversion(Strategy):
    """Buy when price touches lower Bollinger Band, sell at middle band.

    Uses RSI as a confirmation filter to avoid buying into strong downtrends.
    """

    @property
    def name(self) -> str:
        return "Bollinger Mean Reversion"

    def default_params(self) -> StrategyParams:
        params = StrategyParams()
        params.set("bb_period", 20, lo=10, hi=50)
        params.set("bb_std", 2.0, lo=1.0, hi=3.0)
        params.set("rsi_period", 14, lo=5, hi=30)
        params.set("rsi_oversold", 30, lo=15, hi=45)
        params.set("rsi_overbought", 70, lo=55, hi=85)
        return params

    def generate_signals(self, df: pd.DataFrame, params: StrategyParams) -> pd.Series:
        close = df["Close"]

        bb_period = int(params.get("bb_period"))
        bb_std = params.get("bb_std")
        rsi_period = int(params.get("rsi_period"))
        rsi_oversold = params.get("rsi_oversold")
        rsi_overbought = params.get("rsi_overbought")

        upper, middle, lower = bollinger_bands(close, bb_period, bb_std)
        rsi_values = rsi(close, rsi_period)

        signals = pd.Series(0.0, index=df.index)

        # Long when price below lower band AND RSI oversold (confirmation)
        long_entry = (close <= lower) & (rsi_values <= rsi_oversold)

        # Exit long when price reaches middle band or RSI overbought
        long_exit = (close >= middle) | (rsi_values >= rsi_overbought)

        # Build signal series (stateful - need to track position)
        position = 0.0
        for i in range(len(signals)):
            if position == 0 and long_entry.iloc[i]:
                position = 1.0
            elif position == 1.0 and long_exit.iloc[i]:
                position = 0.0
            signals.iloc[i] = position

        return signals
