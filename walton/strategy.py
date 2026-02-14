"""Strategy interface - the core abstraction for pluggable trading ideas.

Design principles:
- Strategies produce signals from data. That's it. No execution logic.
- Parameters are declared explicitly so they can be perturbed for robustness testing.
- Signals are simple: +1 (long), 0 (flat), -1 (short), or continuous [-1, 1].
- The engine handles position sizing, transaction costs, and execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class StrategyParams:
    """Container for strategy parameters with metadata for robustness testing.

    Each parameter has a name, value, and valid range. The range is used by
    the robustness tester to perturb parameters and check sensitivity.
    """

    values: dict[str, float] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def set(self, name: str, value: float, lo: float, hi: float) -> None:
        """Register a parameter with its valid range."""
        if not lo <= value <= hi:
            raise ValueError(f"Parameter '{name}': {value} not in [{lo}, {hi}]")
        self.values[name] = value
        self.ranges[name] = (lo, hi)

    def get(self, name: str) -> float:
        return self.values[name]

    def perturbed(self, name: str, pct: float) -> StrategyParams:
        """Return a copy with one parameter shifted by pct (e.g. 0.1 = +10%)."""
        new = StrategyParams(
            values=dict(self.values),
            ranges=dict(self.ranges),
        )
        lo, hi = self.ranges[name]
        shifted = self.values[name] * (1 + pct)
        new.values[name] = max(lo, min(hi, shifted))
        return new

    def copy(self) -> StrategyParams:
        return StrategyParams(
            values=dict(self.values),
            ranges=dict(self.ranges),
        )

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v}" for k, v in self.values.items())
        return f"StrategyParams({items})"


class Strategy(ABC):
    """Abstract base class for all trading strategies.

    To implement a new strategy:
    1. Subclass Strategy
    2. Implement `name` property
    3. Implement `default_params()` to declare parameters with ranges
    4. Implement `generate_signals()` to produce position signals from data

    The separation of parameters from signal generation is deliberate -
    it enables systematic robustness testing of parameter sensitivity.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name."""

    @abstractmethod
    def default_params(self) -> StrategyParams:
        """Return default parameters with valid ranges for each.

        Example:
            params = StrategyParams()
            params.set("fast_period", 10, lo=3, hi=50)
            params.set("slow_period", 30, lo=10, hi=200)
            return params
        """

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, params: StrategyParams) -> pd.Series:
        """Generate position signals from OHLCV data.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            params: Strategy parameters

        Returns:
            Series of signals aligned with df.index.
            Values should be in [-1, 1] where:
              +1 = fully long
               0 = flat / no position
              -1 = fully short
            Fractional values represent partial positions.

        IMPORTANT: This method must NOT look ahead. Each signal at index i
        should only use data from indices <= i.
        """

    def describe(self) -> dict[str, Any]:
        """Return a description of this strategy for logging."""
        params = self.default_params()
        return {
            "name": self.name,
            "params": params.values,
            "param_ranges": params.ranges,
        }
