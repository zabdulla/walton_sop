"""Backtesting engine with built-in anti-overfitting safeguards.

Design principles:
- Vectorized where possible for speed
- Transaction costs are ALWAYS applied (no "frictionless" mode)
- Position sizing is explicit
- Results include enough metadata to detect overfitting
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from walton.strategy import Strategy, StrategyParams


@dataclass(frozen=True)
class TradeRecord:
    """Record of a single round-trip trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int  # +1 long, -1 short
    entry_price: float
    exit_price: float
    return_pct: float
    bars_held: int


@dataclass
class BacktestResult:
    """Complete backtest results with anti-overfitting metadata."""

    # Core equity series
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    signals: pd.Series

    # Trade-level data
    trades: list[TradeRecord] = field(default_factory=list)

    # Metadata
    strategy_name: str = ""
    params: StrategyParams | None = None
    split_name: str = ""
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None

    # Cost assumptions
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def total_return(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1

    @property
    def cagr(self) -> float:
        if self.equity_curve.empty:
            return 0.0
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        if years <= 0:
            return 0.0
        return (1 + self.total_return) ** (1 / years) - 1

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio (assuming 0 risk-free rate)."""
        if self.returns.empty or self.returns.std() == 0:
            return 0.0
        return self.returns.mean() / self.returns.std() * np.sqrt(252)

    @property
    def sortino_ratio(self) -> float:
        """Sortino ratio - penalizes only downside volatility."""
        if self.returns.empty:
            return 0.0
        downside = self.returns[self.returns < 0]
        if downside.empty or downside.std() == 0:
            return float("inf") if self.returns.mean() > 0 else 0.0
        return self.returns.mean() / downside.std() * np.sqrt(252)

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative fraction (e.g. -0.15 = 15% drawdown)."""
        if self.equity_curve.empty:
            return 0.0
        peak = self.equity_curve.cummax()
        dd = (self.equity_curve - peak) / peak
        return dd.min()

    @property
    def calmar_ratio(self) -> float:
        """CAGR / abs(max drawdown)."""
        mdd = abs(self.max_drawdown)
        if mdd == 0:
            return 0.0
        return self.cagr / mdd

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.return_pct > 0)
        return wins / len(self.trades)

    @property
    def profit_factor(self) -> float:
        """Gross profits / gross losses."""
        gross_profit = sum(t.return_pct for t in self.trades if t.return_pct > 0)
        gross_loss = abs(sum(t.return_pct for t in self.trades if t.return_pct < 0))
        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @property
    def avg_trade_return(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.return_pct for t in self.trades])

    @property
    def avg_bars_held(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.bars_held for t in self.trades])

    def summary(self) -> dict:
        """Return a summary dict of all key metrics."""
        return {
            "strategy": self.strategy_name,
            "split": self.split_name,
            "params": str(self.params),
            "total_return": f"{self.total_return:.2%}",
            "cagr": f"{self.cagr:.2%}",
            "sharpe": f"{self.sharpe_ratio:.2f}",
            "sortino": f"{self.sortino_ratio:.2f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "calmar": f"{self.calmar_ratio:.2f}",
            "n_trades": self.n_trades,
            "win_rate": f"{self.win_rate:.1%}",
            "profit_factor": f"{self.profit_factor:.2f}",
            "avg_trade_return": f"{self.avg_trade_return:.3%}",
            "avg_bars_held": f"{self.avg_bars_held:.1f}",
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
        }

    def passes_minimum_bar(self, min_trades: int = 100) -> tuple[bool, list[str]]:
        """Check if results meet minimum statistical requirements.

        This is a critical anti-overfitting check. Returns (pass, reasons).
        """
        issues = []

        if self.n_trades < min_trades:
            issues.append(
                f"Insufficient trades: {self.n_trades} < {min_trades}. "
                "Results are not statistically meaningful."
            )

        if self.n_trades > 0:
            # Check if Sharpe is suspiciously high (likely overfit)
            if self.sharpe_ratio > 3.0:
                issues.append(
                    f"Sharpe ratio {self.sharpe_ratio:.2f} is suspiciously high. "
                    "This almost certainly indicates overfitting or a bug."
                )

            # Check for unrealistic win rates
            if self.win_rate > 0.75:
                issues.append(
                    f"Win rate {self.win_rate:.1%} is suspiciously high. "
                    "Verify there's no look-ahead bias."
                )

        return len(issues) == 0, issues


def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    params: StrategyParams | None = None,
    initial_capital: float = 100_000,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
    split_name: str = "",
) -> BacktestResult:
    """Run a backtest of a strategy on OHLCV data.

    Args:
        strategy: Strategy instance to test
        df: OHLCV DataFrame with DatetimeIndex
        params: Strategy parameters (uses defaults if None)
        initial_capital: Starting capital
        commission_bps: Round-trip commission in basis points
        slippage_bps: Estimated slippage in basis points per side
        split_name: Label for this data split (for tracking)

    Returns:
        BacktestResult with full equity curve, trades, and metrics
    """
    if params is None:
        params = strategy.default_params()

    # Generate signals
    signals = strategy.generate_signals(df, params)
    signals = signals.reindex(df.index).fillna(0).clip(-1, 1)

    # Shift signals by 1 to avoid look-ahead: signal on day T -> position on day T+1
    # This is CRITICAL. The signal computed from day T's close is acted on at day T+1's close.
    positions = signals.shift(1).fillna(0)

    # Compute returns
    price_returns = df["Close"].pct_change().fillna(0)
    strategy_returns = positions * price_returns

    # Apply transaction costs on position changes
    turnover = positions.diff().abs().fillna(0)
    total_cost_bps = commission_bps + slippage_bps * 2  # slippage on both entry and exit
    cost_per_unit = total_cost_bps / 10_000
    costs = turnover * cost_per_unit
    strategy_returns = strategy_returns - costs

    # Build equity curve
    equity = initial_capital * (1 + strategy_returns).cumprod()

    # Extract trades
    trades = _extract_trades(positions, df["Close"], commission_bps, slippage_bps)

    return BacktestResult(
        equity_curve=equity,
        returns=strategy_returns,
        positions=positions,
        signals=signals,
        trades=trades,
        strategy_name=strategy.name,
        params=params,
        split_name=split_name,
        start_date=df.index[0],
        end_date=df.index[-1],
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )


def _extract_trades(
    positions: pd.Series,
    prices: pd.Series,
    commission_bps: float,
    slippage_bps: float,
) -> list[TradeRecord]:
    """Extract individual round-trip trades from a position series."""
    trades = []
    in_trade = False
    entry_date = None
    entry_price = None
    direction = 0

    for i in range(1, len(positions)):
        prev_pos = positions.iloc[i - 1]
        curr_pos = positions.iloc[i]

        # Entry: 0 -> non-zero
        if prev_pos == 0 and curr_pos != 0:
            in_trade = True
            entry_date = positions.index[i]
            entry_price = prices.iloc[i]
            direction = int(np.sign(curr_pos))

        # Exit: non-zero -> 0, or direction change
        elif in_trade and (curr_pos == 0 or np.sign(curr_pos) != direction):
            exit_price = prices.iloc[i]
            raw_return = direction * (exit_price / entry_price - 1)
            cost = (commission_bps + slippage_bps * 2) / 10_000
            net_return = raw_return - cost
            bars = i - positions.index.get_loc(entry_date)

            trades.append(TradeRecord(
                entry_date=entry_date,
                exit_date=positions.index[i],
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=net_return,
                bars_held=bars,
            ))

            # If direction changed, start new trade immediately
            if curr_pos != 0:
                entry_date = positions.index[i]
                entry_price = prices.iloc[i]
                direction = int(np.sign(curr_pos))
            else:
                in_trade = False
                entry_date = None
                entry_price = None
                direction = 0

    return trades
