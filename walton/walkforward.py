"""Walk-Forward Analysis - the gold standard for strategy validation.

Walk-forward analysis simulates what would actually happen if you were
optimizing and trading in real time:

1. Optimize parameters on a training window
2. Trade the next out-of-sample window with those parameters
3. Roll the window forward and repeat

If a strategy's out-of-sample performance is consistently worse than
in-sample, it's overfit. If it holds up, there may be a real signal.

Key metric: Walk-Forward Efficiency (WFE) = OOS performance / IS performance
- WFE > 0.5 is decent
- WFE > 0.7 is good
- WFE < 0.3 means the strategy is likely overfit
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd

from walton.engine import BacktestResult, run_backtest
from walton.strategy import Strategy, StrategyParams


@dataclass
class WalkForwardWindow:
    """A single window in walk-forward analysis."""
    window_num: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: StrategyParams
    train_result: BacktestResult
    test_result: BacktestResult

    @property
    def wfe(self) -> float:
        """Walk-Forward Efficiency for this window."""
        is_sharpe = self.train_result.sharpe_ratio
        oos_sharpe = self.test_result.sharpe_ratio
        if is_sharpe == 0:
            return 0.0
        return oos_sharpe / is_sharpe


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis results."""
    windows: list[WalkForwardWindow]
    strategy_name: str
    combined_oos_equity: pd.Series  # stitched OOS equity curves

    @property
    def avg_wfe(self) -> float:
        """Average Walk-Forward Efficiency across all windows."""
        wfes = [w.wfe for w in self.windows]
        return np.mean(wfes)

    @property
    def oos_sharpe(self) -> float:
        """Sharpe ratio of the combined OOS equity curve."""
        returns = self.combined_oos_equity.pct_change().dropna()
        if returns.empty or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)

    @property
    def oos_max_drawdown(self) -> float:
        peak = self.combined_oos_equity.cummax()
        dd = (self.combined_oos_equity - peak) / peak
        return dd.min()

    @property
    def total_oos_trades(self) -> int:
        return sum(w.test_result.n_trades for w in self.windows)

    @property
    def param_stability(self) -> dict[str, float]:
        """Coefficient of variation for each parameter across windows.

        Low CV = parameter is stable across time = good sign.
        High CV = parameter is noisy / overfit to each window = bad sign.
        """
        if not self.windows:
            return {}

        param_names = list(self.windows[0].best_params.values.keys())
        stability = {}
        for name in param_names:
            vals = [w.best_params.get(name) for w in self.windows]
            mean = np.mean(vals)
            if mean == 0:
                stability[name] = float("inf")
            else:
                stability[name] = np.std(vals) / abs(mean)
        return stability

    def summary(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "n_windows": len(self.windows),
            "avg_wfe": f"{self.avg_wfe:.2f}",
            "oos_sharpe": f"{self.oos_sharpe:.2f}",
            "oos_max_dd": f"{self.oos_max_drawdown:.2%}",
            "total_oos_trades": self.total_oos_trades,
            "param_stability": {
                k: f"{v:.2f}" for k, v in self.param_stability.items()
            },
        }


def run_walk_forward(
    strategy: Strategy,
    df: pd.DataFrame,
    param_grid: dict[str, list[float]],
    n_windows: int = 5,
    train_ratio: float = 0.7,
    optimize_metric: str = "sharpe",
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> WalkForwardResult:
    """Run anchored walk-forward analysis.

    Args:
        strategy: Strategy to test
        df: Full OHLCV dataset
        param_grid: Dict of param_name -> list of values to try
            Example: {"fast_period": [5, 10, 15], "slow_period": [20, 30, 40]}
        n_windows: Number of walk-forward windows
        train_ratio: Fraction of each window used for training
        optimize_metric: Metric to optimize ("sharpe", "calmar", "sortino")
        commission_bps: Commission in basis points
        slippage_bps: Slippage in basis points

    Returns:
        WalkForwardResult with all windows and combined OOS results
    """
    n_total = len(df)
    window_size = n_total // n_windows
    if window_size < 60:
        raise ValueError(
            f"Window size {window_size} is too small. Need at least 60 bars per window. "
            f"Either reduce n_windows or use more data."
        )

    windows = []
    oos_equities = []

    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = min(start_idx + window_size, n_total)

        window_df = df.iloc[start_idx:end_idx]
        train_end_idx = int(len(window_df) * train_ratio)

        train_df = window_df.iloc[:train_end_idx]
        test_df = window_df.iloc[train_end_idx:]

        if len(train_df) < 30 or len(test_df) < 10:
            continue

        # Optimize on training set
        best_params, best_result = _optimize_params(
            strategy, train_df, param_grid, optimize_metric,
            commission_bps, slippage_bps,
        )

        # Test on OOS set with best params
        test_result = run_backtest(
            strategy, test_df, best_params,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            split_name=f"oos_window_{i}",
        )

        windows.append(WalkForwardWindow(
            window_num=i,
            train_start=train_df.index[0],
            train_end=train_df.index[-1],
            test_start=test_df.index[0],
            test_end=test_df.index[-1],
            best_params=best_params,
            train_result=best_result,
            test_result=test_result,
        ))

        # Normalize OOS equity to chain properly
        oos_eq = test_result.equity_curve / test_result.equity_curve.iloc[0]
        oos_equities.append(oos_eq)

    # Stitch OOS equity curves together
    if oos_equities:
        combined = _stitch_equity_curves(oos_equities)
    else:
        combined = pd.Series(dtype=float)

    return WalkForwardResult(
        windows=windows,
        strategy_name=strategy.name,
        combined_oos_equity=combined,
    )


def _optimize_params(
    strategy: Strategy,
    train_df: pd.DataFrame,
    param_grid: dict[str, list[float]],
    metric: str,
    commission_bps: float,
    slippage_bps: float,
) -> tuple[StrategyParams, BacktestResult]:
    """Grid search over parameters on training data."""
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    best_score = -np.inf
    best_params = None
    best_result = None
    base_params = strategy.default_params()

    for combo in product(*param_values):
        params = base_params.copy()
        for name, val in zip(param_names, combo):
            lo, hi = params.ranges.get(name, (float("-inf"), float("inf")))
            params.values[name] = max(lo, min(hi, val))

        result = run_backtest(
            strategy, train_df, params,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            split_name="train_opt",
        )

        score = _get_metric(result, metric)

        if score > best_score:
            best_score = score
            best_params = params
            best_result = result

    if best_params is None:
        best_params = base_params
        best_result = run_backtest(
            strategy, train_df, base_params,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            split_name="train_opt",
        )

    return best_params, best_result


def _get_metric(result: BacktestResult, metric: str) -> float:
    """Extract a named metric from a backtest result."""
    if metric == "sharpe":
        return result.sharpe_ratio
    elif metric == "sortino":
        return result.sortino_ratio
    elif metric == "calmar":
        return result.calmar_ratio
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _stitch_equity_curves(curves: list[pd.Series]) -> pd.Series:
    """Chain multiple normalized equity curves into one continuous curve."""
    combined = []
    running_value = 1.0

    for curve in curves:
        normalized = curve / curve.iloc[0] * running_value
        combined.append(normalized)
        running_value = normalized.iloc[-1]

    return pd.concat(combined)
