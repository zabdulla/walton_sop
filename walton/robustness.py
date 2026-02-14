"""Robustness testing - systematically stress-test strategies for overfitting.

Three key tests:

1. Parameter Perturbation: If slightly changing a parameter destroys returns,
   the strategy is curve-fit to that exact parameter value.

2. Monte Carlo Permutation: Shuffle the trade returns and see how the strategy
   compares. If randomized trades produce similar results, there's no edge.

3. Regime Analysis: Break down performance by market regime (trending vs
   mean-reverting, high vol vs low vol). A robust strategy should have a
   clear thesis about WHICH regime it profits in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from walton.engine import BacktestResult, run_backtest
from walton.indicators.core import realized_volatility, trend_strength
from walton.strategy import Strategy, StrategyParams


# ---------------------------------------------------------------------------
# 1. Parameter Perturbation
# ---------------------------------------------------------------------------

@dataclass
class PerturbationResult:
    """Results of parameter perturbation analysis."""
    param_name: str
    base_sharpe: float
    perturbed_sharpes: list[float]  # one per perturbation level
    perturbation_levels: list[float]  # e.g. [-0.2, -0.1, 0.1, 0.2]

    @property
    def sensitivity(self) -> float:
        """Average absolute change in Sharpe per 10% parameter change.

        High sensitivity = fragile / overfit.
        Low sensitivity = robust.
        """
        if not self.perturbed_sharpes:
            return 0.0
        changes = [abs(s - self.base_sharpe) for s in self.perturbed_sharpes]
        avg_change = np.mean(changes)
        avg_perturbation = np.mean([abs(p) for p in self.perturbation_levels])
        if avg_perturbation == 0:
            return 0.0
        return avg_change / (avg_perturbation / 0.1)  # normalize to per-10%

    @property
    def is_robust(self) -> bool:
        """Parameter is robust if all perturbations keep Sharpe > 0."""
        return all(s > 0 for s in self.perturbed_sharpes)


def test_parameter_sensitivity(
    strategy: Strategy,
    df: pd.DataFrame,
    params: StrategyParams,
    perturbation_levels: list[float] | None = None,
    commission_bps: float = 5.0,
    slippage_bps: float = 5.0,
) -> list[PerturbationResult]:
    """Test how sensitive results are to each parameter.

    Args:
        strategy: Strategy to test
        df: OHLCV data
        params: Base parameters
        perturbation_levels: Fractional perturbations, e.g. [-0.2, -0.1, 0.1, 0.2]

    Returns:
        List of PerturbationResult, one per parameter
    """
    if perturbation_levels is None:
        perturbation_levels = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]

    # Baseline
    base_result = run_backtest(
        strategy, df, params,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    base_sharpe = base_result.sharpe_ratio

    results = []
    for param_name in params.values:
        perturbed_sharpes = []
        for pct in perturbation_levels:
            perturbed_params = params.perturbed(param_name, pct)
            result = run_backtest(
                strategy, df, perturbed_params,
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
            )
            perturbed_sharpes.append(result.sharpe_ratio)

        results.append(PerturbationResult(
            param_name=param_name,
            base_sharpe=base_sharpe,
            perturbed_sharpes=perturbed_sharpes,
            perturbation_levels=perturbation_levels,
        ))

    return results


# ---------------------------------------------------------------------------
# 2. Monte Carlo Permutation Test
# ---------------------------------------------------------------------------

@dataclass
class MonteCarloResult:
    """Results of Monte Carlo permutation test."""
    actual_sharpe: float
    actual_total_return: float
    simulated_sharpes: list[float]
    simulated_returns: list[float]
    n_simulations: int

    @property
    def sharpe_percentile(self) -> float:
        """Percentile rank of actual Sharpe among simulations.

        If the actual Sharpe is at the 95th percentile, only 5% of random
        orderings produced a better Sharpe = there's likely a real signal.
        Below 80th percentile = probably noise.
        """
        below = sum(1 for s in self.simulated_sharpes if s < self.actual_sharpe)
        return below / len(self.simulated_sharpes) * 100

    @property
    def p_value(self) -> float:
        """p-value: fraction of simulations that beat actual performance."""
        above = sum(1 for s in self.simulated_sharpes if s >= self.actual_sharpe)
        return above / len(self.simulated_sharpes)


def monte_carlo_test(
    result: BacktestResult,
    n_simulations: int = 1000,
    seed: int = 42,
) -> MonteCarloResult:
    """Test if strategy returns could be explained by random chance.

    Permutes the order of trade returns and recomputes metrics.
    If the actual ordering doesn't significantly outperform random
    orderings, the strategy likely has no real edge.
    """
    rng = np.random.default_rng(seed)

    if not result.trades:
        return MonteCarloResult(
            actual_sharpe=result.sharpe_ratio,
            actual_total_return=result.total_return,
            simulated_sharpes=[0.0] * n_simulations,
            simulated_returns=[0.0] * n_simulations,
            n_simulations=n_simulations,
        )

    trade_returns = [t.return_pct for t in result.trades]
    actual_sharpe = result.sharpe_ratio
    actual_return = result.total_return

    sim_sharpes = []
    sim_returns = []

    for _ in range(n_simulations):
        shuffled = rng.permutation(trade_returns)
        eq_curve = np.cumprod(1 + shuffled)
        daily_ret = np.diff(eq_curve) / eq_curve[:-1] if len(eq_curve) > 1 else np.array([0.0])

        if len(daily_ret) > 0 and np.std(daily_ret) > 0:
            sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)
        else:
            sharpe = 0.0

        total_ret = eq_curve[-1] - 1 if len(eq_curve) > 0 else 0.0

        sim_sharpes.append(sharpe)
        sim_returns.append(total_ret)

    return MonteCarloResult(
        actual_sharpe=actual_sharpe,
        actual_total_return=actual_return,
        simulated_sharpes=sim_sharpes,
        simulated_returns=sim_returns,
        n_simulations=n_simulations,
    )


# ---------------------------------------------------------------------------
# 3. Regime Analysis
# ---------------------------------------------------------------------------

@dataclass
class RegimeBreakdown:
    """Performance breakdown by market regime."""
    regime_name: str
    n_bars: int
    n_trades: int
    sharpe: float
    total_return: float
    win_rate: float


def analyze_regimes(
    result: BacktestResult,
    df: pd.DataFrame,
    vol_period: int = 20,
    trend_period: int = 50,
) -> list[RegimeBreakdown]:
    """Break down strategy performance by market regime.

    Regimes are defined by:
    - Volatility: high vs low (above/below median realized vol)
    - Trend: trending vs choppy (above/below median trend strength)

    This creates 4 regimes: high-vol-trending, high-vol-choppy,
    low-vol-trending, low-vol-choppy.

    A good strategy should have a clear thesis about which regime it profits in.
    """
    vol = realized_volatility(df["Close"], vol_period)
    trend = trend_strength(df["Close"], trend_period)

    # Drop NaN warmup period
    valid = vol.dropna().index.intersection(trend.dropna().index)
    vol = vol.loc[valid]
    trend = trend.loc[valid]

    vol_median = vol.median()
    trend_median = trend.median()

    regimes = {
        "high_vol_trending": (vol > vol_median) & (trend > trend_median),
        "high_vol_choppy": (vol > vol_median) & (trend <= trend_median),
        "low_vol_trending": (vol <= vol_median) & (trend > trend_median),
        "low_vol_choppy": (vol <= vol_median) & (trend <= trend_median),
    }

    breakdowns = []
    for regime_name, mask in regimes.items():
        regime_dates = mask[mask].index
        if len(regime_dates) == 0:
            continue

        regime_returns = result.returns.reindex(regime_dates).dropna()
        if regime_returns.empty:
            continue

        # Count trades in this regime
        regime_trades = [
            t for t in result.trades
            if t.entry_date in regime_dates or t.exit_date in regime_dates
        ]

        sharpe = 0.0
        if len(regime_returns) > 1 and regime_returns.std() > 0:
            sharpe = regime_returns.mean() / regime_returns.std() * np.sqrt(252)

        total_ret = (1 + regime_returns).prod() - 1
        win_rate = 0.0
        if regime_trades:
            win_rate = sum(1 for t in regime_trades if t.return_pct > 0) / len(regime_trades)

        breakdowns.append(RegimeBreakdown(
            regime_name=regime_name,
            n_bars=len(regime_returns),
            n_trades=len(regime_trades),
            sharpe=sharpe,
            total_return=total_ret,
            win_rate=win_rate,
        ))

    return breakdowns
