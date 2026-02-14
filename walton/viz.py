"""Visualization module for strategy analysis.

All plotting functions return matplotlib Figure objects so they work
both in notebooks (inline display) and scripts (save to file).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from walton.engine import BacktestResult
from walton.robustness import MonteCarloResult, PerturbationResult, RegimeBreakdown
from walton.walkforward import WalkForwardResult


# Style defaults
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (14, 6)
COLORS = {"equity": "#2196F3", "benchmark": "#9E9E9E", "drawdown": "#F44336"}


def plot_equity_curve(
    result: BacktestResult,
    benchmark: pd.Series | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot equity curve with drawdown overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)

    # Equity curve
    ax1.plot(result.equity_curve.index, result.equity_curve.values,
             color=COLORS["equity"], linewidth=1.5, label="Strategy")

    if benchmark is not None:
        # Normalize benchmark to same starting value
        bm_normalized = benchmark / benchmark.iloc[0] * result.equity_curve.iloc[0]
        ax1.plot(bm_normalized.index, bm_normalized.values,
                 color=COLORS["benchmark"], linewidth=1, alpha=0.7, label="Benchmark")

    ax1.set_ylabel("Equity")
    ax1.legend(loc="upper left")
    ax1.set_title(title or f"{result.strategy_name} | "
                  f"Sharpe: {result.sharpe_ratio:.2f} | "
                  f"CAGR: {result.cagr:.1%} | "
                  f"MaxDD: {result.max_drawdown:.1%} | "
                  f"Trades: {result.n_trades}")

    # Drawdown
    peak = result.equity_curve.cummax()
    drawdown = (result.equity_curve - peak) / peak
    ax2.fill_between(drawdown.index, drawdown.values, 0,
                     color=COLORS["drawdown"], alpha=0.4)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    fig.tight_layout()
    return fig


def plot_returns_distribution(result: BacktestResult) -> plt.Figure:
    """Plot distribution of daily returns with key statistics."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    returns = result.returns.dropna()

    # Histogram
    ax1.hist(returns, bins=50, color=COLORS["equity"], alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.axvline(returns.mean(), color="red", linestyle="--", label=f"Mean: {returns.mean():.4f}")
    ax1.axvline(0, color="black", linestyle="-", alpha=0.3)
    ax1.set_xlabel("Daily Return")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Daily Returns Distribution")
    ax1.legend()

    # QQ-like: rolling Sharpe
    rolling_sharpe = returns.rolling(63).mean() / returns.rolling(63).std() * np.sqrt(252)
    ax2.plot(rolling_sharpe.index, rolling_sharpe.values, color=COLORS["equity"], linewidth=1)
    ax2.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax2.axhline(result.sharpe_ratio, color="red", linestyle="--", alpha=0.5,
                label=f"Overall: {result.sharpe_ratio:.2f}")
    ax2.set_ylabel("Rolling Sharpe (63d)")
    ax2.set_xlabel("Date")
    ax2.set_title("Rolling Sharpe Ratio")
    ax2.legend()

    fig.tight_layout()
    return fig


def plot_trade_analysis(result: BacktestResult) -> plt.Figure:
    """Analyze individual trade characteristics."""
    if not result.trades:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.text(0.5, 0.5, "No trades to analyze", ha="center", va="center", fontsize=14)
        return fig

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    trade_returns = [t.return_pct for t in result.trades]
    bars_held = [t.bars_held for t in result.trades]

    # Trade returns distribution
    ax = axes[0, 0]
    ax.hist(trade_returns, bins=30, color=COLORS["equity"], alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title(f"Trade Returns (n={len(trade_returns)})")
    ax.set_xlabel("Return per trade")

    # Cumulative trade PnL
    ax = axes[0, 1]
    cum_returns = np.cumsum(trade_returns)
    ax.plot(range(len(cum_returns)), cum_returns, color=COLORS["equity"])
    ax.set_title("Cumulative Trade P&L")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative Return")

    # Holding period distribution
    ax = axes[1, 0]
    ax.hist(bars_held, bins=30, color=COLORS["equity"], alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Holding Period (avg: {np.mean(bars_held):.1f} bars)")
    ax.set_xlabel("Bars held")

    # Return by entry month (seasonality check)
    ax = axes[1, 1]
    monthly_returns = {}
    for t in result.trades:
        month = t.entry_date.month
        if month not in monthly_returns:
            monthly_returns[month] = []
        monthly_returns[month].append(t.return_pct)

    months = sorted(monthly_returns.keys())
    avg_returns = [np.mean(monthly_returns[m]) for m in months]
    ax.bar(months, avg_returns, color=COLORS["equity"], alpha=0.7)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Avg Trade Return by Entry Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Return")
    ax.set_xticks(months)

    fig.tight_layout()
    return fig


def plot_parameter_sensitivity(results: list[PerturbationResult]) -> plt.Figure:
    """Visualize parameter sensitivity analysis."""
    n_params = len(results)
    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5), squeeze=False)

    for i, pr in enumerate(results):
        ax = axes[0, i]
        levels = pr.perturbation_levels
        sharpes = pr.perturbed_sharpes

        # Color bars by whether they're positive
        colors = ["#4CAF50" if s > 0 else "#F44336" for s in sharpes]
        ax.bar([f"{l:+.0%}" for l in levels], sharpes, color=colors, alpha=0.7)
        ax.axhline(pr.base_sharpe, color="blue", linestyle="--",
                   label=f"Base: {pr.base_sharpe:.2f}")
        ax.axhline(0, color="black", linestyle="-", alpha=0.3)

        robust_label = "ROBUST" if pr.is_robust else "FRAGILE"
        ax.set_title(f"{pr.param_name}\nSensitivity: {pr.sensitivity:.2f} ({robust_label})")
        ax.set_xlabel("Perturbation")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend()

    fig.suptitle("Parameter Sensitivity Analysis", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_monte_carlo(mc: MonteCarloResult) -> plt.Figure:
    """Visualize Monte Carlo permutation test results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE)

    # Sharpe distribution
    ax1.hist(mc.simulated_sharpes, bins=50, color=COLORS["benchmark"],
             alpha=0.7, edgecolor="black", linewidth=0.5, label="Random permutations")
    ax1.axvline(mc.actual_sharpe, color="red", linewidth=2,
                label=f"Actual: {mc.actual_sharpe:.2f}")
    ax1.set_title(f"Monte Carlo: Sharpe\np-value: {mc.p_value:.3f} | "
                  f"Percentile: {mc.sharpe_percentile:.0f}th")
    ax1.set_xlabel("Sharpe Ratio")
    ax1.legend()

    # Returns distribution
    ax2.hist(mc.simulated_returns, bins=50, color=COLORS["benchmark"],
             alpha=0.7, edgecolor="black", linewidth=0.5, label="Random permutations")
    ax2.axvline(mc.actual_total_return, color="red", linewidth=2,
                label=f"Actual: {mc.actual_total_return:.1%}")
    ax2.set_title("Monte Carlo: Total Return")
    ax2.set_xlabel("Total Return")
    ax2.legend()

    verdict = "LIKELY REAL EDGE" if mc.p_value < 0.05 else "LIKELY NOISE"
    fig.suptitle(f"Monte Carlo Permutation Test ({mc.n_simulations} sims) - {verdict}",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_regime_breakdown(breakdowns: list[RegimeBreakdown]) -> plt.Figure:
    """Visualize performance by market regime."""
    if not breakdowns:
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.text(0.5, 0.5, "No regime data", ha="center", va="center", fontsize=14)
        return fig

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [b.regime_name.replace("_", "\n") for b in breakdowns]
    sharpes = [b.sharpe for b in breakdowns]
    returns = [b.total_return for b in breakdowns]
    n_trades = [b.n_trades for b in breakdowns]

    # Sharpe by regime
    colors = ["#4CAF50" if s > 0 else "#F44336" for s in sharpes]
    axes[0].bar(names, sharpes, color=colors, alpha=0.7)
    axes[0].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[0].set_title("Sharpe by Regime")
    axes[0].set_ylabel("Sharpe Ratio")

    # Returns by regime
    colors = ["#4CAF50" if r > 0 else "#F44336" for r in returns]
    axes[1].bar(names, [r * 100 for r in returns], color=colors, alpha=0.7)
    axes[1].axhline(0, color="black", linestyle="-", alpha=0.3)
    axes[1].set_title("Total Return by Regime")
    axes[1].set_ylabel("Return (%)")

    # Trade count by regime
    axes[2].bar(names, n_trades, color=COLORS["equity"], alpha=0.7)
    axes[2].set_title("# Trades by Regime")
    axes[2].set_ylabel("Count")

    fig.suptitle("Performance by Market Regime", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_walk_forward(wf: WalkForwardResult) -> plt.Figure:
    """Visualize walk-forward analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Combined OOS equity
    ax = axes[0, 0]
    if not wf.combined_oos_equity.empty:
        ax.plot(wf.combined_oos_equity.index, wf.combined_oos_equity.values,
                color=COLORS["equity"], linewidth=1.5)
        # Mark window boundaries
        for w in wf.windows:
            ax.axvline(w.test_start, color="gray", linestyle="--", alpha=0.3)
    ax.set_title(f"Combined OOS Equity | Sharpe: {wf.oos_sharpe:.2f}")
    ax.set_ylabel("Equity (normalized)")

    # WFE per window
    ax = axes[0, 1]
    window_nums = [w.window_num for w in wf.windows]
    wfes = [w.wfe for w in wf.windows]
    colors = ["#4CAF50" if w > 0.5 else "#FF9800" if w > 0.3 else "#F44336" for w in wfes]
    ax.bar(window_nums, wfes, color=colors, alpha=0.7)
    ax.axhline(0.5, color="green", linestyle="--", alpha=0.5, label="Good (0.5)")
    ax.axhline(0.3, color="red", linestyle="--", alpha=0.5, label="Poor (0.3)")
    ax.axhline(wf.avg_wfe, color="blue", linestyle="-", alpha=0.7,
               label=f"Avg: {wf.avg_wfe:.2f}")
    ax.set_title("Walk-Forward Efficiency per Window")
    ax.set_xlabel("Window")
    ax.set_ylabel("WFE")
    ax.legend()

    # IS vs OOS Sharpe per window
    ax = axes[1, 0]
    is_sharpes = [w.train_result.sharpe_ratio for w in wf.windows]
    oos_sharpes = [w.test_result.sharpe_ratio for w in wf.windows]
    x = np.arange(len(window_nums))
    width = 0.35
    ax.bar(x - width / 2, is_sharpes, width, label="In-Sample", color="#2196F3", alpha=0.7)
    ax.bar(x + width / 2, oos_sharpes, width, label="Out-of-Sample", color="#FF9800", alpha=0.7)
    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_title("IS vs OOS Sharpe per Window")
    ax.set_xlabel("Window")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xticks(x)
    ax.legend()

    # Parameter stability
    ax = axes[1, 1]
    stability = wf.param_stability
    if stability:
        param_names = list(stability.keys())
        cvs = list(stability.values())
        colors = ["#4CAF50" if cv < 0.3 else "#FF9800" if cv < 0.5 else "#F44336" for cv in cvs]
        ax.barh(param_names, cvs, color=colors, alpha=0.7)
        ax.axvline(0.3, color="green", linestyle="--", alpha=0.5, label="Stable (<0.3)")
        ax.axvline(0.5, color="red", linestyle="--", alpha=0.5, label="Unstable (>0.5)")
        ax.set_title("Parameter Stability (CV)")
        ax.set_xlabel("Coefficient of Variation")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No parameter data", ha="center", va="center")

    fig.suptitle(f"Walk-Forward Analysis: {wf.strategy_name}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_comparison(results: list[BacktestResult], metric: str = "sharpe") -> plt.Figure:
    """Compare multiple strategy results side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    names = [f"{r.strategy_name}\n({r.split_name})" for r in results]

    # Equity curves (normalized)
    ax = axes[0]
    for r in results:
        normalized = r.equity_curve / r.equity_curve.iloc[0]
        ax.plot(normalized.index, normalized.values, linewidth=1.2, label=r.strategy_name)
    ax.set_title("Normalized Equity Curves")
    ax.set_ylabel("Growth of $1")
    ax.legend(fontsize=8)

    # Metric comparison
    ax = axes[1]
    if metric == "sharpe":
        values = [r.sharpe_ratio for r in results]
        ax.set_ylabel("Sharpe Ratio")
    elif metric == "calmar":
        values = [r.calmar_ratio for r in results]
        ax.set_ylabel("Calmar Ratio")
    else:
        values = [r.cagr for r in results]
        ax.set_ylabel("CAGR")

    colors = ["#4CAF50" if v > 0 else "#F44336" for v in values]
    ax.bar(range(len(names)), values, color=colors, alpha=0.7, tick_label=names)
    ax.axhline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_title(f"{metric.title()} Comparison")

    # Risk comparison
    ax = axes[2]
    mdd = [abs(r.max_drawdown) * 100 for r in results]
    ax.bar(range(len(names)), mdd, color="#F44336", alpha=0.7, tick_label=names)
    ax.set_title("Max Drawdown (%)")
    ax.set_ylabel("Drawdown (%)")

    fig.tight_layout()
    return fig
