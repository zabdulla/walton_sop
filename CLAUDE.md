# CLAUDE.md - Coding Standards for walton_sop

This file defines the coding standards for this project. Follow these rules
when writing or modifying code. When in doubt, read existing code in the
module you're editing and match its patterns.

## Core Principle

Write the minimum code needed. If a function can be 10 lines instead of 20,
use 10. Do not add features, helpers, or abstractions beyond what was asked.
Three similar lines are better than a premature abstraction.

## Python Style

- **Python version**: 3.10+. Use `X | Y` union syntax, not `Union[X, Y]`.
- **Line length**: 100 characters max (configured in `pyproject.toml` via ruff).
- **Formatting**: Follow ruff defaults. Run `ruff check --fix` before committing.
- Every module starts with `from __future__ import annotations`.

### Import Order

Strict ordering, one blank line between groups:

```python
from __future__ import annotations

import stdlib_module
from stdlib_module import thing

import numpy as np
import pandas as pd

from walton.engine import run_backtest
from walton.strategy import Strategy
```

1. `__future__` imports
2. Standard library
3. Third-party (numpy, pandas, matplotlib, scipy)
4. Project-relative (`from walton...`)

No wildcard imports. No unused imports.

### Naming

- Functions and variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE` (define at module level, not buried in functions)
- Private helpers: prefix with `_` (e.g., `_extract_trades`)

### Type Annotations

All public function signatures must have full type annotations on both
parameters and return types. Inner/private functions should also be annotated
unless trivially obvious.

```python
# Good
def compute_sharpe(returns: pd.Series, annualize: bool = True) -> float:

# Bad - missing return type
def compute_sharpe(returns, annualize=True):
```

## Docstrings

Use **Google style** with `Args:` / `Returns:` / `Raises:` sections.
One-liner docstrings are fine for simple functions.

```python
def run_backtest(
    strategy: Strategy,
    df: pd.DataFrame,
    params: StrategyParams | None = None,
) -> BacktestResult:
    """Run a backtest of a strategy on OHLCV data.

    Args:
        strategy: Strategy instance to test.
        df: OHLCV DataFrame with DatetimeIndex.
        params: Strategy parameters. Uses defaults if None.

    Returns:
        BacktestResult with equity curve, trades, and metrics.

    Raises:
        ValueError: If df is empty.
    """
```

**Do not** write docstrings that just restate the function name:

```python
# Bad
def get_sharpe_ratio(self) -> float:
    """Get the Sharpe ratio."""

# Good - just skip the docstring, the name is clear enough
def get_sharpe_ratio(self) -> float:
```

## Comments

Comments explain **why**, not **what**. If the code needs a comment to explain
what it does, the code should be rewritten to be clearer.

```python
# Bad - restates the code
# Compute returns
returns = df["Close"].pct_change()

# Good - explains a non-obvious decision
# Shift by 1 to prevent look-ahead: signal on day T executes on day T+1
positions = signals.shift(1).fillna(0)
```

Delete commented-out code. Do not leave `# TODO` markers without a
corresponding issue or concrete plan.

## Pandas / NumPy Patterns

### Vectorize. Do not loop.

This is the single most important rule for this codebase. Never use a
Python `for` loop to iterate over DataFrame rows when a vectorized
operation exists.

```python
# BAD - Python loop over pandas data
position = 0.0
for i in range(len(signals)):
    if position == 0 and long_entry.iloc[i]:
        position = 1.0
    signals.iloc[i] = position

# ACCEPTABLE - when stateful logic truly requires a loop, operate on
# numpy arrays, not pandas objects
values = signals.values
entry = long_entry.values
pos = 0.0
for i in range(len(values)):
    if pos == 0 and entry[i]:
        pos = 1.0
    values[i] = pos
```

Vectorized pandas/numpy is the default. If you must loop (e.g., for stateful
position tracking), extract `.values` first and loop over numpy arrays. Never
call `.iloc[i]` in a hot loop.

### Chaining

Pandas method chains are fine when readable. Break long chains across lines:

```python
signals = (
    strategy.generate_signals(df, params)
    .reindex(df.index)
    .fillna(0)
    .clip(-1, 1)
)
```

### NaN Handling

Be explicit about NaN. Indicator warmup periods produce NaN - document this
and handle it at the boundary, not silently.

## Magic Numbers

Extract thresholds and constants to module-level names with a brief comment.

```python
# Bad
if self.sharpe_ratio > 3.0:
    issues.append("suspicious")

# Good
MAX_PLAUSIBLE_SHARPE = 3.0  # beyond this, almost certainly overfit or bugged

if self.sharpe_ratio > MAX_PLAUSIBLE_SHARPE:
    issues.append("suspicious")
```

## Strategy Implementation Rules

Strategies are the most frequently added code. Follow this template exactly:

```python
from __future__ import annotations

import pandas as pd

from walton.indicators.core import sma, ema
from walton.strategy import Strategy, StrategyParams


class MyStrategy(Strategy):
    """One-line description of what this strategy does.

    THESIS: Why should this edge exist? What structural/behavioral reason?
    REGIMES: When does it work? When does it fail?
    """

    @property
    def name(self) -> str:
        return "My Strategy"

    def default_params(self) -> StrategyParams:
        params = StrategyParams()
        params.set("param_name", default_value, lo=min_val, hi=max_val)
        return params

    def generate_signals(self, df: pd.DataFrame, params: StrategyParams) -> pd.Series:
        # Signal logic here
        # Return Series of values in [-1, 1]
        ...
```

Every strategy **must** have a THESIS and REGIMES section in its docstring.
If you can't articulate why the edge should exist and when it should fail,
the strategy is not ready to be coded.

## Error Handling

- Validate at boundaries (user input, external data). Do not validate
  internal state that can't be wrong if the code is correct.
- Raise specific exceptions (`ValueError`, `TypeError`) with actionable
  messages. Never bare `raise Exception(...)`.
- Do not catch exceptions just to re-raise them.
- Do not add try/except for conditions that should never happen in correct code.

## Testing

- Smoke tests use `make_synthetic()` data. Never validate a trading thesis
  on synthetic data.
- When adding a new strategy, add a smoke test that verifies it runs without
  error on synthetic data and produces a non-empty signal series.
- Backtest results on synthetic data should NOT show strong performance. If
  they do, there's a bug (likely look-ahead bias).

## What NOT to Do

- Do not add logging frameworks. Use `print()` for debugging, remove before commit.
- Do not add configuration file parsers (YAML, TOML, JSON config). Parameters
  are in code via `StrategyParams`.
- Do not add async/await. This is a single-threaded research tool.
- Do not add CLI frameworks (click, argparse). Research happens in notebooks.
- Do not add database integrations. Data is parquet files and CSV.
- Do not create helper/utility modules for one-off operations.
- Do not add backwards-compatibility shims. Just change the code.
