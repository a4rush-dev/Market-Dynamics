"""
Backtesting engine - design choices.

Strategy-agnostic: consumes a target-weight DataFrame (dates x tickers)
from any strategy and a simple-return panel, and centrally handles
execution lag, P&L, transaction costs, turnover, and exposure. No
strategy module does its own portfolio accounting.

Execution lag: weights[t] is the position decided using information
through day t (already enforced inside each strategy module). The
engine shifts weights forward one day (lagged_weights[t] = weights[t-1])
before multiplying by returns[t], so the return earned on day t reflects
a position decided the day before - this is the single place look-ahead
bias is mechanically prevented.

Return type: simple returns, not log returns, for all portfolio-level
P&L math - matches the additivity argument from the returns-calculation
stage (simple returns aggregate correctly across weighted positions).

Alignment across strategies: all three strategies are backtested over
the SAME full date range (the entire simple_returns.parquet index), not
truncated to each strategy's own warmup-free window. Each strategy's
own weight file is already zero during its own warmup period, so this
produces a fair, same-calendar-period comparison without special-casing
- zero-weight days simply contribute zero return, turnover, and cost.

Turnover / transaction costs: computed on lagged_weights (the actual
realized position path), not on raw weights - turnover on day t is the
change in the ACTUAL held position from day t-1 to day t, and cost is
charged the same day, proportional to TRANSACTION_COST_BPS one-way.

Sharpe ratio assumes a zero risk-free rate (standard simplification for
this kind of project, not a claim that the true risk-free rate is zero).

STARTING_CAPITAL is purely for an intuitive equity-curve dollar figure;
all ratio-based metrics (Sharpe, cumulative/annualized return, max
drawdown) are scale-invariant and unaffected by its value.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "clean"
STRATEGIES_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "strategies" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

STRATEGY_FILES = {
    "mean_reversion": "mean_reversion_weights.parquet",
    "momentum": "momentum_weights.parquet",
    "pairs_trading": "pairs_trading_weights.parquet",
}

TRANSACTION_COST_BPS = 2.0
STARTING_CAPITAL = 100_000.0
TRADING_DAYS_PER_YEAR = 252


def load_simple_returns() -> pd.DataFrame:
    return pd.read_parquet(CLEAN_DIR / "simple_returns.parquet")


def load_weights(name: str) -> pd.DataFrame:
    return pd.read_parquet(STRATEGIES_OUTPUT_DIR / STRATEGY_FILES[name])


def compute_backtest(
    weights: pd.DataFrame, returns: pd.DataFrame, cost_bps: float = TRANSACTION_COST_BPS
) -> pd.DataFrame:
    weights = weights.reindex(returns.index).fillna(0.0)
    weights = weights[returns.columns]

    lagged_weights = weights.shift(1).fillna(0.0)

    gross_return = (lagged_weights * returns).sum(axis=1)

    turnover = lagged_weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = lagged_weights.iloc[0].abs().sum()

    cost_rate = cost_bps / 10000.0
    transaction_cost = turnover * cost_rate

    net_return = gross_return - transaction_cost
    equity = STARTING_CAPITAL * (1 + net_return).cumprod()

    gross_exposure = lagged_weights.abs().sum(axis=1)
    net_exposure = lagged_weights.sum(axis=1)

    return pd.DataFrame(
        {
            "gross_return": gross_return,
            "net_return": net_return,
            "transaction_cost": transaction_cost,
            "turnover": turnover,
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "equity": equity,
        }
    )


def compute_metrics(result: pd.DataFrame) -> dict:
    net_return = result["net_return"]
    n_days = len(net_return)
    years = n_days / TRADING_DAYS_PER_YEAR

    total_return = result["equity"].iloc[-1] / STARTING_CAPITAL - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1
    annualized_vol = net_return.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    sharpe = np.nan
    if net_return.std() > 0:
        sharpe = (net_return.mean() * TRADING_DAYS_PER_YEAR) / annualized_vol

    running_max = result["equity"].cummax()
    drawdown = result["equity"] / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "cumulative_return": total_return,
        "annualized_return": annualized_return,
        "annualized_vol": annualized_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_turnover": result["turnover"].mean(),
        "avg_gross_exposure": result["gross_exposure"].mean(),
        "avg_net_exposure": result["net_exposure"].mean(),
    }


def main() -> None:
    returns = load_simple_returns()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name in STRATEGY_FILES:
        weights = load_weights(name)
        result = compute_backtest(weights, returns)
        result.to_parquet(OUTPUT_DIR / f"{name}_backtest.parquet")
        summary[name] = compute_metrics(result)

    summary_df = pd.DataFrame(summary).T
    summary_df.to_csv(OUTPUT_DIR / "strategy_comparison.csv")

    print(f"date range: {returns.index.min().date()} to {returns.index.max().date()}")
    print(summary_df)


if __name__ == "__main__":
    main()