"""
Transaction cost sensitivity - design choices.

Sweeps COST_LEVELS_BPS (one-way, per trade) through the same backtest
engine used for the baseline comparison, reusing compute_backtest and
compute_metrics unchanged - this is a parameter sweep on top of the
existing engine, not a separate cost model.

0 bps is included as a frictionless upper bound (how much of the
strategy's apparent edge is pure signal vs. execution reality). The
default 2 bps anchors the realistic case. 5/10/20/50 bps progressively
stress-test how much cost each strategy's turnover profile can absorb
before its edge disappears.

Breakeven cost: linear interpolation across the swept grid to estimate
the one-way cost level (in bps) at which cumulative_return crosses
zero, for strategies where it does cross within the tested range. This
is an approximation (the true return-vs-cost relationship isn't
perfectly linear given costs interact with turnover nonlinearly across
time), not an exact solve.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from engine import (
    STRATEGY_FILES,
    compute_backtest,
    compute_metrics,
    load_simple_returns,
    load_weights,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
COST_LEVELS_BPS = [0, 2, 5, 10, 20, 50]


def run_sensitivity(returns: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name in STRATEGY_FILES:
        weights = load_weights(name)
        for cost_bps in COST_LEVELS_BPS:
            result = compute_backtest(weights, returns, cost_bps=cost_bps)
            metrics = compute_metrics(result)
            metrics["strategy"] = name
            metrics["cost_bps"] = cost_bps
            records.append(metrics)

    df = pd.DataFrame(records)
    return df.set_index(["strategy", "cost_bps"])


def estimate_breakeven(df: pd.DataFrame) -> dict:
    breakeven = {}
    for name in STRATEGY_FILES:
        sub = df.loc[name]["cumulative_return"]
        costs = sub.index.to_numpy(dtype=float)
        rets = sub.to_numpy()

        if rets[0] <= 0:
            breakeven[name] = None
            continue

        crossing = None
        for i in range(len(rets) - 1):
            if rets[i] > 0 and rets[i + 1] <= 0:
                x0, x1 = costs[i], costs[i + 1]
                y0, y1 = rets[i], rets[i + 1]
                crossing = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                break

        breakeven[name] = crossing
    return breakeven


def main() -> None:
    returns = load_simple_returns()
    df = run_sensitivity(returns)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "cost_sensitivity.csv")

    print(df.to_string())

    breakeven = estimate_breakeven(df)
    print("\nestimated breakeven one-way cost (bps):")
    for name, value in breakeven.items():
        if value is None:
            print(f"  {name}: already <= 0 at 0 bps (no breakeven in range)")
        else:
            print(f"  {name}: {value:.2f}")


if __name__ == "__main__":
    main()