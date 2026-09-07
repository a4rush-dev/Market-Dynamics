"""
Design choices.

Type: time-series single-asset, no cross-sections (universe is too heterogenous and may give spurious correlations)

Return horizon: 6 month (126 trading days) trailing cumulative log return per ETF, skipping the most recent 1 month (21 trading days)

Rnk lookback: a rolling 756 trading days (~3 year) history of each ETF's formation return, used to calculate its percentile rank against its own history.

Conviction: percentile rank is scaled [-1,1] via 2 * (pct - 0.5)

Volatility scaling: conviction is scaled by each ETF's own trailing 63-day realized std.

Portfolio weights: each ETF's target weight = its own vol-scaled signal/sum of the absolute vol-scaled signals across all ETFs, scaled to 1.0 (GROSS_EXPOSURE)

Rebalancing: signals are recomputed daily, but target weights are updated weekly on the last available trading day of each calendar week

Discussion of the output: strategy is flat on ~9% of trading days.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "clean"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

LOOKBACK_DAYS = 126
SKIP_DAYS = 21
RANK_WINDOW = 756
VOL_WINDOW = 63
GROSS_TARGET = 1.0

RAW_SIGNAL_CAP = 2000.0

def load_log_returns() -> pd.DataFrame:
    return pd.read_parquet(CLEAN_DIR / "log_returns.parquet")


def compute_formation_return(log_returns: pd.DataFrame) -> pd.DataFrame:
    cum_return = log_returns.rolling(LOOKBACK_DAYS).sum()
    return cum_return.shift(SKIP_DAYS)

def _pct_rank_last(x: np.ndarray) -> float:
    valid = x[~np.isnan(x)]
    if len(valid) < 2 or np.isnan(x[-1]):
        return np.nan
    return (valid < valid[-1]).sum() / (len(valid) - 1)


def rolling_percentile_rank(formation: pd.DataFrame) -> pd.DataFrame:
    return formation.apply(
        lambda col: col.rolling(RANK_WINDOW, min_periods=RANK_WINDOW // 2).apply(
            _pct_rank_last, raw=True
        )
    )


def conviction_from_rank(rank_pct: pd.DataFrame) -> pd.DataFrame:
    return 2 * (rank_pct - 0.5)


def realized_vol(log_returns: pd.DataFrame) -> pd.DataFrame:
    return log_returns.rolling(VOL_WINDOW).std()


def vol_scale_signal(conviction: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    raw = conviction / vol
    raw = raw.replace([np.inf, -np.inf], np.nan)
    return raw.clip(lower=-RAW_SIGNAL_CAP, upper=RAW_SIGNAL_CAP)


def signals_to_weights(raw_signals: pd.DataFrame) -> pd.DataFrame:
    abs_sum = raw_signals.abs().sum(axis=1)
    scale = np.where(abs_sum > 0, GROSS_TARGET / abs_sum, 0.0)
    return raw_signals.fillna(0.0).mul(scale, axis=0)

def get_rebalance_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    iso = index.isocalendar()
    marker = pd.Series(index, index=pd.MultiIndex.from_arrays([iso["year"], iso["week"]]))
    marker = marker.groupby(level=[0, 1]).max()
    return pd.DatetimeIndex(marker)

def apply_weekly_rebalance(
    daily_weights: pd.DataFrame, rebalance_dates: pd.DatetimeIndex
) -> pd.DataFrame:
    rebalance_dates = rebalance_dates.intersection(daily_weights.index)
    weights = pd.DataFrame(
        index=daily_weights.index, columns=daily_weights.columns, dtype=float
    )
    weights.loc[rebalance_dates] = daily_weights.loc[rebalance_dates]
    return weights.ffill().fillna(0.0)


def main() -> None:
    log_returns = load_log_returns()

    formation = compute_formation_return(log_returns)
    rank_pct = rolling_percentile_rank(formation)
    conviction = conviction_from_rank(rank_pct)
    vol = realized_vol(log_returns)
    raw_signals = vol_scale_signal(conviction, vol)

    daily_weights = signals_to_weights(raw_signals)
    rebalance_dates = get_rebalance_dates(daily_weights.index)
    weights = apply_weekly_rebalance(daily_weights, rebalance_dates)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(OUTPUT_DIR / "momentum_weights.parquet")

    active_count = (weights != 0).sum(axis=1)
    gross_exposure = weights.abs().sum(axis=1)

    print(f"date range: {weights.index.min().date()} to {weights.index.max().date()}")
    print(f"number of rebalance dates: {len(rebalance_dates)}")
    print(f"average active positions per day: {active_count.mean():.2f}")
    print(f"max active positions per day: {int(active_count.max())}")
    print(f"days with zero active positions: {int((active_count == 0).sum())}")
    print(f"average gross exposure: {gross_exposure.mean():.4f}")
    print(f"max gross exposure: {gross_exposure.max():.4f}")


if __name__ == "__main__":
    main()