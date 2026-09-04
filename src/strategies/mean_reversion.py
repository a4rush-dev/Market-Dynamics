"""
Design choices.

Type: time-series single-asset, each of the 11 ETFs trade on its own signal independently (no cross-sections).

Return horizon: 5 day trailing cumulative log return per ETF.
Z-score lookback: a rolling 100 day mean/std of that return to find the z-score.

Entry: |z| > 2 opens a position
Exit: |z| < 0.5 closes a position

Position size: Linearly proportional to |z| with a cap at Z_CAP = 4.0, then flat beyond that. Negative z implies long, positive z implies short.

Portfolio weights: each active ETF's target weight = its own signal size / sum of all signal sizes that day, scaled to a fixed gross exposure target (GROSS_TARGET = 1.0).

Discussion of the output: strategy is flat on ~47.74% of trading days, since entry z = 2 is a strict bar on a 100-day baseline. Very selective.
"""

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "clean"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

RETURN_WINDOW = 5
Z_WINDOW = 100
ENTRY_Z = 2.0
EXIT_Z = 0.5
Z_CAP = 4.0
GROSS_TARGET = 1.0

# TODO: calculate turnover for multiple transaction costs

def load_log_returns() -> pd.DataFrame:
    return pd.read_parquet(CLEAN_DIR / "log_returns.parquet")

def compute_zscores(log_returns: pd.DataFrame) -> pd.DataFrame:
    cum_return = log_returns.rolling(RETURN_WINDOW).sum()
    rolling_mean = cum_return.shift(1).rolling(Z_WINDOW).mean()
    rolling_std = cum_return.shift(1).rolling(Z_WINDOW).std() # shift them because today's observations shouldn't be in previous observations
    return (cum_return - rolling_mean) / rolling_std # 100 consecutive daily z-scores of overlapping 5-day returns, might be a problem when backtesting starts


def generate_raw_signal(z: pd.Series) -> pd.Series:
    values = z.to_numpy()
    signal = np.zeros(len(values))
    in_position = False

    for i, zi in enumerate(values):
        if np.isnan(zi):
            in_position = False
            continue

        abs_z = abs(zi)

        if not in_position:
            if abs_z >= ENTRY_Z:
                in_position = True
        else:
            if abs_z < EXIT_Z:
                in_position = False

        if in_position:
            size = np.clip(abs_z / Z_CAP, 0, 1)
            signal[i] = -np.sign(zi) * size

    return pd.Series(signal, index=z.index)


def generate_raw_signals(z_panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {ticker: generate_raw_signal(z_panel[ticker]) for ticker in z_panel.columns}
    )

def signals_to_weights(raw_signals: pd.DataFrame) -> pd.DataFrame:
    abs_sum = raw_signals.abs().sum(axis=1)
    scale = np.where(abs_sum > 0, GROSS_TARGET / abs_sum, 0.0)
    return raw_signals.mul(scale, axis=0)


def main() -> None:
    log_returns = load_log_returns()
    z = compute_zscores(log_returns)
    raw_signals = generate_raw_signals(z)
    weights = signals_to_weights(raw_signals)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(OUTPUT_DIR / "mean_reversion_weights.parquet")

    active_count = (raw_signals != 0).sum(axis=1)
    gross_exposure = weights.abs().sum(axis=1)

    print(f"date range: {weights.index.min().date()} to {weights.index.max().date()}")
    print(f"average active positions per day: {active_count.mean():.2f}")
    print(f"max active positions per day: {int(active_count.max())}")
    print(f"days with zero active positions: {int((active_count == 0).sum())}")
    print(f"average gross exposure: {gross_exposure.mean():.4f}")
    print(f"max gross exposure: {gross_exposure.max():.4f}")


if __name__ == "__main__":
    main()