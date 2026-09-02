from pathlib import Path

import numpy as np
import pandas as pd

from acquisition import TICKERS

CLEAN_DIR = Path("clean")

def load_adj_close() -> pd.DataFrame:
    panel = pd.read_parquet(CLEAN_DIR / "panel.parquet")
    adj_close = pd.concat(
        {ticker: panel[ticker]["Adj Close"] for ticker in TICKERS}, axis=1
    )
    return adj_close # use adjusted close prices to calculate returns, use raw closes to calc realistic transaction costs

def compute_returns(adj_close: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    simple_returns = adj_close.pct_change().iloc[1:]
    log_returns = np.log(adj_close / adj_close.shift(1)).iloc[1:]
    return simple_returns, log_returns

def save_returns(simple_returns: pd.DataFrame, log_returns: pd.DataFrame) -> None:
    simple_returns.to_parquet(CLEAN_DIR / "simple_returns.parquet")
    log_returns.to_parquet(CLEAN_DIR / "log_returns.parquet")

def validate_reconstruction(
    adj_close: pd.DataFrame, simple_returns: pd.DataFrame, log_returns: pd.DataFrame
) -> None:
    simple_reconstructed = (1 + simple_returns).cumprod()
    simple_actual = adj_close.iloc[1:] / adj_close.iloc[0]
    simple_error = (simple_reconstructed - simple_actual).abs().max()

    log_reconstructed = log_returns.cumsum()
    log_actual = np.log(adj_close.iloc[1:] / adj_close.iloc[0])
    log_error = (log_reconstructed - log_actual).abs().max()

    print("max reconstruction error, simple returns:")
    print(simple_error)
    print()
    print("max reconstruction error, log returns:")
    print(log_error)
    print()

def summarize_distribution(
    simple_returns: pd.DataFrame, log_returns: pd.DataFrame
) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "simple_mean": simple_returns.mean(),
            "simple_std": simple_returns.std(),
            "simple_skew": simple_returns.skew(),
            "simple_kurtosis": simple_returns.kurtosis(),
            "log_mean": log_returns.mean(),
            "log_std": log_returns.std(),
            "log_skew": log_returns.skew(),
            "log_kurtosis": log_returns.kurtosis(),
        }
    )
    print(summary)
    return summary

def main() -> None:
    adj_close = load_adj_close()
    simple_returns, log_returns = compute_returns(adj_close)
    save_returns(simple_returns, log_returns)

    validate_reconstruction(adj_close, simple_returns, log_returns)
    summarize_distribution(simple_returns, log_returns)

if __name__ == "__main__":
    main()