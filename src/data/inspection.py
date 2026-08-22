# a sanity check for cached data using basic stats and flagging outliers.
# Flagged outliers look good though, based on real market events of 2008, 2020 and the AI boom

import pandas as pd

from acquisition import TICKERS, CACHE_DIR

STD_THRESHOLD = 5

def load_ticker(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / f"{ticker}.parquet")

def inspect_ticker(ticker: str, df: pd.DataFrame, reference_dates: set) -> pd.Timestamp:
    print(f"--- {ticker} ---")
    print(f"rows: {len(df)}")
    print(f"index type: {type(df.index)}")
    print(f"index tz: {df.index.tz}")
    print(f"dtypes:\n{df.dtypes}")
    print(f"date range: {df.index.min().date()} to {df.index.max().date()}")

    duplicate_count = df.index.duplicated().sum()
    print(f"duplicate dates: {duplicate_count}")

    nan_counts = df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    print(f"NaN counts:\n{nan_counts if len(nan_counts) else 'none'}")

    zero_volume = (df["Volume"] == 0).sum()
    print(f"zero volume days: {zero_volume}")

    non_positive_close = (df["Close"] <= 0).sum()
    print(f"non-positive close days: {non_positive_close}")

    relevant_reference = {d for d in reference_dates if d >= df.index.min()}
    missing_dates = relevant_reference - set(df.index)
    print(f"missing dates vs SPY calendar: {len(missing_dates)}")
    if missing_dates:
        print(sorted(missing_dates)[:10])

    adj_diff = (df["Close"] - df["Adj Close"]).abs()
    divergence_days = (adj_diff > 1e-8).sum()
    max_relative_divergence = (adj_diff / df["Close"]).max()
    print(f"days Close != Adj Close: {divergence_days}")
    print(f"max relative Close/Adj Close divergence: {max_relative_divergence:.4f}")

    returns = df["Adj Close"].pct_change().dropna()
    mean_r = returns.mean()
    std_r = returns.std()
    z_scores = (returns - mean_r) / std_r
    flagged = returns[z_scores.abs() > STD_THRESHOLD] # 5 sig
    print(f"return mean: {mean_r:.6f}, std: {std_r:.6f}")
    print(f"flagged extreme return days (> {STD_THRESHOLD} std): {len(flagged)}")
    if len(flagged) > 0:
        print(flagged)

    print()
    return df.index.min()


def main() -> None:
    spy_df = load_ticker("SPY")
    reference_dates = set(spy_df.index)

    start_dates = []
    for ticker in TICKERS:
        df = load_ticker(ticker)
        start_dates.append(inspect_ticker(ticker, df, reference_dates))

    print(f"latest per-ticker start date (intersection candidate): {max(start_dates).date()}")


if __name__ == "__main__":
    main()