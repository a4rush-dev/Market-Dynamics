from pathlib import Path

import pandas as pd
from acquisition import TICKERS, CACHE_DIR

CLEAN_DIR = Path("clean")

def load_raw(ticker: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE_DIR / f"{ticker}.parquet")

def build_panel() -> pd.DataFrame:
    frames = [load_raw(ticker) for ticker in TICKERS]
    panel = pd.concat(frames, axis=1, keys=TICKERS, join="inner") # intersection of all ETF dates
    panel = panel.dropna(how="all") # remove NaN rows common to all ETFs.
    return panel

def save_panel(panel: pd.DataFrame) -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CLEAN_DIR / "panel.parquet")

def main() -> None:
    panel = build_panel()
    save_panel(panel)

    print(f"panel shape: {panel.shape}")
    print(f"date range: {panel.index.min().date()} to {panel.index.max().date()}")
    print(f"remaining NaNs: {int(panel.isna().sum().sum())}")


if __name__ == "__main__":
    main()