import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS = [
    "SPY",  # US large cap equity
    "MDY",  # US mid cap equity
    "DIA",  # US large cap equity, price-weighted
    "QQQ",  # US large cap growth / tech
    "IYR",  # US real estate
    "IWM",  # US small-cap equity
    "EFA",  # developed intl equity
    "SHY",  # short-term US treasuries
    "TLT",  # long-term US treasuries
    "LQD",  # investment grade corporate bonds
    "EEM",  # emerging markets equity
]

START_DATE = "2003-04-01"
END_DATE = "2026-07-31"

CACHE_DIR = Path("raw")

EXPECTED_COLUMNS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

def fetch_ticker(ticker: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        actions=False,
    )

    if df.empty:
        raise RuntimeError(f"{ticker}: empty dataframe returned")

    if not EXPECTED_COLUMNS.issubset(set(df.columns)):
        raise RuntimeError(
            f"{ticker}: missing expected columns, got {list(df.columns)}"
        )

    return df


def save_ticker(ticker: str, df: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{ticker}.parquet"
    df.to_parquet(out_path)


def main() -> None:
    for ticker in TICKERS:
        try:
            df = fetch_ticker(ticker)
            save_ticker(ticker, df)
            print(f"Saved {ticker}")
        except Exception as exc:
            print(f"acquisition failed on {ticker}: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"{ticker}: {len(df)} rows, {df.index.min().date()} to {df.index.max().date()}")


if __name__ == "__main__":
    main()