"""
Design choices.

Universe: all 11_C_2 = 55 pairs tested every formation period, no pair hardcoded (universe is too heterogenous to assume any pair is "obviously" related).

Formation/trading cycle: 252-day formation (estimate hedge ratio, spread mean/std), 126-day trading, rolled forward. Every 21 days during trading, original hedge ratio re-tested for cointegration on fresh data; fails -> flatten pair for rest of cycle.

Selection pipeline: correlation screen on daily returns (not price levels, avoids spurious trend correlation) at |corr| >= 0.5 -> Engle-Granger on log prices, both directions (test not symmetric, keep lower p-value direction) -> keep p-value <= 0.05 -> rank by p-value, take top 5. 55 tests at 5% threshold implies ~2-3 false positives by chance; acknowledged, not corrected for.

Variance guard: reject a regression direction if independent variable's formation variance < 10% of dependent variable's (prevents ill-conditioned beta, e.g. SHY as independent regressor).

Entry/exit/stop: |z| >= 1.75 entry, |z| < 0.75 exit, |z| >= 3.0 stop (flatten, no re-entry same cycle).

Position sizing: per-pair signal proportional to |z| capped at stop. Dollar-neutral legs via 1/(1+|beta|) and beta/(1+|beta|). Active pairs share GROSS_EXPOSURE=1.0 proportional to their own |signal|.

Discussion of the output: 35 of 55 pairs ever selected; top pairs (EFA~MDY, EEM~EFA, IYR~SPY, DIA~IYR, DIA~SPY) persist across many cycles and are economically sensible; single-appearance pairs likely reflect the multiple-testing noise above. Strategy flat ~74% of days, net exposure ~0 on average (dollar neutrality holding up in aggregate).
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

CLEAN_DIR = Path(__file__).resolve().parent.parent / "data" / "clean"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

FORMATION_DAYS = 252
TRADING_DAYS = 126
REVALIDATION_EVERY = 21
CORR_THRESHOLD = 0.5
ADF_PVALUE_THRESHOLD = 0.05
TOP_N_PAIRS = 5
ENTRY_Z = 1.75
EXIT_Z = 0.75
STOP_Z = 3.0
GROSS_TARGET = 1.0
MIN_VARIANCE_RATIO = 0.1

def load_log_prices() -> pd.DataFrame:
    panel = pd.read_parquet(CLEAN_DIR / "panel.parquet")
    tickers = panel.columns.get_level_values(0).unique()
    return pd.concat({t: np.log(panel[t]["Adj Close"]) for t in tickers}, axis=1)


def load_log_returns() -> pd.DataFrame:
    return pd.read_parquet(CLEAN_DIR / "log_returns.parquet")

def test_direction(y: pd.Series, x: pd.Series):
    if x.var() < MIN_VARIANCE_RATIO * y.var():
        return None
    x_const = sm.add_constant(x.to_numpy())
    model = sm.OLS(y.to_numpy(), x_const).fit()
    intercept, beta = model.params
    resid = pd.Series(model.resid, index=y.index)
    pvalue = adfuller(resid, result_object=False)[1]
    return beta, intercept, resid, pvalue

def select_pairs(
    log_prices: pd.DataFrame,
    log_returns: pd.DataFrame,
    formation_start: int,
    formation_end: int) -> list:

    prices = log_prices.iloc[formation_start:formation_end]
    returns = log_returns.iloc[formation_start:formation_end]
    tickers = log_prices.columns

    candidates = []
    for a, b in combinations(tickers, 2):
        corr = returns[a].corr(returns[b])
        if abs(corr) < CORR_THRESHOLD:
            continue

        result_ab = test_direction(prices[a], prices[b])
        result_ba = test_direction(prices[b], prices[a])

        if result_ab is None and result_ba is None:
            continue
        elif result_ba is None:
            dependent, independent = a, b
            beta, intercept, resid, pvalue = result_ab
        elif result_ab is None:
            dependent, independent = b, a
            beta, intercept, resid, pvalue = result_ba
        else:
            beta_ab, intercept_ab, resid_ab, p_ab = result_ab
            beta_ba, intercept_ba, resid_ba, p_ba = result_ba
            if p_ab <= p_ba:
                dependent, independent = a, b
                beta, intercept, resid, pvalue = beta_ab, intercept_ab, resid_ab, p_ab
            else:
                dependent, independent = b, a
                beta, intercept, resid, pvalue = beta_ba, intercept_ba, resid_ba, p_ba

        if pvalue > ADF_PVALUE_THRESHOLD:
            continue

        candidates.append(
            {
                "dependent": dependent,
                "independent": independent,
                "beta": beta,
                "intercept": intercept,
                "spread_mean": resid.mean(),
                "spread_std": resid.std(),
                "pvalue": pvalue,
                "correlation": corr,
            }
        )

    candidates.sort(key=lambda c: c["pvalue"])
    return candidates[:TOP_N_PAIRS]


def compute_spread(log_prices: pd.DataFrame, pair: dict) -> pd.Series:
    dep = log_prices[pair["dependent"]]
    indep = log_prices[pair["independent"]]
    return dep - pair["beta"] * indep - pair["intercept"]


def revalidate(log_prices: pd.DataFrame, pair: dict, end_idx: int) -> bool:
    start_idx = end_idx - FORMATION_DAYS
    if start_idx < 0:
        return True
    window = log_prices.iloc[start_idx:end_idx]
    spread = compute_spread(window, pair).dropna()
    if len(spread) < 20:
        return True
    return adfuller(spread, result_object=False)[1] <= ADF_PVALUE_THRESHOLD

def generate_pair_signal(z: pd.Series) -> pd.Series:
    values = z.to_numpy()
    signal = np.zeros(len(values))
    in_position = False
    stopped = False

    for i, zi in enumerate(values):
        if stopped:
            continue
        if np.isnan(zi):
            in_position = False
            continue

        abs_z = abs(zi)

        if not in_position:
            if abs_z >= ENTRY_Z:
                in_position = True
        else:
            if abs_z >= STOP_Z:
                in_position = False
                stopped = True
                continue
            if abs_z < EXIT_Z:
                in_position = False

        if in_position:
            signal[i] = -np.sign(zi) * min(abs_z / STOP_Z, 1.0)

    return pd.Series(signal, index=z.index)


def run_pairs_backtest(
    log_prices: pd.DataFrame, log_returns: pd.DataFrame
) -> pd.DataFrame:
    n = len(log_prices)
    weights = pd.DataFrame(0.0, index=log_prices.index, columns=log_prices.columns)

    cycle_count = 0
    pair_count_total = 0
    pair_frequency = {}

    start_idx = FORMATION_DAYS
    while start_idx + 1 < n:
        formation_start = start_idx - FORMATION_DAYS
        formation_end = start_idx
        trading_end = min(start_idx + TRADING_DAYS, n)

        pairs = select_pairs(log_prices, log_returns, formation_start, formation_end)
        cycle_count += 1
        pair_count_total += len(pairs)

        cycle_date = log_prices.index[start_idx].date()
        print(f"cycle {cycle_count} ({cycle_date}): {len(pairs)} pairs selected")
        for pair in pairs:
            key = tuple(sorted([pair["dependent"], pair["independent"]]))
            pair_frequency[key] = pair_frequency.get(key, 0) + 1
            print(
                f"  {pair['dependent']} ~ {pair['independent']}  "
                f"corr={pair['correlation']:.3f}  beta={pair['beta']:.3f}  "
                f"pvalue={pair['pvalue']:.4f}"
            )

        pair_signals = []
        for pair in pairs:
            trading_slice = log_prices.iloc[start_idx:trading_end]
            spread = compute_spread(trading_slice, pair)
            z = (spread - pair["spread_mean"]) / pair["spread_std"]

            check_idx = start_idx + REVALIDATION_EVERY
            while check_idx < trading_end:
                if not revalidate(log_prices, pair, check_idx):
                    cutoff = check_idx - start_idx
                    z.iloc[cutoff:] = np.nan
                    break
                check_idx += REVALIDATION_EVERY

            signal = generate_pair_signal(z)
            pair_signals.append((pair, signal))

        if pair_signals:
            abs_frame = pd.concat(
                [s.abs().rename(i) for i, (p, s) in enumerate(pair_signals)], axis=1
            )
            total_abs = abs_frame.sum(axis=1)

            for pair, signal in pair_signals:
                scale = np.where(total_abs > 0, GROSS_TARGET / total_abs, 0.0)
                alloc = signal.abs() * scale
                direction = np.sign(signal)
                beta = pair["beta"]

                dep_w = direction * alloc / (1 + abs(beta))
                indep_w = -direction * beta * alloc / (1 + abs(beta))

                weights.loc[signal.index, pair["dependent"]] += dep_w.to_numpy()
                weights.loc[signal.index, pair["independent"]] += indep_w.to_numpy()

        start_idx += TRADING_DAYS

    print(f"\nformation/trading cycles: {cycle_count}")
    print(f"average pairs selected per cycle: {pair_count_total / cycle_count:.2f}")
    print(f"distinct pairs ever selected: {len(pair_frequency)} out of 55 possible")
    print("\npair selection frequency (out of {} cycles):".format(cycle_count))
    for (a, b), count in sorted(pair_frequency.items(), key=lambda kv: -kv[1]):
        print(f"  {a} ~ {b}: {count}")

    return weights

def main() -> None:
    log_prices = load_log_prices()
    log_returns = load_log_returns()

    weights = run_pairs_backtest(log_prices, log_returns)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    weights.to_parquet(OUTPUT_DIR / "pairs_trading_weights.parquet")

    active_count = (weights != 0).sum(axis=1)
    gross_exposure = weights.abs().sum(axis=1)
    net_exposure = weights.sum(axis=1)

    print(f"date range: {weights.index.min().date()} to {weights.index.max().date()}")
    print(f"average active legs per day: {active_count.mean():.2f}")
    print(f"max active legs per day: {int(active_count.max())}")
    print(f"days with zero active legs: {int((active_count == 0).sum())}")
    print(f"average gross exposure: {gross_exposure.mean():.4f}")
    print(f"max gross exposure: {gross_exposure.max():.4f}")
    print(f"average net exposure: {net_exposure.mean():.4f}")
    print(f"max abs net exposure: {net_exposure.abs().max():.4f}")


if __name__ == "__main__":
    main()