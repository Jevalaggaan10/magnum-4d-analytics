"""
Walk-forward backtest for every method in predictor_suite.

For each test year, train on data strictly before that year and predict the
top-N numbers. Score = fraction of draws in the test year where the top-N
contained any of the 1st/2nd/3rd prize numbers.

Compare each method against a uniform-random baseline using a two-proportion
z-test. With Bonferroni correction for multiple comparisons.

Honest expectation: no method significantly beats random.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

import predictor_suite as ps


PRIZE_COLS = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']


@dataclass
class BacktestRow:
    method: str
    year: int
    n_test_draws: int
    hit_rate: float
    random_baseline: float
    p_value: float
    significant: bool


def _draws_in_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    return df[df['DrawDate'].dt.year == year]


def _hit_rate(top_set: set[str], test_df: pd.DataFrame) -> float:
    """Fraction of draws in test_df where ANY top-3 prize is in top_set."""
    if test_df.empty:
        return 0.0
    hits = 0
    for _, row in test_df.iterrows():
        prizes = {str(row[c]).zfill(4) for c in PRIZE_COLS if c in test_df.columns}
        if prizes & top_set:
            hits += 1
    return hits / len(test_df)


def _random_baseline_rate(n_pick: int, n_draws: int, seed: int = 0) -> float:
    """Closed-form: P(any of 3 prizes in random n_pick) = 1 - C(10000-n_pick,3)/C(10000,3)."""
    from math import comb
    total = comb(10000, 3)
    miss = comb(10000 - n_pick, 3)
    return 1 - miss / total


def run_year(
    df: pd.DataFrame,
    test_year: int,
    top_n: int = 100,
    include_ml: bool = False,
    lstm_epochs: int = 2,
) -> list[BacktestRow]:
    """Train methods on data < test_year, evaluate on draws in test_year."""
    train_df = df[df['DrawDate'].dt.year < test_year]
    test_df = _draws_in_year(df, test_year)
    if train_df.empty or test_df.empty:
        return []

    print(f"\n=== Backtest year {test_year}: train={len(train_df)} test={len(test_df)} ===")

    results = ps.run_all(train_df, include_ml=include_ml, top_n=top_n, lstm_epochs=lstm_epochs)

    # Add a uniform-random control method
    rng = np.random.default_rng(test_year)
    random_picks = [f'{i:04d}' for i in rng.choice(10000, size=top_n, replace=False)]
    results['_random_control'] = ps.MethodResult(
        name='_random_control',
        top=[(num, 1.0 / (i + 1)) for i, num in enumerate(random_picks)],
        explanation='Uniform random control',
        family='control',
    )

    baseline_rate = _random_baseline_rate(top_n, len(test_df))
    n_test = len(test_df)

    rows = []
    for method, res in results.items():
        top_set = set(res.numbers(top_n))
        if not top_set:
            rows.append(BacktestRow(method, test_year, n_test, 0.0, baseline_rate, float('nan'), False))
            continue

        rate = _hit_rate(top_set, test_df)
        hits = int(round(rate * n_test))
        expected_hits = baseline_rate * n_test
        # Two-sided binomial test vs baseline
        if n_test > 0:
            p_val = float(stats.binomtest(hits, n_test, baseline_rate).pvalue)
        else:
            p_val = float('nan')
        rows.append(BacktestRow(method, test_year, n_test, rate, baseline_rate, p_val, False))

    return rows


def run(
    df: pd.DataFrame,
    years: list[int],
    top_n: int = 100,
    include_ml: bool = False,
) -> pd.DataFrame:
    all_rows: list[BacktestRow] = []
    for y in years:
        all_rows.extend(run_year(df, y, top_n=top_n, include_ml=include_ml))

    out = pd.DataFrame([r.__dict__ for r in all_rows])
    if out.empty:
        return out

    # Bonferroni across methods x years for multiple testing
    n_tests = (out['method'] != '_random_control').sum()
    alpha_corrected = 0.05 / max(n_tests, 1)
    out['significant'] = (out['p_value'] < alpha_corrected) & (out['hit_rate'] > out['random_baseline'])
    out['alpha_bonferroni'] = alpha_corrected
    return out


def summary(bt_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across years per method."""
    if bt_df.empty:
        return bt_df
    agg = bt_df.groupby('method').agg(
        mean_hit_rate=('hit_rate', 'mean'),
        std_hit_rate=('hit_rate', 'std'),
        mean_baseline=('random_baseline', 'mean'),
        n_years=('year', 'count'),
        any_significant=('significant', 'any'),
    ).reset_index()
    agg['edge_over_random'] = agg['mean_hit_rate'] - agg['mean_baseline']
    agg = agg.sort_values('edge_over_random', ascending=False)
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', default='2020-2023', help='start-end inclusive')
    parser.add_argument('--top-n', type=int, default=100)
    parser.add_argument('--ml', action='store_true', help='Include XGBoost/LSTM (slow)')
    args = parser.parse_args()

    start, end = (int(s) for s in args.years.split('-'))
    years = list(range(start, end + 1))

    from data_loader import load_combined_data, extract_top3_prizes
    df = load_combined_data("c:/codex/lottery")
    top3 = extract_top3_prizes(df)

    bt = run(top3, years, top_n=args.top_n, include_ml=args.ml)
    print("\n=== Per-year results ===")
    print(bt.to_string(index=False))

    s = summary(bt)
    print("\n=== Summary across years ===")
    print(s.to_string(index=False))

    n_sig = int(bt['significant'].sum())
    print(f"\nMethods significantly beating random (Bonferroni-corrected): {n_sig}")
    print("Expected under H0: ~0. If positive, suspect overfit, multiple-testing, or rare luck.")
