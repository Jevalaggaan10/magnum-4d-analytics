"""
Cross-method consensus scoring.

For each of the 10,000 possible numbers, count how many methods ranked it in
their top-N and produce a weighted score where higher-ranked entries count
for more.

IMPORTANT: agreement across methods does NOT translate to a predictive edge.
Use the backtester to measure performance. See `backtest.py`.
"""

from __future__ import annotations

from collections import Counter
import re

import pandas as pd

from predictor_suite import MethodResult


DEFAULT_WEIGHTS = {
    'frequency': 1.0,
    'recent_hot_90d': 1.0,
    'overdue': 0.6,
    'digit_probability': 0.8,
    'permutation_family': 0.7,
    'markov': 1.0,
    'arima': 0.5,
    'bayesian': 1.0,
    'xgboost': 1.2,
    'lstm': 1.0,
}


def build_consensus(
    results: dict[str, MethodResult],
    top_n: int = 50,
    weights: dict[str, float] = None,
) -> pd.DataFrame:
    """Return DataFrame: number, agreement_count, weighted_score, methods."""
    weights = DEFAULT_WEIGHTS if weights is None else weights

    agreement: dict[str, int] = Counter()
    weighted: dict[str, float] = Counter()
    methods_for: dict[str, list[str]] = {}

    for method, res in results.items():
        if not res.top:
            continue
        w = weights.get(method, 1.0)
        seen = set()
        for rank, (num, _) in enumerate(res.top[:top_n]):
            if not isinstance(num, str) or re.fullmatch(r'[0-9]{4}', num) is None or num in seen:
                continue
            seen.add(num)
            agreement[num] += 1
            weighted[num] += w / (rank + 1)
            methods_for.setdefault(num, []).append(method)

    if not agreement:
        return pd.DataFrame(columns=['number', 'agreement_count', 'weighted_score', 'methods'])

    rows = []
    for num, cnt in agreement.items():
        rows.append({
            'number': num,
            'agreement_count': cnt,
            'weighted_score': weighted[num],
            'methods': ','.join(methods_for[num]),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(
        ['agreement_count', 'weighted_score', 'number'],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return df


def best_number(consensus_df: pd.DataFrame) -> str | None:
    if consensus_df.empty:
        return None
    return consensus_df.iloc[0]['number']


if __name__ == "__main__":
    from data_loader import load_combined_data, extract_top3_prizes
    from predictor_suite import run_all

    df = load_combined_data()
    top3 = extract_top3_prizes(df)
    results = run_all(top3, include_ml=False, top_n=50)
    cons = build_consensus(results, top_n=50)
    print(cons.head(20).to_string(index=False))
    print(f"\nBest consensus number: {best_number(cons)}")
    print("REMINDER: consensus is descriptive, not predictive.")
