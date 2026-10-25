"""
Unified predictor suite for Magnum 4D.

Wraps every method (statistical, ML, time-series, Bayesian) behind one
uniform interface so the UI and the backtester can iterate over them.

NONE of these methods beat the random baseline on i.i.d. data.
They are scored for ranking only, not for predictive edge.
"""

from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MethodResult:
    name: str
    top: list[tuple[str, float]]
    explanation: str
    family: str = "statistical"

    def top_n(self, n: int = 20) -> list[tuple[str, float]]:
        return self.top[:n]

    def numbers(self, n: int = 20) -> list[str]:
        return [num for num, _ in self.top[:n]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_numbers_iter(df: pd.DataFrame, cols=('1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo')):
    for col in cols:
        if col in df.columns:
            for n in df[col].astype(str).str.zfill(4):
                yield n


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------

def method_frequency(df: pd.DataFrame, top_n: int = 100) -> MethodResult:
    freq = Counter(_all_numbers_iter(df))
    ranked = freq.most_common(top_n)
    return MethodResult(
        name="frequency_alltime",
        family="statistical",
        top=[(num, float(c)) for num, c in ranked],
        explanation="Top numbers by total historical appearances across 1st/2nd/3rd prize columns.",
    )


def method_recent_hot(df: pd.DataFrame, days: int = 90, top_n: int = 100) -> MethodResult:
    from datetime import timedelta
    cutoff = df['DrawDate'].max() - timedelta(days=days)
    recent = df[df['DrawDate'] >= cutoff]
    freq = Counter(_all_numbers_iter(recent))
    ranked = freq.most_common(top_n)
    return MethodResult(
        name=f"recent_hot_{days}d",
        family="statistical",
        top=[(num, float(c)) for num, c in ranked],
        explanation=f"Most frequent numbers in the last {days} days.",
    )


def method_overdue(df: pd.DataFrame, top_n: int = 100, min_freq: int = 2) -> MethodResult:
    from feature_engineer import build_last_seen_index
    last_seen = build_last_seen_index(df)
    freq = Counter(_all_numbers_iter(df))
    current = df['DrawDate'].max()

    rows = []
    for num, count in freq.items():
        if count < min_freq:
            continue
        days_since = (current - pd.Timestamp(last_seen[num])).days
        rows.append((num, days_since))
    rows.sort(key=lambda x: x[1], reverse=True)
    return MethodResult(
        name="overdue_gap",
        family="statistical",
        top=[(num, float(d)) for num, d in rows[:top_n]],
        explanation="Numbers with longest gap since last appearance (gambler's fallacy test).",
    )


def method_digit_probability(df: pd.DataFrame, top_n: int = 100, seed: int = 42) -> MethodResult:
    rng = np.random.default_rng(seed)
    pos_counts = [Counter() for _ in range(4)]
    for num in _all_numbers_iter(df):
        for i, c in enumerate(num):
            pos_counts[i][c] += 1
    pos_probs = []
    for c in pos_counts:
        total = sum(c.values())
        pos_probs.append({d: c[d] / total for d in '0123456789'})

    # Score every number = product of per-position probabilities, top-N by score
    scores = []
    for i in range(10000):
        num = f'{i:04d}'
        p = 1.0
        for k, ch in enumerate(num):
            p *= pos_probs[k].get(ch, 0.0)
        scores.append((num, p))
    scores.sort(key=lambda x: x[1], reverse=True)
    return MethodResult(
        name="digit_position_probability",
        family="statistical",
        top=scores[:top_n],
        explanation="Score = product of per-position digit empirical probabilities.",
    )


def method_permutation_family(df: pd.DataFrame, top_n: int = 100, seed: int = 42) -> MethodResult:
    rng = np.random.default_rng(seed)
    perm_freq = Counter()
    perm_to_nums = {}
    for num in _all_numbers_iter(df):
        perm = ''.join(sorted(num))
        perm_freq[perm] += 1
        perm_to_nums.setdefault(perm, []).append(num)

    result = []
    seen = set()
    for perm, count in perm_freq.most_common(top_n * 4):
        for cand in perm_to_nums[perm]:
            if cand not in seen:
                seen.add(cand)
                result.append((cand, float(count)))
                if len(result) >= top_n:
                    break
        if len(result) >= top_n:
            break
    return MethodResult(
        name="permutation_family",
        family="statistical",
        top=result,
        explanation="Numbers belonging to the most frequently occurring digit-multisets.",
    )


def method_markov(df: pd.DataFrame, top_n: int = 100) -> MethodResult:
    from train_model import MarkovChainModel
    m = MarkovChainModel()
    m.train(df)
    scores = []
    for i in range(10000):
        num = f'{i:04d}'
        scores.append((num, m.get_number_score(num)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return MethodResult(
        name="markov_chain",
        family="ml",
        top=scores[:top_n],
        explanation="Score = P(d0)*P(d1|d0)*P(d2|d1)*P(d3|d2) under empirical transition matrix.",
    )


def method_xgboost(df: pd.DataFrame, top_n: int = 100) -> MethodResult:
    try:
        from train_model import XGBoostModel, HAS_XGBOOST
    except ImportError:
        HAS_XGBOOST = False
    if not HAS_XGBOOST:
        return MethodResult(
            name="xgboost",
            family="ml",
            top=[],
            explanation="XGBoost not installed.",
        )
    from train_model import XGBoostModel
    m = XGBoostModel()
    m.train(df)
    preds = m.predict(n=top_n, df=df)
    return MethodResult(
        name="xgboost",
        family="ml",
        top=[(num, float(p)) for num, p in preds],
        explanation="Binary classifier (appeared vs never-appeared) scoring all 10,000 numbers.",
    )


def method_lstm(df: pd.DataFrame, top_n: int = 50, epochs: int = 3) -> MethodResult:
    try:
        from train_model import LSTMModel, HAS_TENSORFLOW
    except ImportError:
        HAS_TENSORFLOW = False
    if not HAS_TENSORFLOW:
        return MethodResult(
            name="lstm",
            family="ml",
            top=[],
            explanation="TensorFlow not installed.",
        )
    from train_model import LSTMModel
    m = LSTMModel(sequence_length=40)
    m.train(df, epochs=epochs)
    preds = m.predict(df=df, n=top_n)
    # score by predicted joint prob
    scored = [(num, m.get_number_score(df, num)) for num in preds]
    scored.sort(key=lambda x: x[1], reverse=True)
    return MethodResult(
        name="lstm",
        family="ml",
        top=scored,
        explanation="Real LSTM over per-digit one-hot stream; numbers sampled from learned distribution.",
    )


def method_arima_digit_sum(df: pd.DataFrame, top_n: int = 100) -> MethodResult:
    """ARIMA on digit-sum series of 1st prize. Score numbers whose digit-sum
    matches the next-step forecast best."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return MethodResult("arima_digit_sum", [], "statsmodels missing.", family="ts")
    if '1stPrizeNo' not in df.columns:
        return MethodResult("arima_digit_sum", "ts", [], "no 1stPrizeNo")
    s = df['1stPrizeNo'].astype(str).str.zfill(4).apply(lambda n: sum(int(c) for c in n)).values
    try:
        fit = ARIMA(s, order=(2, 0, 2)).fit()
        forecast = float(fit.forecast(1)[0])
    except Exception:
        forecast = float(np.mean(s))
    # Score every number by inverse abs distance of its digit-sum to forecast
    scores = []
    for i in range(10000):
        num = f'{i:04d}'
        ds = sum(int(c) for c in num)
        score = 1.0 / (1.0 + abs(ds - forecast))
        scores.append((num, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return MethodResult(
        name="arima_digit_sum",
        family="ts",
        top=scores[:top_n],
        explanation=f"ARIMA(2,0,2) next-step digit-sum forecast = {forecast:.2f}. Score = 1/(1+|digit_sum - forecast|).",
    )


def method_bayesian_dirichlet(df: pd.DataFrame, top_n: int = 100, alpha: float = 1.0) -> MethodResult:
    """Dirichlet-multinomial posterior mean per position.

    Posterior pred. P(d at pos i) = (count_i_d + alpha) / (N_i + 10*alpha).
    """
    pos_counts = [Counter() for _ in range(4)]
    for num in _all_numbers_iter(df):
        for i, c in enumerate(num):
            pos_counts[i][c] += 1
    pos_probs = []
    for c in pos_counts:
        total = sum(c.values())
        denom = total + 10 * alpha
        pos_probs.append({d: (c[d] + alpha) / denom for d in '0123456789'})

    scores = []
    for i in range(10000):
        num = f'{i:04d}'
        p = 1.0
        for k, ch in enumerate(num):
            p *= pos_probs[k][ch]
        scores.append((num, p))
    scores.sort(key=lambda x: x[1], reverse=True)
    return MethodResult(
        name=f"bayesian_dirichlet_a{alpha}",
        family="bayes",
        top=scores[:top_n],
        explanation=f"Dirichlet-multinomial posterior (alpha={alpha}) over per-position digit distribution.",
    )


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

ALL_METHODS: dict[str, Callable[[pd.DataFrame], MethodResult]] = {
    'frequency': method_frequency,
    'recent_hot_90d': method_recent_hot,
    'overdue': method_overdue,
    'digit_probability': method_digit_probability,
    'permutation_family': method_permutation_family,
    'markov': method_markov,
    'arima': method_arima_digit_sum,
    'bayesian': method_bayesian_dirichlet,
}

ML_METHODS: dict[str, Callable[[pd.DataFrame], MethodResult]] = {
    'xgboost': method_xgboost,
    'lstm': method_lstm,
}


def run_all(df: pd.DataFrame, include_ml: bool = True, top_n: int = 100, lstm_epochs: int = 3) -> dict[str, MethodResult]:
    """Run every method. ML methods optional (slow)."""
    out: dict[str, MethodResult] = {}
    for key, fn in ALL_METHODS.items():
        try:
            if key == 'recent_hot_90d':
                out[key] = fn(df, days=90, top_n=top_n)
            else:
                out[key] = fn(df, top_n=top_n)
        except Exception as e:
            out[key] = MethodResult(name=key, top=[], explanation=f"ERROR: {e}", family="error")

    if include_ml:
        try:
            out['xgboost'] = method_xgboost(df, top_n=top_n)
        except Exception as e:
            out['xgboost'] = MethodResult("xgboost", [], f"ERROR: {e}", family="error")
        try:
            out['lstm'] = method_lstm(df, top_n=min(top_n, 50), epochs=lstm_epochs)
        except Exception as e:
            out['lstm'] = MethodResult("lstm", [], f"ERROR: {e}", family="error")

    return out


if __name__ == "__main__":
    from data_loader import load_combined_data, extract_top3_prizes
    df = load_combined_data("c:/codex/lottery")
    top3 = extract_top3_prizes(df)
    results = run_all(top3, include_ml=False, top_n=20)
    for name, r in results.items():
        print(f"\n--- {name} ({r.family}) ---")
        print(f"  {r.explanation}")
        print(f"  Top 10: {[n for n, _ in r.top[:10]]}")
