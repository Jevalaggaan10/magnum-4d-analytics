"""
Randomness Test Battery for Magnum 4D Historical Data.

Honest answer to "is there exploitable structure in past draws?":
run a battery of standard statistical tests and let the p-values speak.

If any test rejects H0 at alpha=0.05 the data deviates from the
null in that dimension - but a single low p-value out of many is also
expected under multiple testing (Bonferroni).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from data_loader import normalize_prize_number


PRIZE_COLS = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']
ALPHA = 0.05


def _conclude(p_value: float, alpha: float = ALPHA) -> str:
    if not np.isfinite(p_value):
        return 'not assessed: insufficient or degenerate data'
    if p_value < alpha:
        return f"reject H0 at alpha={alpha} (p={p_value:.4f})"
    return f"fail to reject H0 at alpha={alpha} (p={p_value:.4f}) - consistent with random"


def _all_digits(df: pd.DataFrame, columns: list = PRIZE_COLS) -> np.ndarray:
    """Flatten every digit from every draw column into a 1-D int array."""
    digits = []
    for col in columns:
        if col not in df.columns:
            continue
        for num in df[col].map(normalize_prize_number).dropna():
            digits.extend(int(c) for c in num)
    return np.asarray(digits, dtype=np.int8)


def _all_numbers(df: pd.DataFrame, columns: list = PRIZE_COLS) -> list[str]:
    nums = []
    for col in columns:
        if col not in df.columns:
            continue
        nums.extend(df[col].map(normalize_prize_number).dropna().tolist())
    return nums


def chi_square_position_uniformity(df: pd.DataFrame) -> dict:
    """Per-position digit uniformity. H0: each digit equally likely at each position."""
    out = {}
    for pos in range(4):
        observed = Counter()
        for col in PRIZE_COLS:
            if col not in df.columns:
                continue
            for num in df[col].map(normalize_prize_number).dropna():
                observed[num[pos]] += 1
        counts = np.array([observed.get(str(d), 0) for d in range(10)], dtype=float)
        expected = np.full(10, counts.sum() / 10.0)
        chi2, p = stats.chisquare(counts, expected) if counts.sum() else (float('nan'), float('nan'))
        out[f'position_{pos}'] = {
            'statistic': float(chi2),
            'p_value': float(p),
            'conclusion': _conclude(p),
            'counts': counts.tolist(),
        }
    return out


def chi_square_full_number(df: pd.DataFrame) -> dict:
    """Uniformity over 10,000 numbers. With ~30k draws, expected count ~3/number."""
    nums = _all_numbers(df)
    if not nums:
        return {'statistic': float('nan'), 'p_value': float('nan'), 'conclusion': 'no data'}
    observed = Counter(nums)
    counts = np.array([observed.get(f'{i:04d}', 0) for i in range(10000)], dtype=float)
    expected = np.full(10000, counts.sum() / 10000.0)
    chi2, p = stats.chisquare(counts, expected)
    sparse = expected[0] < 5
    if sparse:
        p = float('nan')
    return {
        'statistic': float(chi2),
        'p_value': float(p),
        'conclusion': ('not assessed: expected cell counts below 5; use a calibrated Monte Carlo null'
                       if sparse else _conclude(p)),
        'n_observations': int(counts.sum()),
        'mean_expected': float(expected[0]),
    }


def runs_test(df: pd.DataFrame) -> dict:
    """Wald-Wolfowitz runs test on digit stream (above/below median)."""
    digits = _all_digits(df)
    if len(digits) < 2:
        return {'statistic': float('nan'), 'p_value': float('nan'), 'conclusion': 'not enough data'}
    median = np.median(digits)
    binary = (digits > median).astype(int)
    n1 = int(binary.sum())
    n0 = int(len(binary) - n1)
    if n0 == 0 or n1 == 0:
        return {'statistic': float('nan'), 'p_value': float('nan'), 'conclusion': 'degenerate'}
    runs = 1 + int(np.sum(binary[1:] != binary[:-1]))
    mu = 2 * n0 * n1 / (n0 + n1) + 1
    var = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / (((n0 + n1) ** 2) * (n0 + n1 - 1))
    z = (runs - mu) / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {
        'statistic': float(z),
        'p_value': float(p),
        'conclusion': _conclude(p),
        'runs': runs,
        'expected_runs': float(mu),
    }


def autocorrelation_test(df: pd.DataFrame, lags: list = (1, 2, 3, 7, 30)) -> dict:
    """Ljung-Box autocorrelation on first-prize digit sums."""
    if '1stPrizeNo' not in df.columns:
        return {'conclusion': 'no 1stPrizeNo'}
    ordered = df.sort_values('DrawDate') if 'DrawDate' in df.columns else df
    numbers = ordered['1stPrizeNo'].map(normalize_prize_number).dropna()
    sums = numbers.apply(lambda s: sum(int(c) for c in s))
    usable_lags = sorted({int(lag) for lag in lags if 0 < int(lag) < len(sums)})
    if not usable_lags or sums.nunique() < 2:
        return {'conclusion': 'not enough nonconstant data for requested lags'}
    res = acorr_ljungbox(sums, lags=usable_lags, return_df=True)
    return {
        f'lag_{int(lag)}': {
            'statistic': float(res.loc[lag, 'lb_stat']),
            'p_value': float(res.loc[lag, 'lb_pvalue']),
            'conclusion': _conclude(float(res.loc[lag, 'lb_pvalue'])),
        }
        for lag in usable_lags
    }


def serial_independence_pairs(df: pd.DataFrame) -> dict:
    """Chi-square on adjacent-digit pairs. H0: P(d_{i+1}|d_i) = P(d_{i+1})."""
    digits = _all_digits(df)
    if len(digits) < 2:
        return {'conclusion': 'not enough data'}
    pairs = list(zip(digits[:-1], digits[1:]))
    contingency = np.zeros((10, 10), dtype=float)
    for a, b in pairs:
        contingency[a, b] += 1
    contingency = contingency[contingency.sum(axis=1) > 0][:, contingency.sum(axis=0) > 0]
    if min(contingency.shape) < 2:
        return {'statistic': float('nan'), 'p_value': float('nan'), 'conclusion': 'degenerate digit pairs'}
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    sparse = bool((expected < 5).any())
    if sparse:
        p = float('nan')
    return {
        'statistic': float(chi2),
        'p_value': float(p),
        'dof': int(dof),
        'conclusion': ('not assessed: sparse expected pair counts; use a calibrated Monte Carlo null'
                       if sparse else _conclude(p)),
    }


def benford_first_digit(df: pd.DataFrame) -> dict:
    """4D first digit should be uniform, NOT Benford. Test as sanity check."""
    if '1stPrizeNo' not in df.columns:
        return {'conclusion': 'no 1stPrizeNo'}
    first = df['1stPrizeNo'].map(normalize_prize_number).dropna().str[0]
    counts = np.array([(first == str(d)).sum() for d in range(1, 10)], dtype=float)
    benford = np.log10(1 + 1.0 / np.arange(1, 10))
    expected = benford / benford.sum() * counts.sum()
    chi2, p = stats.chisquare(counts, expected) if counts.sum() else (float('nan'), float('nan'))
    return {
        'statistic': float(chi2),
        'p_value': float(p),
        'note': "Expected to REJECT Benford for true 4D - uniform != Benford",
        'conclusion': ('not assessed: no valid nonzero first digits' if not np.isfinite(p)
                       else 'reject Benford; expected for uniform four-digit codes' if p < ALPHA
                       else 'Benford not rejected; this is not a lottery predictability test'),
    }


def run_all(df: pd.DataFrame) -> dict:
    return {
        'position_uniformity': chi_square_position_uniformity(df),
        'full_number_uniformity': chi_square_full_number(df),
        'runs_test': runs_test(df),
        'autocorrelation_ljung_box': autocorrelation_test(df),
        'serial_independence_pairs': serial_independence_pairs(df),
        'benford_first_digit': benford_first_digit(df),
    }


def summary_table(results: dict) -> pd.DataFrame:
    rows = []

    def add(name, sub):
        if isinstance(sub, dict) and 'p_value' in sub:
            rows.append({
                'test': name,
                'statistic': sub.get('statistic'),
                'p_value': sub.get('p_value'),
                'conclusion': sub.get('conclusion'),
            })
        elif isinstance(sub, dict):
            for k, v in sub.items():
                add(f"{name}.{k}", v)

    for key, val in results.items():
        add(key, val)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import load_combined_data, extract_top3_prizes
    df = load_combined_data()
    top3 = extract_top3_prizes(df)
    results = run_all(top3)
    table = summary_table(results)
    print(table.to_string(index=False))
