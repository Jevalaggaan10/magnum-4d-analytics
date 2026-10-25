"""
Feature Engineering Module for 4D Lottery Prediction
Creates features for ML models based on historical patterns
"""

import pandas as pd
import numpy as np
from collections import Counter
from itertools import permutations
from datetime import timedelta


def get_digit_features(number: str) -> dict:
    """Extract digit-level features from a 4-digit number."""
    digits = [int(d) for d in str(number).zfill(4)]
    
    return {
        'digit_sum': sum(digits),
        'digit_mean': np.mean(digits),
        'digit_std': np.std(digits),
        'digit_min': min(digits),
        'digit_max': max(digits),
        'digit_range': max(digits) - min(digits),
        'odd_count': sum(1 for d in digits if d % 2 == 1),
        'even_count': sum(1 for d in digits if d % 2 == 0),
        'high_count': sum(1 for d in digits if d >= 5),  # 5-9
        'low_count': sum(1 for d in digits if d < 5),     # 0-4
        'unique_digits': len(set(digits)),
        'has_zero': 1 if 0 in digits else 0,
        'has_repeat': 1 if len(set(digits)) < 4 else 0,
    }


def get_canonical_permutation(number: str) -> str:
    """Get the canonical (sorted) form of a number for permutation grouping."""
    return ''.join(sorted(str(number).zfill(4)))


def calculate_number_frequency(df: pd.DataFrame, window_days: int = None) -> dict:
    """Calculate frequency of each number in the dataset."""
    freq = Counter()
    
    if window_days:
        cutoff = df['DrawDate'].max() - timedelta(days=window_days)
        df = df[df['DrawDate'] >= cutoff]
    
    for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
        if col in df.columns:
            freq.update(df[col].astype(str).str.zfill(4))
    
    return dict(freq)


def calculate_digit_frequency(df: pd.DataFrame, position: int = None) -> dict:
    """Calculate frequency of each digit (0-9) across all numbers."""
    digit_freq = Counter()
    
    for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
        if col in df.columns:
            for num in df[col].astype(str).str.zfill(4):
                if position is not None:
                    digit_freq[num[position]] += 1
                else:
                    digit_freq.update(num)
    
    return dict(digit_freq)


def calculate_number_lag(df: pd.DataFrame, number: str, current_date: pd.Timestamp) -> int:
    """Calculate days since a number last appeared."""
    number = str(number).zfill(4)

    appearances = []
    for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
        if col in df.columns:
            matches = df[df[col].astype(str).str.zfill(4) == number]
            appearances.extend(matches['DrawDate'].tolist())

    if not appearances:
        return 9999  # Never appeared

    last_appearance = max(appearances)
    if last_appearance >= current_date:
        return 0

    return (current_date - last_appearance).days


def build_last_seen_index(df: pd.DataFrame, columns: list = None) -> dict:
    """Pre-build {number -> latest DrawDate} index. O(n) once, O(1) lookups."""
    if columns is None:
        columns = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']

    last_seen = {}
    sorted_df = df.sort_values('DrawDate')

    for col in columns:
        if col not in sorted_df.columns:
            continue
        series = sorted_df[col].astype(str).str.zfill(4)
        dates = sorted_df['DrawDate'].values
        for num, date in zip(series.values, dates):
            last_seen[num] = date  # later iterations overwrite, leaving latest

    return last_seen


def calculate_number_lag_indexed(last_seen: dict, number: str, current_date: pd.Timestamp) -> int:
    """O(1) lag lookup using pre-built index."""
    number = str(number).zfill(4)
    if number not in last_seen:
        return 9999
    last_date = pd.Timestamp(last_seen[number])
    if last_date >= current_date:
        return 0
    return (current_date - last_date).days


def calculate_permutation_frequency(df: pd.DataFrame) -> dict:
    """Calculate frequency of permutation families."""
    perm_freq = Counter()
    
    for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
        if col in df.columns:
            for num in df[col].astype(str).str.zfill(4):
                canonical = get_canonical_permutation(num)
                perm_freq[canonical] += 1
    
    return dict(perm_freq)


def create_time_features(date: pd.Timestamp) -> dict:
    """Extract time-based features from a date."""
    return {
        'day_of_week': date.dayofweek,
        'day_of_month': date.day,
        'month': date.month,
        'year': date.year,
        'is_weekend': 1 if date.dayofweek >= 5 else 0,
        'week_of_year': date.isocalendar()[1],
    }


def create_rolling_features(df: pd.DataFrame, windows: list = [7, 30, 90]) -> pd.DataFrame:
    """Create rolling window features for digit frequencies (vectorized)."""
    df = df.sort_values('DrawDate').reset_index(drop=True).copy()

    prize_cols = [c for c in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo'] if c in df.columns]
    padded = {c: df[c].astype(str).str.zfill(4) for c in prize_cols}

    for digit in range(10):
        digit_str = str(digit)
        total = sum(padded[c].str.count(digit_str) for c in prize_cols)
        df[f'digit_{digit}_count'] = total

        for window in windows:
            df[f'digit_{digit}_roll_{window}'] = (
                total.rolling(window=window, min_periods=1).mean()
            )

    return df


def create_feature_matrix_for_candidate(
    candidate: str, 
    df: pd.DataFrame, 
    current_date: pd.Timestamp
) -> dict:
    """Create a feature vector for a candidate number."""
    candidate = str(candidate).zfill(4)
    
    # Digit features
    features = get_digit_features(candidate)
    
    # Historical frequency
    freq_all = calculate_number_frequency(df)
    freq_30 = calculate_number_frequency(df, window_days=30)
    freq_90 = calculate_number_frequency(df, window_days=90)
    
    features['freq_all'] = freq_all.get(candidate, 0)
    features['freq_30'] = freq_30.get(candidate, 0)
    features['freq_90'] = freq_90.get(candidate, 0)
    
    # Lag
    features['lag_days'] = calculate_number_lag(df, candidate, current_date)
    
    # Permutation frequency
    perm_freq = calculate_permutation_frequency(df)
    canonical = get_canonical_permutation(candidate)
    features['perm_freq'] = perm_freq.get(canonical, 0)
    
    # Time features
    features.update(create_time_features(current_date))
    
    return features


def get_hot_cold_numbers(df: pd.DataFrame, top_n: int = 100) -> tuple:
    """Get the hottest (most frequent) and coldest (least frequent) numbers."""
    freq = calculate_number_frequency(df)
    
    sorted_nums = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    hot = [num for num, _ in sorted_nums[:top_n]]
    cold = [num for num, _ in sorted_nums[-top_n:]]
    
    return hot, cold


def get_overdue_numbers(df: pd.DataFrame, current_date: pd.Timestamp, min_gap: int = 180) -> list:
    """Get numbers that haven't appeared in a long time but have appeared before."""
    freq = calculate_number_frequency(df)
    
    overdue = []
    for num in freq.keys():
        lag = calculate_number_lag(df, num, current_date)
        if lag >= min_gap:
            overdue.append((num, lag, freq[num]))
    
    # Sort by frequency (higher = more "due") then by lag (higher = more overdue)
    overdue.sort(key=lambda x: (x[2], x[1]), reverse=True)
    
    return overdue


if __name__ == "__main__":
    from data_loader import load_combined_data, extract_top3_prizes
    
    # Load data
    df = load_combined_data("c:/codex/lottery")
    top3 = extract_top3_prizes(df)
    
    # Test features
    print("\n=== Digit Frequencies ===")
    digit_freq = calculate_digit_frequency(top3)
    for digit in sorted(digit_freq.keys()):
        print(f"  Digit {digit}: {digit_freq[digit]}")
    
    print("\n=== Hot Numbers (Top 20) ===")
    hot, cold = get_hot_cold_numbers(top3, top_n=20)
    print(f"  Hot: {hot[:10]}")
    
    print("\n=== Overdue Numbers (Top 10) ===")
    current_date = top3['DrawDate'].max()
    overdue = get_overdue_numbers(top3, current_date, min_gap=365)
    for num, lag, freq in overdue[:10]:
        print(f"  {num}: {lag} days since last, appeared {freq} times total")
