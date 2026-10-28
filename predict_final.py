"""
Final Prediction Script for 4D Lottery - Improved Version
Uses refined models and better ensemble strategy
"""

import pandas as pd
import numpy as np
from collections import Counter
from datetime import timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)


def load_data(data_dir: str = None):
    """Load and preprocess data."""
    from data_loader import load_combined_data, extract_top3_prizes
    df = load_combined_data(data_dir)
    return extract_top3_prizes(df)


def get_frequency_based_predictions(df: pd.DataFrame, n: int = 30) -> list:
    """Get predictions based on historical frequency."""
    freq = Counter()
    for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
        freq.update(df[col].astype(str).str.zfill(4))
    
    return [num for num, _ in freq.most_common(n)]


def get_recent_hot_numbers(df: pd.DataFrame, days: int = 90, n: int = 20) -> list:
    """Get numbers that appeared frequently in recent period."""
    cutoff = df['DrawDate'].max() - timedelta(days=days)
    recent = df[df['DrawDate'] >= cutoff]
    
    freq = Counter()
    for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
        freq.update(recent[col].astype(str).str.zfill(4))
    
    return [num for num, _ in freq.most_common(n)]


def get_overdue_numbers(df: pd.DataFrame, min_freq: int = 3, n: int = 20) -> list:
    """Get numbers that haven't appeared in a long time but have appeared before."""
    # Get last appearance date for each number
    last_seen = {}
    freq = Counter()
    
    for _, row in df.sort_values('DrawDate').iterrows():
        date = row['DrawDate']
        for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
            num = str(row[col]).zfill(4)
            last_seen[num] = date
            freq[num] += 1
    
    current_date = df['DrawDate'].max()
    
    # Filter for numbers with minimum frequency
    candidates = []
    for num, count in freq.items():
        if count >= min_freq:
            days_since = (current_date - last_seen[num]).days
            candidates.append((num, days_since, count))
    
    # Sort by days since last appearance (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    return [num for num, _, _ in candidates[:n]]


def get_digit_probability_numbers(df: pd.DataFrame, n: int = 20) -> list:
    """Generate numbers based on digit position probabilities."""
    # Calculate probability for each digit at each position
    position_probs = [Counter() for _ in range(4)]
    
    for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
        for num in df[col].astype(str).str.zfill(4):
            for i, digit in enumerate(num):
                position_probs[i][digit] += 1
    
    # Normalize to probabilities
    for i in range(4):
        total = sum(position_probs[i].values())
        if not total:
            raise ValueError('digit sampling requires valid historical numbers')
        for digit in position_probs[i]:
            position_probs[i][digit] /= total
    support = int(np.prod([len(probabilities) for probabilities in position_probs]))
    if not 0 <= n <= support:
        raise ValueError(f'cannot sample {n} distinct numbers from support of {support}')
    
    # Generate numbers
    generated = set()
    while len(generated) < n:
        num = ''
        for i in range(4):
            digits = list(position_probs[i].keys())
            probs = [position_probs[i][d] for d in digits]
            num += np.random.choice(digits, p=probs)
        generated.add(num)
    
    return list(generated)


def get_permutation_based_numbers(df: pd.DataFrame, n: int = 20) -> list:
    """Get numbers from hot permutation families."""
    # Count permutation families
    perm_freq = Counter()
    perm_to_nums = {}
    
    for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
        for num in df[col].astype(str).str.zfill(4):
            perm = ''.join(sorted(num))
            perm_freq[perm] += 1
            if perm not in perm_to_nums:
                perm_to_nums[perm] = []
            perm_to_nums[perm].append(num)
    
    # Get top permutation families and sample numbers from them
    result = []
    for perm, count in perm_freq.most_common(n * 2):
        # Get a random arrangement of this permutation
        nums_in_family = perm_to_nums[perm]
        result.append(random.choice(nums_in_family))
        if len(result) >= n:
            break
    
    return result


def get_pattern_based_numbers(df: pd.DataFrame, n: int = 20) -> list:
    """Generate numbers based on observed patterns."""
    results = []
    
    # Random pattern demonstration; no learned pattern weights are used.
    for _ in range(n // 4):
        # XXYY pattern
        x, y = random.randint(0, 9), random.randint(0, 9)
        results.append(f'{x}{x}{y}{y}')
        
        # XYXY pattern
        x, y = random.randint(0, 9), random.randint(0, 9)
        results.append(f'{x}{y}{x}{y}')
    
    return results[:n]


def calculate_ensemble_score(all_predictions: dict) -> list:
    """Calculate ensemble scores with weighted voting."""
    scores = Counter()
    
    weights = {
        'frequency': 1.0,
        'recent_hot': 1.2,
        'overdue': 0.8,
        'digit_prob': 0.6,
        'permutation': 0.7,
        'pattern': 0.5,
    }
    
    for method, numbers in all_predictions.items():
        weight = weights.get(method, 1.0)
        for rank, num in enumerate(numbers):
            # Higher rank = lower score contribution
            rank_weight = 1.0 / (rank + 1)
            scores[num] += weight * rank_weight
    
    # Sort by score
    sorted_nums = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_nums


def main():
    """Main prediction function."""
    print("=" * 60)
    print("MAGNUM 4D PREDICTION - IMPROVED ENSEMBLE MODEL")
    print("Target: next draw (descriptive rankings only)")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading historical data...")
    df = load_data()
    print(f"   Loaded {len(df)} draws")
    print(f"   Date range: {df['DrawDate'].min().date()} to {df['DrawDate'].max().date()}")
    
    # Generate predictions from each method
    print("\n🔮 Generating predictions from multiple methods...")
    
    predictions = {}
    
    print("   • Frequency analysis...")
    predictions['frequency'] = get_frequency_based_predictions(df, n=30)
    
    print("   • Recent hot numbers (last 90 days)...")
    predictions['recent_hot'] = get_recent_hot_numbers(df, days=90, n=20)
    
    print("   • Overdue numbers (long gap)...")
    predictions['overdue'] = get_overdue_numbers(df, min_freq=3, n=20)
    
    print("   • Digit probability model...")
    predictions['digit_prob'] = get_digit_probability_numbers(df, n=20)
    
    print("   • Permutation family analysis...")
    predictions['permutation'] = get_permutation_based_numbers(df, n=20)
    
    print("   • Pattern recognition...")
    predictions['pattern'] = get_pattern_based_numbers(df, n=20)
    
    # Ensemble
    print("\n⚡ Computing ensemble predictions...")
    ensemble = calculate_ensemble_score(predictions)
    
    # Results
    print("\n" + "=" * 60)
    print("🎯 TOP 15 PREDICTED NUMBERS FOR TODAY'S DRAW")
    print("=" * 60)
    
    for i, (num, score) in enumerate(ensemble[:15], 1):
        stars = "⭐" * min(5, int(score + 1))
        print(f"  {i:2d}. {num}   Score: {score:.3f}  {stars}")
    
    print("\n" + "-" * 60)
    print("📋 EXTENDED PREDICTIONS (16-30)")
    print("-" * 60)
    for i, (num, score) in enumerate(ensemble[15:30], 16):
        print(f"  {i:2d}. {num}   Score: {score:.3f}")
    
    # Prize category suggestions
    print("\n" + "=" * 60)
    print("🏆 SUGGESTED NUMBERS BY PRIZE CATEGORY")
    print("=" * 60)
    
    top_15 = [num for num, _ in ensemble[:15]]
    
    print("\n🥇 1st Prize Candidates (Top 5):")
    for num in top_15[:5]:
        print(f"     {num}")
    
    print("\n🥈 2nd Prize Candidates (6-10):")
    for num in top_15[5:10]:
        print(f"     {num}")
    
    print("\n🥉 3rd Prize Candidates (11-15):")
    for num in top_15[10:15]:
        print(f"     {num}")
    
    # Method breakdown
    print("\n" + "=" * 60)
    print("📊 PREDICTIONS BY METHOD")
    print("=" * 60)
    
    for method, nums in predictions.items():
        print(f"\n{method.upper().replace('_', ' ')}:")
        print(f"   {nums[:10]}")
    
    # Historical stats
    print("\n" + "=" * 60)
    print("📈 STATISTICAL INSIGHTS")
    print("=" * 60)
    
    print("\n🔥 Historically Hottest (All Time):")
    for num in predictions['frequency'][:5]:
        count = sum(1 for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo'] 
                   for n in df[col].astype(str).str.zfill(4) if n == num)
        print(f"     {num}: appeared {count} times")
    
    print("\n🌡️ Recent Hot (Last 90 Days):")
    for num in predictions['recent_hot'][:5]:
        print(f"     {num}")
    
    print("\n❄️ Overdue (Long Gap But Frequent):")
    for num in predictions['overdue'][:5]:
        print(f"     {num}")
    
    print("\n" + "=" * 60)
    print("⚠️  IMPORTANT DISCLAIMER")
    print("=" * 60)
    print("These predictions are based on historical patterns only.")
    print("Lottery draws are RANDOM and past performance does NOT")
    print("predict future results. Please gamble responsibly.")
    print("=" * 60)
    
    return ensemble


if __name__ == "__main__":
    main()
