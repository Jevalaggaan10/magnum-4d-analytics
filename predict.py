"""
Prediction Module for 4D Lottery
Generates predictions for the next draw using all trained models
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import sys


def ensemble_predictions(all_predictions: dict, top_n: int = 20) -> list:
    """Combine predictions from all models using voting."""
    vote_counter = Counter()
    
    for model_name, preds in all_predictions.items():
        if isinstance(preds[0], tuple):
            # XGBoost returns (number, probability)
            for num, prob in preds:
                vote_counter[num] += prob * 2  # Weight by probability
        else:
            # Other models return just numbers
            for i, num in enumerate(preds):
                # Higher weight for earlier predictions
                vote_counter[num] += 1.0 / (i + 1)
    
    # Sort by vote count
    sorted_nums = sorted(vote_counter.items(), key=lambda x: x[1], reverse=True)
    
    return [(num, score) for num, score in sorted_nums[:top_n]]


def generate_predictions(data_dir: str = ".", target_date: str = None):
    """Main prediction function."""
    from data_loader import load_combined_data, extract_top3_prizes
    from train_model import train_all_models
    
    print("="*60)
    print("4D LOTTERY PREDICTION SYSTEM")
    print("="*60)
    
    if target_date:
        print(f"Target Date: {target_date}")
    else:
        print(f"Target Date: Next Draw")
    print()
    
    # Load data
    print("Loading historical data...")
    df = load_combined_data(data_dir)
    top3 = extract_top3_prizes(df)
    
    # Train models
    print("\nTraining prediction models...")
    models = train_all_models(top3)
    
    # Generate predictions from each model
    all_predictions = {}
    
    print("\n" + "-"*60)
    print("INDIVIDUAL MODEL PREDICTIONS")
    print("-"*60)
    
    for name, model in models.items():
        print(f"\n### {model.name} Model ###")
        
        if name == 'xgboost':
            preds = model.predict(n=20, df=top3)
            all_predictions[name] = preds
            print("Top 10 predictions:")
            for i, (num, prob) in enumerate(preds[:10], 1):
                print(f"  {i:2d}. {num}  (confidence: {prob:.3f})")
        elif name == 'lstm':
            preds = model.predict(df=top3, n=20)
            all_predictions[name] = preds
            print(f"Predictions: {preds[:10]}")
        else:
            preds = model.predict(n=20)
            all_predictions[name] = preds
            print(f"Predictions: {preds[:10]}")
    
    # Ensemble predictions
    print("\n" + "="*60)
    print("ENSEMBLE PREDICTIONS (Combined from all models)")
    print("="*60)
    
    ensemble = ensemble_predictions(all_predictions, top_n=30)
    
    print("\n🎯 TOP 10 PREDICTED NUMBERS FOR NEXT DRAW:")
    print("-"*40)
    for i, (num, score) in enumerate(ensemble[:10], 1):
        print(f"  {i:2d}. {num}  (ensemble score: {score:.3f})")
    
    print("\n📊 Extended Predictions (11-30):")
    for i, (num, score) in enumerate(ensemble[10:30], 11):
        print(f"  {i:2d}. {num}  (score: {score:.3f})")
    
    # Category breakdown
    print("\n" + "="*60)
    print("PREDICTIONS BY PRIZE CATEGORY SUGGESTION")
    print("="*60)
    
    print("\n🥇 1st Prize Candidates (highest confidence):")
    for num, score in ensemble[:5]:
        print(f"     {num}")
    
    print("\n🥈 2nd Prize Candidates:")
    for num, score in ensemble[5:10]:
        print(f"     {num}")
    
    print("\n🥉 3rd Prize Candidates:")
    for num, score in ensemble[10:15]:
        print(f"     {num}")
    
    # Hot numbers analysis
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)
    
    stat_model = models.get('statistical')
    if stat_model:
        # Most frequent historically
        freq_sorted = sorted(stat_model.number_freq.items(), key=lambda x: x[1], reverse=True)
        print("\n🔥 Historically Hottest Numbers (appeared most often):")
        for num, count in freq_sorted[:10]:
            print(f"     {num}: appeared {count} times")
        
        # Recently cold (long gaps)
        print("\n❄️ Cold Numbers (potential 'due' numbers):")
        from feature_engineer import get_overdue_numbers
        current_date = top3['DrawDate'].max()
        overdue = get_overdue_numbers(top3, current_date, min_gap=180)
        for num, lag, freq in overdue[:10]:
            print(f"     {num}: {lag} days since last appearance (appeared {freq}x total)")
    
    print("\n" + "="*60)
    print("⚠️  DISCLAIMER")
    print("="*60)
    print("Lottery numbers are RANDOM. These predictions are based on")
    print("statistical patterns in historical data and do NOT guarantee")
    print("any wins. Please gamble responsibly.")
    print("="*60)
    
    return ensemble


if __name__ == "__main__":
    data_dir = "c:/codex/lottery"
    target_date = "2026-01-03"  # Today
    
    if len(sys.argv) > 1:
        target_date = sys.argv[1]
    
    predictions = generate_predictions(data_dir, target_date)
