"""
Data Loader Module for 4D Lottery Prediction
Loads and preprocesses historical draw data from Magnum 4D
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path


def load_4d_txt(filepath: str = "4D.txt") -> pd.DataFrame:
    """Load the 4D.txt file containing Magnum draw history."""
    df = pd.read_csv(filepath)
    
    # Parse date column
    df['DrawDate'] = pd.to_datetime(df['DrawDate'], format='%Y%m%d')
    
    # Ensure prize columns are strings (preserve leading zeros)
    prize_cols = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']
    for col in prize_cols:
        df[col] = df[col].astype(str).str.zfill(4)
    
    # Special and Consolation columns
    special_cols = [f'SpecialNo{i}' for i in range(1, 11)]
    consolation_cols = [f'ConsolationNo{i}' for i in range(1, 11)]
    
    for col in special_cols + consolation_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.zfill(4)
    
    return df


def load_mdt_excel(filepath: str = "MDTdec28.xlsx") -> pd.DataFrame:
    """Load the MDT Excel file with additional draw data."""
    df = pd.read_excel(filepath, header=None)
    
    # Based on data exploration, columns are:
    # 0: Year, 1: Month, 2: Day, 3-25: Prize numbers, 26: Day of week
    column_names = ['Year', 'Month', 'Day']
    column_names += ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']
    column_names += [f'SpecialNo{i}' for i in range(1, 11)]
    column_names += [f'ConsolationNo{i}' for i in range(1, 11)]
    column_names += ['DayOfWeek']
    
    # Ensure we have enough column names
    while len(column_names) < len(df.columns):
        column_names.append(f'Unknown{len(column_names)}')
    
    df.columns = column_names[:len(df.columns)]
    
    # Create date column
    df['DrawDate'] = pd.to_datetime(
        df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d',
        errors='coerce'
    )
    
    # Ensure prize columns are strings with leading zeros
    prize_cols = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']
    for col in prize_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.zfill(4)
    
    return df


def load_data_cache(operator: str, cache_dir: str = "c:/codex/lottery/data_cache") -> pd.DataFrame | None:
    """Read live-sync cache CSV for an operator. Returns None if absent."""
    p = Path(cache_dir) / f"{operator}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, dtype={'DrawDate': str})
    df['DrawDate'] = pd.to_datetime(df['DrawDate'], format='%Y%m%d', errors='coerce')
    for col in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.zfill(4)
    return df


def load_combined_data(data_dir: str = ".", include_cache: bool = True) -> pd.DataFrame:
    """Load and combine all available data sources (file + live-sync cache)."""
    data_path = Path(data_dir)

    dfs = []

    txt_path = data_path / "4D.txt"
    if txt_path.exists():
        df_txt = load_4d_txt(str(txt_path))
        df_txt['Source'] = '4D.txt'
        dfs.append(df_txt)
        print(f"Loaded {len(df_txt)} draws from 4D.txt")

    excel_path = data_path / "MDTdec28.xlsx"
    if excel_path.exists():
        df_excel = load_mdt_excel(str(excel_path))
        df_excel['Source'] = 'MDTdec28.xlsx'
        dfs.append(df_excel)
        print(f"Loaded {len(df_excel)} draws from MDTdec28.xlsx")

    if include_cache:
        cache_dir = data_path / "data_cache"
        if cache_dir.exists():
            for csv_file in cache_dir.glob("*.csv"):
                operator = csv_file.stem
                df_cache = load_data_cache(operator, cache_dir=str(cache_dir))
                if df_cache is not None and not df_cache.empty:
                    if 'Source' not in df_cache.columns:
                        df_cache['Source'] = operator
                    dfs.append(df_cache)
                    print(f"Loaded {len(df_cache)} draws from cache ({operator})")

    if not dfs:
        raise FileNotFoundError("No data files found!")

    combined = pd.concat(dfs, ignore_index=True)
    if 'Source' in combined.columns:
        combined = combined.sort_values(['DrawDate', 'Source']).drop_duplicates(
            subset=['DrawDate', 'Source'], keep='first'
        )
    else:
        combined = combined.sort_values('DrawDate').drop_duplicates(subset=['DrawDate'], keep='first')
    combined = combined.sort_values('DrawDate').reset_index(drop=True)

    print(f"Combined dataset: {len(combined)} unique draws")
    print(f"Date range: {combined['DrawDate'].min()} to {combined['DrawDate'].max()}")

    return combined


def extract_top3_prizes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract only the top 3 prize columns for focused analysis."""
    cols = ['DrawDate', '1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']
    if 'Source' in df.columns:
        cols.append('Source')
    
    result = df[cols].copy()
    result = result.dropna(subset=['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo'])
    
    return result


def create_number_history(df: pd.DataFrame) -> dict:
    """Create a dictionary tracking when each number appeared."""
    history = {}
    
    for idx, row in df.iterrows():
        date = row['DrawDate']
        for prize in ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']:
            num = str(row[prize]).zfill(4)
            if num not in history:
                history[num] = {'dates': [], 'prizes': []}
            history[num]['dates'].append(date)
            history[num]['prizes'].append(prize)
    
    return history


if __name__ == "__main__":
    # Test loading
    df = load_combined_data("c:/codex/lottery")
    top3 = extract_top3_prizes(df)
    print(f"\nTop 3 prizes dataset: {len(top3)} draws")
    print(top3.head(10))
