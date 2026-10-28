"""Load operator-specific 4D histories without mixing unrelated draw series."""

from pathlib import Path
import re

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
TOP3_COLS = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']
SLOT_COLS = (TOP3_COLS + [f'SpecialNo{i}' for i in range(1, 11)]
             + [f'ConsolationNo{i}' for i in range(1, 11)])
OPERATORS = ('magnum', 'damacai', 'unverified_excel', 'fouryes3')


def normalize_prize_number(value) -> str | None:
    """Return four ASCII digits, or None for missing/invalid prize cells.

    Integral decimal strings are accepted to repair older numeric CSV/Excel
    round trips. Missing values must never turn into a fabricated ``0000``.
    """
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not re.fullmatch(r'[0-9]{1,4}(?:\.0+)?', text):
        return None
    return text.split('.')[0].zfill(4)


def _normalize_prizes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in SLOT_COLS:
        if col in df.columns:
            df[col] = df[col].map(normalize_prize_number)
    return df


def _parse_dates(values: pd.Series) -> pd.Series:
    """Accept canonical YYYYMMDD and previously saved ISO dates."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return pd.to_datetime(values, errors='coerce').dt.normalize()
    text = values.astype('string').str.strip()
    parsed = pd.to_datetime(text, format='%Y%m%d', errors='coerce')
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            text.loc[missing], format='ISO8601', errors='coerce')
    return parsed.dt.normalize()


def load_4d_txt(filepath: str | Path | None = None) -> pd.DataFrame:
    """Load the bundled Magnum history, retaining all 23 prize slots."""
    df = pd.read_csv(filepath or REPO_ROOT / '4D.txt', dtype='string',
                     keep_default_na=False)
    df['DrawDate'] = _parse_dates(df['DrawDate'])
    return _normalize_prizes(df)


def load_mdt_excel(filepath: str | Path | None = None) -> pd.DataFrame:
    """Read the separate, provenance-unverified Excel draw series explicitly."""
    df = pd.read_excel(filepath or REPO_ROOT / 'MDTdec28.xlsx', header=None)
    names = ['Year', 'Month', 'Day'] + SLOT_COLS + ['DayOfWeek']
    names.extend(f'Unknown{i}' for i in range(len(names), len(df.columns)))
    df.columns = names[:len(df.columns)]
    date_parts = df[['Year', 'Month', 'Day']].apply(pd.to_numeric, errors='coerce')
    df['DrawDate'] = pd.to_datetime(date_parts, errors='coerce')
    return _normalize_prizes(df)


def load_data_cache(operator: str, cache_dir: str | Path | None = None) -> pd.DataFrame | None:
    """Read an operator CSV, preserving leading zeros and empty prize cells."""
    if not re.fullmatch(r'[a-z0-9_]+', operator):
        raise ValueError(f'Invalid operator name: {operator!r}')
    path = Path(cache_dir) if cache_dir is not None else REPO_ROOT / 'data_cache'
    path = path / f'{operator}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype='string', keep_default_na=False)
    df['DrawDate'] = _parse_dates(df['DrawDate'])
    return _normalize_prizes(df)


def load_combined_data(data_dir: str | Path | None = None,
                       include_cache: bool = True,
                       operator: str = 'magnum') -> pd.DataFrame:
    """Combine files for ONE series, keyed by date and operator.

    Magnum is the default. The conflicting Excel history requires the explicit
    ``unverified_excel`` selection; the configurable fouryes3 feed also stays
    separate until its operator provenance is verified. Tracked ``data/`` files
    precede optional local caches, whose latest records take precedence.
    """
    if operator not in OPERATORS:
        raise ValueError(f'Unknown operator {operator!r}; choose from {OPERATORS}')
    data_path = Path(data_dir) if data_dir is not None else REPO_ROOT
    frames = []

    def append(frame, source):
        if frame is None or frame.empty:
            return
        frame = frame.copy()
        if 'Source' not in frame.columns:
            frame['Source'] = source
        else:
            frame['Source'] = frame['Source'].fillna(source).replace('', source)
        frame['Operator'] = operator
        frames.append(frame)

    if operator == 'magnum' and (data_path / '4D.txt').exists():
        append(load_4d_txt(data_path / '4D.txt'), '4D.txt')
    if operator == 'unverified_excel' and (data_path / 'MDTdec28.xlsx').exists():
        append(load_mdt_excel(data_path / 'MDTdec28.xlsx'), 'MDTdec28.xlsx')
    if operator != 'unverified_excel':
        append(load_data_cache(operator, data_path / 'data'), operator)
        if include_cache:
            append(load_data_cache(operator, data_path / 'data_cache'), operator)

    if not frames:
        raise FileNotFoundError(f'No data files found for {operator!r} in {data_path}')

    combined = pd.concat(frames, ignore_index=True).dropna(subset=['DrawDate'])
    combined = combined.drop_duplicates(['DrawDate', 'Operator'], keep='last')
    combined = combined.sort_values('DrawDate').reset_index(drop=True)
    print(f'{operator}: {len(combined)} unique draws')
    print(f"Date range: {combined['DrawDate'].min()} to {combined['DrawDate'].max()}")
    return combined


def extract_top3_prizes(df: pd.DataFrame) -> pd.DataFrame:
    """Return valid top-three results plus their provenance columns."""
    cols = ['DrawDate'] + TOP3_COLS + [c for c in ['Source', 'Operator'] if c in df]
    result = _normalize_prizes(df[cols])
    result['DrawDate'] = _parse_dates(result['DrawDate'])
    return result.dropna(subset=['DrawDate'] + TOP3_COLS).copy()


def create_number_history(df: pd.DataFrame) -> dict:
    """Track valid top-three number occurrences without inventing missing values."""
    history = {}
    for _, row in extract_top3_prizes(df).iterrows():
        for prize in TOP3_COLS:
            number = row[prize]
            history.setdefault(number, {'dates': [], 'prizes': []})
            history[number]['dates'].append(row['DrawDate'])
            history[number]['prizes'].append(prize)
    return history


if __name__ == '__main__':
    draws = load_combined_data()
    top3 = extract_top3_prizes(draws)
    print(f'\nTop 3 prizes dataset: {len(top3)} draws')
    print(top3.head(10))
