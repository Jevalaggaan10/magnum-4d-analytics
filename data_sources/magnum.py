"""Magnum 4D source — thin wrapper over the bundled `4D.txt` historical file.

Magnum doesn't expose a public JSON endpoint we can hit reliably, so this
source serves the static dataset already in the repo (1992-05-06 onward).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_loader import REPO_ROOT, load_4d_txt, normalize_prize_number

from .base import LotterySource, NormalizedDraw


DEFAULT_PATH = REPO_ROOT / '4D.txt'


class Magnum(LotterySource):
    name = "magnum"

    def __init__(self, csv_path: str | Path = DEFAULT_PATH):
        self.csv_path = Path(csv_path)
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            df = load_4d_txt(self.csv_path)
            df['DrawDate'] = df['DrawDate'].dt.strftime('%Y%m%d')
            self._df = df
        return self._df

    def fetch_dates(self) -> list[str]:
        df = self._load()
        return sorted(df['DrawDate'].astype(str).unique().tolist(), reverse=True)

    def fetch_draw(self, draw_date: str) -> NormalizedDraw | None:
        df = self._load()
        row = df[df['DrawDate'].astype(str) == str(draw_date)]
        if row.empty:
            return None
        r = row.iloc[0]

        def cell(col):
            return normalize_prize_number(r[col]) or '' if col in r else ''

        special = [cell(f'SpecialNo{i}') for i in range(1, 11)]
        consolation = [cell(f'ConsolationNo{i}') for i in range(1, 11)]

        return NormalizedDraw(
            draw_date=str(draw_date),
            first=cell('1stPrizeNo'),
            second=cell('2ndPrizeNo'),
            third=cell('3rdPrizeNo'),
            special=special,
            consolation=consolation,
            source=self.name,
        )

    def available(self) -> bool:
        return self.csv_path.exists()
