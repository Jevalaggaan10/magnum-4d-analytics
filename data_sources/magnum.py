"""Magnum 4D source — thin wrapper over the bundled `4D.txt` historical file.

Magnum doesn't expose a public JSON endpoint we can hit reliably, so this
source serves the static dataset already in the repo (1985-04-25 onward).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import LotterySource, NormalizedDraw


DEFAULT_PATH = Path("c:/codex/lottery/4D.txt")


class Magnum(LotterySource):
    name = "magnum"

    def __init__(self, csv_path: str | Path = DEFAULT_PATH):
        self.csv_path = Path(csv_path)
        self._df: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self.csv_path)
            df['DrawDate'] = df['DrawDate'].astype(str)
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
            return str(r[col]).zfill(4) if col in r and pd.notna(r[col]) else ''

        special = [cell(f'SpecialNo{i}') for i in range(1, 11)]
        consolation = [cell(f'ConsolationNo{i}') for i in range(1, 11)]

        return NormalizedDraw(
            draw_date=str(draw_date),
            first=cell('1stPrizeNo'),
            second=cell('2ndPrizeNo'),
            third=cell('3rdPrizeNo'),
            special=[x for x in special if x],
            consolation=[x for x in consolation if x],
            source=self.name,
        )

    def available(self) -> bool:
        return self.csv_path.exists()
