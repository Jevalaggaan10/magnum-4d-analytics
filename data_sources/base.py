"""Base abstractions for 4D data sources."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterable


SCHEMA_COLUMNS = (
    ['DrawDate', '1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']
    + [f'SpecialNo{i}' for i in range(1, 11)]
    + [f'ConsolationNo{i}' for i in range(1, 11)]
    + ['Source']
)


@dataclass
class NormalizedDraw:
    """One Malaysian 4D draw in canonical schema."""
    draw_date: str  # YYYYMMDD
    first: str
    second: str
    third: str
    special: list[str] = field(default_factory=list)
    consolation: list[str] = field(default_factory=list)
    source: str = ""

    def to_row(self) -> dict:
        row = {
            'DrawDate': self.draw_date,
            '1stPrizeNo': self.first.zfill(4),
            '2ndPrizeNo': self.second.zfill(4),
            '3rdPrizeNo': self.third.zfill(4),
            'Source': self.source,
        }
        for i in range(10):
            row[f'SpecialNo{i + 1}'] = self.special[i].zfill(4) if i < len(self.special) else ''
            row[f'ConsolationNo{i + 1}'] = self.consolation[i].zfill(4) if i < len(self.consolation) else ''
        return row


class LotterySource(abc.ABC):
    """Abstract Malaysian 4D operator adapter."""

    name: str = "abstract"

    @abc.abstractmethod
    def fetch_dates(self) -> list[str]:
        """Return recent draw dates as `YYYYMMDD` strings, newest first."""

    @abc.abstractmethod
    def fetch_draw(self, draw_date: str) -> NormalizedDraw | None:
        """Return the normalized draw for `draw_date`, or None if unavailable."""

    def available(self) -> bool:
        """Best-effort liveness check. Override if cheap probe is possible."""
        try:
            self.fetch_dates()
            return True
        except Exception:
            return False
