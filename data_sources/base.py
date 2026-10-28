"""Base abstractions for 4D data sources."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime

from data_loader import normalize_prize_number


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
        date = str(self.draw_date)
        if len(date) != 8 or not date.isascii() or not date.isdigit():
            raise ValueError(f'Invalid draw date: {date!r}')
        datetime.strptime(date, '%Y%m%d')

        def cell(value, required=False):
            number = normalize_prize_number(value)
            if number is not None:
                return number
            if required or str(value).strip() not in {'', 'None', 'nan', '<NA>'}:
                raise ValueError(f'Invalid prize number: {value!r}')
            return ''

        row = {
            'DrawDate': date,
            '1stPrizeNo': cell(self.first, required=True),
            '2ndPrizeNo': cell(self.second, required=True),
            '3rdPrizeNo': cell(self.third, required=True),
            'Source': self.source,
        }
        for i in range(10):
            row[f'SpecialNo{i + 1}'] = cell(self.special[i]) if i < len(self.special) else ''
            row[f'ConsolationNo{i + 1}'] = cell(self.consolation[i]) if i < len(self.consolation) else ''
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
