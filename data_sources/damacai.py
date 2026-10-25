"""Da Ma Cai (PMP 4D) adapter using Hong's public-endpoint technique.

Reference: https://hongineer.com/scrapping-da-ma-cai-4d-results
"""

from __future__ import annotations

import time

import requests

from .base import LotterySource, NormalizedDraw


BASE = "https://www.damacai.com.my"
LIST_PATH = "/ListPastResult"
CALL_PATH = "/callpassresult"
COOKIE_SESSION = "363"
USER_AGENT = "Mozilla/5.0 (compatible; lottery-analytics/1.0; +https://example.invalid)"


class Damacai(LotterySource):
    name = "damacai"

    def __init__(self, polite_sleep_seconds: float = 1.0, timeout: float = 15.0):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/plain,*/*",
        })
        self.polite_sleep = polite_sleep_seconds
        self.timeout = timeout

    def fetch_dates(self) -> list[str]:
        resp = self.session.get(f"{BASE}{LIST_PATH}", timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("drawdate", "") or ""
        if not isinstance(raw, str):
            return []
        dates = [d.strip() for d in raw.split() if d.strip()]
        return sorted(set(dates), reverse=True)

    def _signed_link(self, draw_date: str) -> str | None:
        time.sleep(self.polite_sleep)
        r = self.session.get(
            f"{BASE}{CALL_PATH}",
            params={"pastdate": draw_date},
            headers={"cookiesession": COOKIE_SESSION},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return (r.json() or {}).get("link")

    @staticmethod
    def _flexget(obj: dict, *keys: str):
        """Pull from `obj` trying each key (handles 'pass'/'past', 'consolation'/'consolidate')."""
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        for k in keys:
            for actual in obj.keys():
                if actual.lower() == k.lower():
                    return obj[actual]
        return None

    def fetch_draw(self, draw_date: str) -> NormalizedDraw | None:
        link = self._signed_link(draw_date)
        if not link:
            return None

        time.sleep(self.polite_sleep)
        r = self.session.get(link, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            return None

        first = str(self._flexget(data, "p1", "first", "prize1", "FirstPrize") or "")
        second = str(self._flexget(data, "p2", "second", "prize2", "SecondPrize") or "")
        third = str(self._flexget(data, "p3", "third", "prize3", "ThirdPrize") or "")

        special_raw = (
            self._flexget(
                data,
                "starterList", "special", "Special", "specials", "specialPrize",
            )
            or []
        )
        consolation_raw = (
            self._flexget(
                data,
                "consolidateList", "consolation", "Consolation",
                "consolidation", "Consolidation",
                "consolations", "consolationPrize",
            )
            or []
        )

        if isinstance(special_raw, dict):
            special_list = [str(v) for _, v in sorted(special_raw.items())]
        else:
            special_list = [str(v) for v in special_raw]

        if isinstance(consolation_raw, dict):
            cons_list = [str(v) for _, v in sorted(consolation_raw.items())]
        else:
            cons_list = [str(v) for v in consolation_raw]

        if not first and not second and not third:
            return None

        return NormalizedDraw(
            draw_date=draw_date,
            first=first,
            second=second,
            third=third,
            special=special_list[:10],
            consolation=cons_list[:10],
            source=self.name,
        )

    def available(self) -> bool:
        try:
            dates = self.fetch_dates()
            return len(dates) > 0
        except Exception:
            return False


if __name__ == "__main__":
    d = Damacai()
    dates = d.fetch_dates()
    print(f"Got {len(dates)} dates, first 5: {dates[:5]}")
    if dates:
        draw = d.fetch_draw(dates[0])
        print(f"Draw {dates[0]}: {draw}")
