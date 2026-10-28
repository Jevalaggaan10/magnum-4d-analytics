"""Sync orchestrator for 4D data sources.

Reads existing per-operator CSV cache, fetches only missing recent dates,
appends + deduplicates + writes. Polite rate-limit baked into adapters.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_loader import REPO_ROOT, SLOT_COLS, load_data_cache, normalize_prize_number

from .base import LotterySource, SCHEMA_COLUMNS
from .damacai import Damacai
from .fouryes3 import FourYes3
from .magnum import Magnum


CACHE_DIR = REPO_ROOT / 'data_cache'
LAST_SYNC_FILE = CACHE_DIR / "last_sync.json"


SOURCES: dict[str, type[LotterySource]] = {
    "damacai": Damacai,
    "fouryes3": FourYes3,
    "magnum": Magnum,
}


@dataclass
class SyncReport:
    operator: str
    fetched: int
    skipped_existing: int
    failed: int
    available: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _cache_path(operator: str) -> Path:
    return CACHE_DIR / f"{operator}.csv"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache(operator: str) -> pd.DataFrame:
    p = _cache_path(operator)
    if p.exists():
        df = load_data_cache(operator, CACHE_DIR)
        df = df.dropna(subset=['DrawDate']).copy()
        df['DrawDate'] = df['DrawDate'].dt.strftime('%Y%m%d')
        return df
    return pd.DataFrame(columns=SCHEMA_COLUMNS)


def _write_cache(operator: str, df: pd.DataFrame) -> None:
    _ensure_cache_dir()
    df = df[[c for c in SCHEMA_COLUMNS if c in df.columns]].copy()
    for col in SLOT_COLS:
        if col in df:
            df[col] = df[col].map(normalize_prize_number)
    df.to_csv(_cache_path(operator), index=False)


def _update_last_sync(operator: str) -> None:
    _ensure_cache_dir()
    data: dict = {}
    if LAST_SYNC_FILE.exists():
        try:
            data = json.loads(LAST_SYNC_FILE.read_text())
        except Exception:
            data = {}
    data[operator] = datetime.utcnow().isoformat() + "Z"
    LAST_SYNC_FILE.write_text(json.dumps(data, indent=2))


def get_last_sync() -> dict:
    if not LAST_SYNC_FILE.exists():
        return {}
    try:
        return json.loads(LAST_SYNC_FILE.read_text())
    except Exception:
        return {}


def sync(operator: str, recent: int = 30, verbose: bool = True) -> SyncReport:
    if operator not in SOURCES:
        return SyncReport(operator, 0, 0, 0, False, f"unknown operator: {operator}")

    source = SOURCES[operator]()
    cache = _load_cache(operator)
    existing = set(cache['DrawDate'].astype(str).tolist()) if not cache.empty else set()

    try:
        all_dates = source.fetch_dates()
    except Exception as e:
        return SyncReport(operator, 0, 0, 0, False, f"{type(e).__name__}: {e}")

    if not all_dates:
        err = getattr(source, "last_error", None) or "no dates returned"
        return SyncReport(operator, 0, 0, 0, False, err)

    target_dates = all_dates[: max(0, recent)]
    missing = [d for d in target_dates if d not in existing]
    skipped = len(target_dates) - len(missing)

    rows = []
    failed = 0
    for d in missing:
        try:
            draw = source.fetch_draw(d)
            row = draw.to_row() if draw is not None else None
        except Exception as e:
            failed += 1
            if verbose:
                print(f"  ! {d} fetch error: {e}", file=sys.stderr)
            continue
        if row is None:
            failed += 1
            continue
        rows.append(row)
        if verbose:
            print(f"  + {d} fetched")

    if rows:
        new_df = pd.DataFrame(rows)
        merged = pd.concat([cache, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=['DrawDate'], keep='last')
        merged = merged.sort_values('DrawDate', ascending=False).reset_index(drop=True)
        _write_cache(operator, merged)

    _update_last_sync(operator)
    return SyncReport(operator, len(rows), skipped, failed, True)


def sync_all(recent: int = 30, verbose: bool = True) -> list[SyncReport]:
    out = []
    for op in SOURCES.keys():
        if verbose:
            print(f"=== syncing {op} ===")
        out.append(sync(op, recent=recent, verbose=verbose))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="data_sources.manager")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sync = sub.add_parser("sync")
    p_sync.add_argument("operator", choices=list(SOURCES.keys()) + ["all"])
    p_sync.add_argument("--recent", type=int, default=30)
    p_sync.add_argument("-q", "--quiet", action="store_true")

    p_status = sub.add_parser("status")

    args = parser.parse_args(argv)

    if args.cmd == "sync":
        reports = (
            sync_all(args.recent, verbose=not args.quiet)
            if args.operator == "all"
            else [sync(args.operator, args.recent, verbose=not args.quiet)]
        )
        for r in reports:
            print(json.dumps(r.to_dict(), indent=2))
        return 0 if all(r.available for r in reports) else 1

    if args.cmd == "status":
        print(json.dumps(get_last_sync(), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
