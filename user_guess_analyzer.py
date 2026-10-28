"""
User-input 4-digit number analyzer.

Tells the user everything the historical record has to say about their pick:
how often it's appeared, where, when, in which permutation, and how it ranks
under every available method. Closes with an honest verdict line that puts
"historical resonance" alongside the random baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from feature_engineer import get_digit_features
from data_loader import normalize_prize_number


ALL_SLOT_COLS = (
    ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']
    + [f'SpecialNo{i}' for i in range(1, 11)]
    + [f'ConsolationNo{i}' for i in range(1, 11)]
)
TOP3_COLS = ['1stPrizeNo', '2ndPrizeNo', '3rdPrizeNo']

# Compatibility constant for the full 23-slot independent-uniform approximation.
RANDOM_BASELINE_PER_DRAW = 1 - (1 - 1 / 10000) ** 23


@dataclass
class GuessReport:
    number: str
    digit_features: dict
    appearances_by_column: dict
    total_appearances: int
    last_seen: pd.Timestamp | None
    days_since_last: int | None
    permutation_family: str
    permutation_appearances: int
    empirical_next_draw_hit_prob: float
    random_baseline: float
    method_ranks: dict[str, int | None]
    agreement_count: int
    weighted_score: float
    verdict: str
    evaluated_slots: int = 3


def _normalize(number: str) -> str:
    normalized = normalize_prize_number(number)
    if normalized is None:
        raise ValueError(f"Number must be 4 digits, got {number!r}")
    return normalized


def _count_in_columns(df: pd.DataFrame, number: str, columns: list[str]) -> dict[str, int]:
    out = {}
    for col in columns:
        if col not in df.columns:
            continue
        out[col] = int((df[col].map(normalize_prize_number) == number).sum())
    return out


def _last_seen(df: pd.DataFrame, number: str, columns: list[str]) -> pd.Timestamp | None:
    dates = []
    for col in columns:
        if col not in df.columns:
            continue
        mask = df[col].map(normalize_prize_number) == number
        if mask.any():
            dates.append(df.loc[mask, 'DrawDate'].max())
    return max(dates) if dates else None


def _empirical_next_hit_prob(df: pd.DataFrame, number: str, columns: list[str]) -> float:
    """For every historical draw where the number appeared, check the NEXT draw.

    Returns hits / eligible transitions. Under independent uniform slots,
    the matching baseline depends on the columns actually evaluated.
    """
    sorted_df = df.sort_values('DrawDate').reset_index(drop=True)
    if len(sorted_df) < 2:
        return 0.0

    appeared_mask = pd.Series(False, index=sorted_df.index)
    for col in columns:
        if col in sorted_df.columns:
            appeared_mask |= sorted_df[col].map(normalize_prize_number) == number

    occurrence_idx = np.flatnonzero(appeared_mask.values)
    if len(occurrence_idx) == 0:
        return 0.0
    next_idx = occurrence_idx[occurrence_idx < len(sorted_df) - 1] + 1
    if len(next_idx) == 0:
        return 0.0

    next_appeared = pd.Series(False, index=sorted_df.index)
    for col in columns:
        if col in sorted_df.columns:
            next_appeared |= sorted_df[col].map(normalize_prize_number) == number
    hits = int(next_appeared.iloc[next_idx].sum())
    return hits / len(next_idx)


def _permutation_count(df: pd.DataFrame, number: str, columns: list[str]) -> tuple[str, int]:
    canonical = ''.join(sorted(number))
    count = 0
    for col in columns:
        if col not in df.columns:
            continue
        col_canon = df[col].map(normalize_prize_number).dropna().apply(lambda n: ''.join(sorted(n)))
        count += int((col_canon == canonical).sum())
    return canonical, count


def _ranks_in_methods(method_results: dict, number: str) -> dict[str, int | None]:
    ranks = {}
    for name, res in method_results.items():
        nums = [n for n, _ in res.top]
        try:
            ranks[name] = nums.index(number) + 1
        except ValueError:
            ranks[name] = None
    return ranks


def _agreement_from_consensus(consensus_df: pd.DataFrame, number: str) -> tuple[int, float]:
    row = consensus_df[consensus_df['number'] == number]
    if row.empty:
        return 0, 0.0
    return int(row.iloc[0]['agreement_count']), float(row.iloc[0]['weighted_score'])


def _verdict(empirical: float, baseline: float, agreement: int) -> str:
    delta = empirical - baseline
    if abs(delta) < baseline * 0.5:
        edge = "sample rate near the uniform baseline; no predictive edge established"
    elif delta > 0:
        edge = f"empirical rate {delta:.4f} above baseline; uncertainty and predictive significance are not established"
    else:
        edge = f"empirical rate {-delta:.4f} below baseline; this descriptive difference is not proof of a predictive effect"
    return (
        f"Historical resonance: agreement={agreement}/methods. "
        f"Next-draw edge: {edge}. "
        "Do not bet on this number expecting a positive return."
    )


def analyze(
    number: str,
    df: pd.DataFrame,
    method_results: dict | None = None,
    consensus_df: pd.DataFrame | None = None,
    columns: list[str] = TOP3_COLS,
) -> GuessReport:
    """Run the full analysis pipeline for a user-supplied 4-digit number."""
    number = _normalize(number)
    columns = [column for column in dict.fromkeys(columns) if column in df.columns]
    baseline = 1 - (1 - 1 / 10000) ** len(columns)

    appearances_by_col = _count_in_columns(df, number, ALL_SLOT_COLS)
    total = sum(appearances_by_col.values())

    last = _last_seen(df, number, ALL_SLOT_COLS)
    current = df['DrawDate'].max()
    days_since = (current - last).days if last is not None else None

    perm_str, perm_count = _permutation_count(df, number, columns)
    empirical = _empirical_next_hit_prob(df, number, columns)
    digit_features = get_digit_features(number)

    if method_results is not None:
        ranks = _ranks_in_methods(method_results, number)
    else:
        ranks = {}

    if consensus_df is not None and not consensus_df.empty:
        agreement, weighted = _agreement_from_consensus(consensus_df, number)
    else:
        agreement, weighted = 0, 0.0

    return GuessReport(
        number=number,
        digit_features=digit_features,
        appearances_by_column=appearances_by_col,
        total_appearances=total,
        last_seen=last,
        days_since_last=days_since,
        permutation_family=perm_str,
        permutation_appearances=perm_count,
        empirical_next_draw_hit_prob=empirical,
        random_baseline=baseline,
        method_ranks=ranks,
        agreement_count=agreement,
        weighted_score=weighted,
        verdict=_verdict(empirical, baseline, agreement),
        evaluated_slots=len(columns),
    )


def format_report(report: GuessReport) -> str:
    lines = []
    lines.append(f"=== Report for {report.number} ===")
    lines.append("")
    lines.append(f"Total appearances across all 23 slots: {report.total_appearances}")
    nonzero = {k: v for k, v in report.appearances_by_column.items() if v > 0}
    if nonzero:
        lines.append(f"  Breakdown: {nonzero}")
    lines.append(f"Permutation family: {report.permutation_family}  (appearances: {report.permutation_appearances})")
    if report.last_seen is not None:
        lines.append(f"Last seen: {report.last_seen.date()} ({report.days_since_last} days ago)")
    else:
        lines.append("Last seen: NEVER (in the available prize columns)")
    lines.append("")
    lines.append("Digit features:")
    for k, v in report.digit_features.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"Empirical next-draw hit prob: {report.empirical_next_draw_hit_prob:.4f}")
    lines.append(f"Independent-uniform baseline ({report.evaluated_slots} analyzed slots): {report.random_baseline:.4f}")
    lines.append("")
    if report.method_ranks:
        lines.append("Method ranks (None = not in top-N):")
        for m, r in report.method_ranks.items():
            lines.append(f"  {m}: {r}")
    lines.append("")
    lines.append(f"Consensus agreement: {report.agreement_count} methods, weighted={report.weighted_score:.4f}")
    lines.append("")
    lines.append(f"VERDICT: {report.verdict}")
    return '\n'.join(lines)


if __name__ == "__main__":
    import sys
    from data_loader import load_combined_data, extract_top3_prizes
    from predictor_suite import run_all
    from consensus import build_consensus

    guess = sys.argv[1] if len(sys.argv) > 1 else "1234"

    df = load_combined_data()
    top3 = extract_top3_prizes(df)
    method_results = run_all(top3, include_ml=False, top_n=100)
    cons = build_consensus(method_results, top_n=100)

    report = analyze(guess, df, method_results=method_results, consensus_df=cons)
    print(format_report(report))
