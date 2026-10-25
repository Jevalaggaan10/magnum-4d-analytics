"""Live next-draw prediction tracker for GitHub Actions.

Runs on a schedule (MWSS draw evenings). Each run:
1. Syncs the latest published Da Ma Cai result (polite, few requests).
2. If a pending prediction exists for a published draw, scores it against
   the real 23 prize numbers and archives it.
3. If no pending prediction exists, generates one for the next draw date
   (next Wednesday / Saturday / Sunday after the newest known draw) using
   the predictor suite + cross-method consensus.
4. Renders/refreshes `PREDICTION_LOG.md` from the tracked state file
   `predictions.json` so the log stays reproducible.

Design notes
------------
- Single-file and path-safe: everything derives from this script's
  location, so it behaves the same on a laptop and in a fresh CI clone.
- Tracked state lives in `predictions.json` (committed), not in the
  gitignored sync cache, because Actions start from a clean checkout.
- Prediction corpus = bundled Magnum `4D.txt` history + recent Da Ma Cai
  syncs. The bundled Excel is a separate draw series that conflicts with
  `4D.txt` on shared dates, so it is excluded here pending provenance fix.
- Random-control picks use `seed = int(YYYYMMDD)` so the luck-only
  comparison is reproducible by anyone.

Usage:
    python scripts/track_predictions.py             # sync + verify + log
    python scripts/track_predictions.py --no-sync   # offline: local cache only
    python scripts/track_predictions.py --dry-run   # print log, write nothing
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import load_4d_txt, load_data_cache, extract_top3_prizes  # noqa: E402
from predictor_suite import run_all  # noqa: E402
from consensus import build_consensus  # noqa: E402

STATE_FILE = REPO_ROOT / "predictions.json"
LOG_FILE = REPO_ROOT / "PREDICTION_LOG.md"

TOP3_COLS = ["1stPrizeNo", "2ndPrizeNo", "3rdPrizeNo"]
SLOT_COLS = (
    TOP3_COLS
    + [f"SpecialNo{i}" for i in range(1, 11)]
    + [f"ConsolationNo{i}" for i in range(1, 11)]
)
SLOT_LABELS = {
    "1stPrizeNo": "1st prize",
    "2ndPrizeNo": "2nd prize",
    "3rdPrizeNo": "3rd prize",
}
# MWSS draws: Wednesday(2) / Saturday(5) / Sunday(6), Python weekday().
DRAW_WEEKDAYS = {2, 5, 6}
NUM_PICKS = 10
PENDING_TTL_DAYS = 60


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_valid_4d(v) -> bool:
    return isinstance(v, str) and re.fullmatch(r"\d{4}", v) is not None


def _slot_label(col: str) -> str:
    if col in SLOT_LABELS:
        return SLOT_LABELS[col]
    if col.startswith("SpecialNo"):
        return f"special #{int(col[-2:])}"
    return f"consolation #{int(col[-2:])}"


def load_corpus() -> pd.DataFrame:
    """Single-series corpus: bundled Magnum 4D.txt (+ local Da Ma Cai cache
    when present). The conflicting MDT Excel series is excluded on purpose."""
    frames = []
    txt = REPO_ROOT / "4D.txt"
    if txt.exists():
        df_txt = load_4d_txt(str(txt))
        df_txt["Source"] = "4D.txt"
        frames.append(df_txt)
    cache = load_data_cache("damacai", cache_dir=str(REPO_ROOT / "data_cache"))
    if cache is not None and not cache.empty:
        # repair the known cache round-trip corruption: prize cells may have
        # lost leading zeros when they were type-inferred as ints
        for col in SLOT_COLS:
            if col in cache.columns:
                s = cache[col].astype(str).str.strip()
                s = s.where(s.str.fullmatch(r"\d{1,4}"), other=None)
                cache[col] = s.str.zfill(4)
        frames.append(cache)
    if not frames:
        raise FileNotFoundError("No corpus found: need 4D.txt in repo root")
    return pd.concat(frames, ignore_index=True)


def _draw_to_slots(row: dict) -> dict:
    out = {}
    for col in SLOT_COLS:
        v = str(row.get(col, "")).strip()
        v = v if re.fullmatch(r"\d{1,4}", v) else ""
        out[col] = v.zfill(4)
    return out


def fetch_recent_draws(max_dates: int = 2) -> dict[str, dict]:
    """Fetch the latest published Da Ma Cai draws -> {YYYYMMDD: slots}."""
    from data_sources.damacai import Damacai

    src = Damacai()
    dates = src.fetch_dates()
    out: dict[str, dict] = {}
    for d in dates[:max_dates]:
        draw = src.fetch_draw(str(d))
        if draw is None:
            continue
        out[str(d).zfill(8)] = _draw_to_slots(draw.to_row())
    return out


def fetch_single_draw(draw_date: str) -> dict | None:
    """Fetch one specific draw date (used for pending resolution)."""
    from data_sources.damacai import Damacai

    src = Damacai()
    draw = src.fetch_draw(draw_date)
    if draw is None:
        return None
    return _draw_to_slots(draw.to_row())


def draw_from_local_cache(cache: pd.DataFrame | None, draw_date: str) -> dict | None:
    """Look up one draw date in the locally cached Da Ma Cai frame."""
    if cache is None or cache.empty:
        return None
    sub = cache[cache["DrawDate"].astype(str) == draw_date]
    if sub.empty:
        return None
    return _draw_to_slots(sub.iloc[0].to_dict())


def next_draw_date(after: str) -> str:
    """Next MWSS date strictly after `after` (YYYYMMDD)."""
    d = datetime.strptime(after, "%Y%m%d") + timedelta(days=1)
    for _ in range(14):
        if d.weekday() in DRAW_WEEKDAYS:
            return d.strftime("%Y%m%d")
        d += timedelta(days=1)
    raise RuntimeError("no draw date within 14 days")


def random_control(for_date: str, n: int = NUM_PICKS) -> list[str]:
    """Reproducible luck-only set of 10 numbers, seeded by the draw date."""
    rng = random.Random(int(for_date))
    return [f"{i:04d}" for i in rng.sample(range(10000), n)]


def score_hits(picks: list[str], actual: dict) -> list[dict]:
    """Which picks appear on which prize slots of the actual draw."""
    pick_set = set(picks)
    hits = []
    for col in SLOT_COLS:
        num = actual.get(col, "")
        if _is_valid_4d(num) and num in pick_set:
            hits.append({"number": num, "tier": _slot_label(col)})
    return hits


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"pending": None, "history": []}


# ---------------------------------------------------------------------------
# main pipeline
# ---------------------------------------------------------------------------

def main(argv: list | None = None) -> int:
    # Windows consoles default to cp1252 and would crash on emoji in stdout
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(prog="track_predictions")
    ap.add_argument("--no-sync", action="store_true",
                    help="offline: use local cache only, no network")
    ap.add_argument("--dry-run", action="store_true",
                    help="render the log to stdout, write nothing")
    args = ap.parse_args(argv)

    state = load_state()
    corpus = load_corpus()
    damacai_cache = load_data_cache("damacai",
                                    cache_dir=str(REPO_ROOT / "data_cache"))

    # ---- 1) sync: latest published draws ---------------------------------
    network_ok = True
    recent: dict[str, dict] = {}
    if not args.no_sync:
        try:
            recent = fetch_recent_draws(max_dates=2)
            print(f"synced: {sorted(recent.keys())}")
        except Exception as e:
            network_ok = False
            print(f"WARNING: sync failed ({type(e).__name__}: {e}); state unchanged")

    # ---- 2) resolve the pending prediction (if any) ----------------------
    pending = state.get("pending")
    if pending:
        target = str(pending["for_date"]).zfill(8)
        actual = recent.get(target)
        if actual is None:
            actual = draw_from_local_cache(damacai_cache, target)
        if actual is None and network_ok:
            today = datetime.now(timezone.utc).date()
            try:
                pub = datetime.strptime(target, "%Y%m%d").date()
            except ValueError:
                pub = None
            if pub and (today - pub).days > PENDING_TTL_DAYS:
                print(f"pending for {target} exceeded {PENDING_TTL_DAYS}d TTL; expiring unverified")
                state["history"].append({
                    "for_date": target,
                    "predicted": pending["picks"],
                    "random_control": pending["random_control"],
                    "actual": None,
                    "hits": [],
                    "random_hits": [],
                    "status": "expired-unverified",
                    "checked_at": datetime.now(timezone.utc).isoformat(),
                    "source": "damacai",
                })
                state["history"].sort(key=lambda h: h["for_date"])
                state["pending"] = None
            else:
                try:
                    actual = fetch_single_draw(target)
                except Exception as e:
                    print(f"targeted fetch for {target} failed: {e}")
        if actual and any(_is_valid_4d(actual.get(c, "")) for c in TOP3_COLS):
            state["history"].append({
                "for_date": target,
                "predicted": pending["picks"],
                "random_control": pending["random_control"],
                "actual": actual,
                "hits": score_hits(pending["picks"], actual),
                "random_hits": score_hits(pending["random_control"], actual),
                "status": "verified",
                "checked_at": datetime.now(timezone.utc).isoformat(),
                "source": "damacai",
            })
            state["history"].sort(key=lambda h: h["for_date"])
            state["pending"] = None
            print(f"resolved pending prediction for {target}: "
                  f"{len(state['history'][-1]['hits'])} hit(s)")

    # ---- 3) generate a new prediction if none pending --------------------
    if state.get("pending") is None:
        newest = corpus["DrawDate"].max() if corpus["DrawDate"].max() is not pd.NaT else pd.NaT
        if recent:
            recent_ts = pd.to_datetime(max(recent.keys()), format="%Y%m%d",
                                       errors="coerce")
            newest = max(newest, recent_ts) if pd.notna(newest) else recent_ts
        if newest is None or pd.isna(newest):
            return 1
        target = next_draw_date(newest.strftime("%Y%m%d"))

        # identical to the app's own prediction path (includes DrawDate+Source)
        top3 = extract_top3_prizes(corpus)
        results = run_all(top3, include_ml=False, top_n=50)
        cons = build_consensus(results, top_n=50)
        if len(cons):
            cons = cons[cons["number"].astype(str).str.fullmatch(r"\d{4}", na=False)]
        picks: list[str] = cons.head(NUM_PICKS)["number"].tolist() if len(cons) else []
        if len(picks) < NUM_PICKS:  # top up with all-time frequency
            freq = Counter()
            for col in TOP3_COLS:
                freq.update(n for n in corpus[col].astype(str).str.zfill(4)
                            if re.fullmatch(r"\d{4}", n))
            for num, _ in freq.most_common(200):
                if num not in picks:
                    picks.append(num)
                if len(picks) >= NUM_PICKS:
                    break
        methods_ok = [k for k, r in results.items() if r.top]
        state["pending"] = {
            "for_date": target,
            "picks": picks,
            "random_control": random_control(target),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "corpus_cutoff": newest.strftime("%Y-%m-%d"),
            "methods_count": len(methods_ok),
            "methods_names": methods_ok,
        }
        print(f"generated prediction for {target} ({len(methods_ok)} methods)")

    # ---- 4) render --------------------------------------------------------
    log = render_log(state, network_ok=network_ok)
    if args.dry_run:
        print("----- rendered log -----")
        print(log)
        return 0

    existing_log = LOG_FILE.read_text(encoding="utf-8") if LOG_FILE.exists() else None
    if existing_log != log:
        LOG_FILE.write_text(log, encoding="utf-8")
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        print("updated PREDICTION_LOG.md + predictions.json")
    else:
        print("no changes")
    return 0


# ---------------------------------------------------------------------------
# markdown rendering
# ---------------------------------------------------------------------------

def render_log(state: dict, network_ok: bool = True) -> str:
    hist = [h for h in state.get("history", []) if h.get("actual")]
    expired = [h for h in state.get("history", []) if not h.get("actual")]
    resolved = len(hist)
    model_slot_hits = sum(len(h["hits"]) for h in hist)
    random_slot_hits = sum(len(h["random_hits"]) for h in hist)
    model_draw_hits = sum(1 for h in hist if h["hits"])
    random_draw_hits = sum(1 for h in hist if h["random_hits"])

    lines: list[str] = []
    a = lines.append

    a("# \U0001F3AF Live Prediction Tracker")
    a("")
    a("> Updated automatically by a scheduled GitHub Action on draw evenings"
      " (Wed / Sat / Sun), after results publish. Predictions are generated"
      " and logged **before** the next draw, then verified against the real"
      " result from Da Ma Cai's public endpoint.")
    a("")
    a("## How to read this")
    a("")
    a("Every draw gets two 10-number sets, published in advance:")
    a("")
    a("- **Model picks** — consensus of the suite's statistical methods.")
    a("- **Random control** — 10 purely random numbers, seeded by the draw date"
      " (`random.Random(YYYYMMDD).sample(range(10000), 10)`), so anyone can"
      " reproduce the luck-only baseline.")
    a("")
    a("A pick *hits* if it appears among the draw's 23 prize slots"
      " (1st/2nd/3rd prize + 10 special + 10 consolation). Per-draw hit"
      " probability for *any* 10-number set:")
    a("")
    a("```text")
    a("P(hit) = 1 - C(9990, 10) / C(10000, 10) ≈ 2.27%  →  ~1 hit per 44 draws")
    a("```")
    a("")
    a("Over time both columns should hover at that baseline. If the model"
      " column ever drifts meaningfully above the random column on this"
      " i.i.d. stream, that would be extraordinary — the log makes that self-checkable.")
    a("")
    if not network_ok:
        a("> ⚠️ The last sync failed; the values below are the last recorded state.")
        a("")

    pending = state.get("pending")
    if pending:
        a(f"## 🔮 Upcoming draw — {pending['for_date']} "
          f"_(generated {pending['generated_at'][:10]}, corpus cutoff "
          f"{pending['corpus_cutoff']}, {pending['methods_count']} methods: "
          f"{', '.join(pending['methods_names'])})_")
        a("")
        a("| # | Consensus pick |")
        a("|---|----------------|")
        for i, num in enumerate(pending["picks"], 1):
            a(f"| {i} | `{num}` |")
        a("")
        a(f"**Random control:** "
          f"{', '.join('`' + n + '`' for n in pending['random_control'])}")
        a("")

    a("## ✅ Verified draws")
    a("")
    if hist:
        a("| Draw | 10 picks | Actual 1st/2nd/3rd | Model hits | Random-control hits |")
        a("|------|----------|--------------------|------------|---------------------|")
        for h in hist:
            picks_md = ", ".join("`" + n + "`" for n in h["predicted"])
            top3_actual = "/".join(h["actual"].get(c, "–") for c in TOP3_COLS)
            hit_txt = (" ✅ " + "; ".join(
                f"{x['number']} ({x['tier']})" for x in h["hits"])) if h["hits"] else "—"
            rand_txt = (" ✅ " + ", ".join(x["number"] for x in h["random_hits"])) \
                if h["random_hits"] else "—"
            a(f"| {h['for_date']} | {picks_md} | `{top3_actual}` | {hit_txt} | {rand_txt} |")
    else:
        a("_No draws verified yet — the first scheduled prediction is pending._")
    a("")
    if expired:
        a("_Expired unverified (older than the sync window): "
          + ", ".join(h["for_date"] for h in expired) + "._")
        a("")
    a("## 📊 Lifetime tally")
    a("")
    a("| Set | Draws tracked | Draws with a hit | Slot hits | Hit frequency | Expected |")
    a("|-----|--------------:|-----------------:|----------:|--------------:|---------:|")
    m_lbl = f"{model_draw_hits / resolved * 100:.2f}% (draws) · {model_slot_hits} slots" if resolved else "—"
    r_lbl = f"{random_draw_hits / resolved * 100:.2f}% (draws) · {random_slot_hits} slots" if resolved else "—"
    if not resolved:
        m_lbl = r_lbl = "—"
    a(f"| Model picks | {resolved} | {model_draw_hits} | {model_slot_hits} | {m_lbl} | ~2.27% |")
    a(f"| Random control | {resolved} | {random_draw_hits} | {random_slot_hits} | {r_lbl} | ~2.27% |")
    a("")
    a("## Corpus & method notes")
    a("")
    a("- **Corpus** — bundled Magnum `4D.txt` history (up to "
      f"{(pending or {}).get('corpus_cutoff', 'n/a')}) + recent Da Ma Cai"
      " live syncs. The bundled `MDTdec28.xlsx` is a separate draw series"
      " that conflicts with `4D.txt` on shared dates, so it is excluded here"
      " until its provenance is sorted.")
    a("- **Prediction method** — `predictor_suite.run_all(include_ml=False)`"
      " (8 statistical lenses: frequency, 90-day hot, overdue-gap, digit"
      " probabilities, permutation families, Markov, ARIMA digit-sum,"
      " Bayesian Dirichlet) aggregated by cross-method consensus.")
    a("- **Verification** — Da Ma Cai public endpoint, polite 1 req/sec"
      " fetches, no scraping of protected pages.")
    a("")
    a("_Reminder: Malaysian 4D draws are i.i.d. uniform. Nothing here is a")
    a("betting strategy — this log is an open experiment whose expected outcome")
    a("is that both columns match the ~2.27% baseline._")
    a("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
