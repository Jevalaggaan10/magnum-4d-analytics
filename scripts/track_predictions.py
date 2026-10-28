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
- Prediction corpus = the tracked Da Ma Cai snapshot plus observed Da Ma
  Cai results in the ledger. Other operators and the unverified Excel
  series are excluded. New results are retained for subsequent training.
- Random-control picks use `seed = int(YYYYMMDD)` so the luck-only
  comparison is reproducible by anyone.

Usage:
    python scripts/track_predictions.py             # sync + verify + log
    python scripts/track_predictions.py --no-sync   # offline: local cache only
    python scripts/track_predictions.py --dry-run   # print log, write nothing
    python scripts/track_predictions.py --replay 40 # walk-forward replay of the
                                                    # last 40 published draws
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_loader import load_data_cache, extract_top3_prizes, normalize_prize_number  # noqa: E402
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
PIPELINE_VERSION = 'damacai-v2'


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _is_valid_4d(v) -> bool:
    return isinstance(v, str) and re.fullmatch(r"[0-9]{4}", v) is not None


def _slot_label(col: str) -> str:
    if col in SLOT_LABELS:
        return SLOT_LABELS[col]
    if col.startswith("SpecialNo"):
        return f"special #{int(col[len('SpecialNo'):])}"
    return f"consolation #{int(col[len('ConsolationNo'):])}"


def observed_draws(state: dict) -> dict[str, dict]:
    """Retain published outcomes without altering any earlier prediction."""
    draws = {}
    for record in state.get('replay', {}).get('rows', []) + state.get('history', []):
        actual = record.get('actual')
        if record.get('source') == 'damacai' and actual and complete_draw(actual):
            draws[record['for_date']] = _draw_to_slots(actual)
    for date, actual in state.get('observed_draws', {}).items():
        if complete_draw(actual):
            draws[date] = _draw_to_slots(actual)
    return draws


def load_corpus(state: dict | None = None) -> pd.DataFrame:
    """Same-operator, reproducible corpus; do not read a machine-local cache."""
    frames = []
    snapshot = load_data_cache('damacai', cache_dir=REPO_ROOT / 'data')
    if snapshot is not None and not snapshot.empty:
        frames.append(snapshot)
    rows = [dict(actual, DrawDate=pd.to_datetime(date, format='%Y%m%d'))
            for date, actual in observed_draws(state or {}).items()]
    if rows:
        frames.append(pd.DataFrame(rows))
    if not frames:
        raise FileNotFoundError('No Da Ma Cai history: need data/damacai.csv or observed ledger results')
    corpus = pd.concat(frames, ignore_index=True)
    corpus['Source'] = 'damacai'
    corpus['Operator'] = 'damacai'
    return corpus.dropna(subset=['DrawDate']).drop_duplicates('DrawDate', keep='last').sort_values('DrawDate').reset_index(drop=True)


def _draw_to_slots(row: dict) -> dict:
    out = {}
    for col in SLOT_COLS:
        out[col] = normalize_prize_number(row.get(col)) or ''
    return out


def complete_draw(actual: dict) -> bool:
    """Only published, complete results can resolve the 23-slot experiment."""
    return all(_is_valid_4d(actual.get(col)) for col in SLOT_COLS)


def generate_picks(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """One prediction path shared by live runs and chronological replays."""
    if train.empty:
        raise ValueError('cannot generate a trained prediction without prior draws')
    results = run_all(train, include_ml=False, top_n=50)
    cons = build_consensus(results, top_n=50)
    picks = cons.head(NUM_PICKS)['number'].tolist()
    methods = [key for key, result in results.items() if result.top]
    if not methods:
        raise ValueError('no prediction method produced a valid result')
    if len(picks) < NUM_PICKS:
        freq = Counter()
        for col in TOP3_COLS:
            freq.update(n for n in train[col].map(normalize_prize_number) if n is not None)
        for number, _ in freq.most_common():
            if number not in picks:
                picks.append(number)
            if len(picks) == NUM_PICKS:
                break
    if len(picks) != NUM_PICKS or len(set(picks)) != NUM_PICKS:
        raise ValueError('could not produce ten distinct valid picks')
    return picks, methods


def training_metadata(train: pd.DataFrame) -> dict:
    canonical = train.sort_values('DrawDate')[['DrawDate'] + TOP3_COLS].to_csv(
        index=False, lineterminator='\n', date_format='%Y-%m-%d')
    return {'pipeline_version': PIPELINE_VERSION, 'training_operator': 'damacai',
            'training_rows': len(train), 'corpus_cutoff': train['DrawDate'].max().strftime('%Y-%m-%d'),
            'corpus_sha256': hashlib.sha256(canonical.encode('utf-8')).hexdigest()}


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
        actual = _draw_to_slots(draw.to_row())
        if complete_draw(actual):
            out[str(d).zfill(8)] = actual
        else:
            raise ValueError(f'{d}: incomplete result; refusing to score missing prize slots')
    return out


def fetch_single_draw(draw_date: str) -> dict | None:
    """Fetch one specific draw date (used for pending resolution)."""
    from data_sources.damacai import Damacai

    src = Damacai()
    draw = src.fetch_draw(draw_date)
    if draw is None:
        return None
    actual = _draw_to_slots(draw.to_row())
    return actual if complete_draw(actual) else None


def draw_from_local_cache(cache: pd.DataFrame | None, draw_date: str) -> dict | None:
    """Look up one draw date in the locally cached Da Ma Cai frame."""
    if cache is None or cache.empty:
        return None
    dates = pd.to_datetime(cache['DrawDate']).dt.strftime('%Y%m%d')
    sub = cache[dates == draw_date]
    if sub.empty:
        return None
    actual = _draw_to_slots(sub.iloc[0].to_dict())
    return actual if complete_draw(actual) else None


# ---------------------------------------------------------------------------
# walk-forward replay (--replay N)
# ---------------------------------------------------------------------------

def collect_recent_draws(n: int) -> list[dict]:
    """Fetch the n most recent published Da Ma Cai draws, oldest -> newest."""
    from data_sources.damacai import Damacai

    src = Damacai()
    # fetch_dates() is newest-first; [:n] takes the most recent n
    dates = sorted(str(d).zfill(8) for d in src.fetch_dates()[:n])
    out: list[dict] = []
    for d in dates:
        draw = src.fetch_draw(d)
        if draw is None:
            raise ValueError(f'{d}: requested replay result unavailable')
        actual = _draw_to_slots(draw.to_row())
        if not complete_draw(actual):
            raise ValueError(f'{d}: requested replay result incomplete')
        out.append({"for_date": d, "slots": actual})
    return out


def walk_forward_replay(draws: list[dict], state: dict | None = None) -> tuple[list[dict], list[dict]]:
    """For each published draw D, train only on draws strictly before D —
    the Da Ma Cai corpus, including earlier observed replay draws, then
    score the consensus picks and the date-seeded
    random control against the real 23 prize slots of D.

    Returns (rows, skipped): skipped draws keep their error for the log
    instead of aborting the whole backtest."""
    base = load_corpus(state)
    observations = pd.DataFrame([dict(spec['slots'], DrawDate=pd.to_datetime(spec['for_date'], format='%Y%m%d'),
                                      Source='damacai', Operator='damacai') for spec in draws])
    base_top3 = extract_top3_prizes(pd.concat([base, observations], ignore_index=True)
                                  .drop_duplicates('DrawDate', keep='last').sort_values('DrawDate'))

    rows: list[dict] = []
    skipped: list[dict] = []
    for spec in sorted(draws, key=lambda item: item['for_date']):
        d = spec["for_date"]
        actual = spec["slots"]
        try:
            ts = pd.to_datetime(d, format="%Y%m%d")
            train = base_top3[base_top3["DrawDate"] < ts]
            if train.empty:
                print(f"  ! {d}: empty training set, skipped")
                skipped.append({"for_date": d, "error": "empty training set"})
                continue

            if not complete_draw(actual):
                raise ValueError('incomplete published draw')
            picks, methods = generate_picks(train)

            rnd = random_control(d)
            row = {
                "for_date": d,
                "predicted": picks,
                "random_control": rnd,
                "actual": {c: actual.get(c, "") for c in SLOT_COLS},
                "hits": score_hits(picks, actual),
                "random_hits": score_hits(rnd, actual),
                "methods_count": len(methods),
                "training_cutoff": train["DrawDate"].max().strftime("%Y-%m-%d"),
                "status": "replay",
                "source": "damacai",
                **training_metadata(train),
            }
            rows.append(row)
            print(f"  {d}: model {len(row['hits'])} hit(s), random "
                  f"{len(row['random_hits'])} hit(s) ({row['methods_count']} "
                  f"methods, trained until {row['training_cutoff']})")
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            skipped.append({"for_date": d, "error": msg})
            print(f"  ! {d}: replay failed ({msg}); skipped — traceback follows")
            traceback.print_exc()
    return rows, skipped


def run_replay_mode(n: int, state: dict, dry_run: bool = False, no_sync: bool = False) -> int:
    """--replay N: fetch the N latest published draws, backtest this
    tracker's exact pipeline on them, and write state + log. The live
    pending prediction and verified history are left untouched."""
    print(f"replay: loading the {n} most recent published Da Ma Cai draws...")
    try:
        if no_sync:
            corpus = load_corpus(state).tail(n)
            draws = [{'for_date': row['DrawDate'].strftime('%Y%m%d'), 'slots': _draw_to_slots(row)}
                     for _, row in corpus.iterrows()]
        else:
            draws = collect_recent_draws(n)
    except Exception as e:
        print(f"ERROR: replay fetch failed ({type(e).__name__}: {e}); "
              "nothing written")
        return 1
    if not draws:
        print("replay: no draws available; nothing written")
        return 1

    print(f"replay: walk-forward over {len(draws)} draws "
          f"({draws[0]['for_date']} .. {draws[-1]['for_date']})")
    rows, skipped = walk_forward_replay(draws, state)
    if not rows:
        print(f"ERROR: every replay draw failed "
              f"({len(skipped)} skipped); nothing written")
        return 1
    if dry_run:
        print("replay: dry-run, state and log not written")
        return 0

    previous_replay = state.get('replay')
    if previous_replay:
        state.setdefault('replay_archive', []).append(previous_replay)
    state['observed_draws'] = {**observed_draws(state), **{spec['for_date']: spec['slots'] for spec in draws}}
    state["replay"] = {
        'pipeline_version': PIPELINE_VERSION,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "window": [rows[0]["for_date"], rows[-1]["for_date"]] if rows else [],
        "rows": rows,
        "skipped": skipped,
    }

    log = render_log(state, network_ok=True)
    save_state_and_log(state, log)
    return 0


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
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if not isinstance(state, dict) or not isinstance(state.get('history'), list):
            raise ValueError('invalid prediction ledger; refusing to discard existing records')
        return state
    return {"pending": None, "history": []}


def save_state_and_log(state: dict, log: str) -> None:
    """Persist state even when a metadata-only change leaves the log identical."""
    changed = False
    for path, content in [(STATE_FILE, json.dumps(state, indent=2, ensure_ascii=False) + '\n'),
                          (LOG_FILE, log)]:
        if not path.exists() or path.read_text(encoding='utf-8') != content:
            temporary = path.with_suffix(path.suffix + '.tmp')
            temporary.write_text(content, encoding='utf-8')
            temporary.replace(path)
            changed = True
    print('updated prediction state/log' if changed else 'no changes')


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
    ap.add_argument('--for-date', help='explicit future YYYYMMDD target, e.g. an announced Tuesday special draw')
    ap.add_argument("--replay", type=int, default=0, metavar="N",
                    help="walk-forward replay over the N most recent published "
                         "draws (backtests this tracker's own pipeline; writes "
                         "state + log, leaves the live pending untouched)")
    args = ap.parse_args(argv)
    if args.replay < 0:
        ap.error('--replay must be nonnegative')

    state = load_state()

    # replay mode is a standalone backtest: it shares the prediction logic
    # but skips the sync/verify/generate pipeline entirely
    if args.replay > 0:
        return run_replay_mode(args.replay, state, dry_run=args.dry_run, no_sync=args.no_sync)

    corpus = load_corpus(state)
    damacai_cache = corpus

    # ---- 1) sync: latest published draws ---------------------------------
    network_ok = True
    recent: dict[str, dict] = {}
    if not args.no_sync:
        try:
            recent = fetch_recent_draws(max_dates=5)
            print(f"synced: {sorted(recent.keys())}")
        except Exception as e:
            network_ok = False
            print(f"WARNING: sync failed ({type(e).__name__}: {e})")

    # ---- 2) resolve the pending prediction (if any) ----------------------
    pending = state.get("pending")
    if args.for_date and pending and args.for_date != pending['for_date']:
        raise ValueError('a published prediction is already pending; refusing to replace it')
    if pending:
        target = str(pending["for_date"]).zfill(8)
        actual = recent.get(target)
        if actual is None:
            actual = draw_from_local_cache(damacai_cache, target)
        if actual is None and network_ok and not args.no_sync:
            today = datetime.now(timezone(timedelta(hours=8))).date()
            try:
                pub = datetime.strptime(target, "%Y%m%d").date()
            except ValueError:
                pub = None
            if pub and pub > today:
                pass  # Never ask a result endpoint to resolve a future draw.
            elif pub and (today - pub).days > PENDING_TTL_DAYS:
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
        if actual and complete_draw(actual):
            recent[target] = actual
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
                'prediction_metadata': {k: v for k, v in pending.items()
                                        if k not in {'picks', 'random_control', 'for_date'}},
            })
            state["history"].sort(key=lambda h: h["for_date"])
            state["pending"] = None
            print(f"resolved pending prediction for {target}: "
                  f"{len(state['history'][-1]['hits'])} hit(s)")

    state['observed_draws'] = {**observed_draws(state), **recent}
    corpus = load_corpus(state)

    # ---- 3) generate a new prediction if none pending --------------------
    if state.get("pending") is None:
        if not network_ok:
            print('sync failed; declining to create a new prediction')
            return 1
        newest = corpus['DrawDate'].max()
        today_myt = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d')
        target = args.for_date or next_draw_date(max(newest.strftime('%Y%m%d'), today_myt))
        target_date = datetime.strptime(target, '%Y%m%d')
        if target <= today_myt:
            raise ValueError('target must be a future Malaysian calendar date; cannot backdate predictions')
        top3 = extract_top3_prizes(corpus[corpus['DrawDate'] < target_date])
        picks, methods_ok = generate_picks(top3)
        state["pending"] = {
            "for_date": target,
            "picks": picks,
            "random_control": random_control(target),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **training_metadata(top3),
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

    save_state_and_log(state, log)
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
    a("> Checked daily by a scheduled GitHub Action after normal draw hours."
      " Regular targets are Wed / Sat / Sun; announced special draws require"
      " an explicit target date. Predictions are generated"
      " and logged **before** the next draw, then verified against the real"
      " result from Da Ma Cai's public endpoint.")
    a("")
    a("## How to read this")
    a("")
    a("Every tracked target gets two 10-number sets, published in advance:")
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
    a("P(hit) ≈ 1 - (1 - 10/10000)^23 ≈ 2.27%  →  ~1 hit per 44 draws")
    a("```")
    a("")
    a("This uses independent uniform four-digit slots as an approximation."
      " Da Ma Cai derives four-digit results from six-digit draws; four-digit"
      " suffixes need not be distinct. The earlier combination formula assumed"
      " 23 distinct four-digit winners. Duplicate prize slots remain separate"
      " when counting slot hits.")
    a("")
    a("Under that null, both columns should approach the baseline. If the model"
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
        if pending.get('pipeline_version') == PIPELINE_VERSION:
            a(f"Training: **Da Ma Cai**, {pending['training_rows']:,} draws;"
              f" pipeline `{PIPELINE_VERSION}`. Cutoff above is the actual latest training draw.")
        else:
            a("> **Audit correction:** this prediction belongs to the original pipeline"
              " and is preserved as published. Its displayed corpus cutoff advanced"
              " with fetched results even though those results were not added to"
              " training. In a fresh Actions checkout, training actually ended on"
              " **2025-12-28** in the bundled Magnum history. New predictions will use"
              " the separately versioned Da Ma Cai corpus.")
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
    if hist:
        a('### Results by pipeline version')
        a('')
        a('| Pipeline | Draws | Model hit draws | Random hit draws |')
        a('|---|---:|---:|---:|')
        versions = sorted({h.get('prediction_metadata', {}).get('pipeline_version', 'legacy-v1') for h in hist})
        for version in versions:
            group = [h for h in hist if h.get('prediction_metadata', {}).get('pipeline_version', 'legacy-v1') == version]
            a(f"| `{version}` | {len(group)} | {sum(bool(h['hits']) for h in group)} | {sum(bool(h['random_hits']) for h in group)} |")
        a('')

    replay = state.get("replay") or {}
    replay_rows = replay.get("rows", [])
    if replay_rows:
        n_r = len(replay_rows)
        rp_md = sum(1 for r in replay_rows if r["hits"])
        rp_rd = sum(1 for r in replay_rows if r["random_hits"])
        rp_ms = sum(len(r["hits"]) for r in replay_rows)
        rp_rs = sum(len(r["random_hits"]) for r in replay_rows)
        legacy_replay = replay.get('pipeline_version') != PIPELINE_VERSION
        replay_title = 'Legacy fixed-corpus comparison' if legacy_replay else 'Walk-forward replay'
        a(f"## 🔁 {replay_title} over {n_r} draws (retrospective)")
        a("")
        if replay.get("window"):
            a(f"_Replay window: {replay['window'][0]} → {replay['window'][1]} •"
              f" run {replay.get('run_at', 'n/a')[:10]}._")
            a("")
        sk = replay.get("skipped") or []
        if sk:
            a(f"_Note: {len(sk)} draw(s) could not be replayed in this run and"
              f" are absent from the table below: "
              f"{', '.join(s['for_date'] for s in sk)}._")
            a("")
        if legacy_replay:
            a("> **Audit correction:** all these replay dates trained on the same"
              " 5,568 Magnum draws ending **2025-12-28**, then scored against Da Ma Cai."
              " There was no new 2026 training data between folds, explaining the"
              " repeated picks. This is a cross-operator, fixed-corpus comparison;"
              " it is not evidence of an updating same-operator forecast. Original"
              " picks and scores are retained for transparency.")
        else:
            a("> Retrospective evaluation of the Da Ma Cai pipeline: each fold"
              " trains strictly before its target date and includes earlier"
              " observed replay draws. These rows were reconstructed after the"
              " fact; the separately timestamped live ledger remains primary evidence.")
        a("")
        a("| Draw | Model picks | Actual 1st/2nd/3rd | Model hits | Random hits |")
        a("|------|-------------|--------------------|------------|-------------|")
        for r in replay_rows:
            picks_md = ", ".join("`" + p + "`" for p in r["predicted"])
            top3_actual = "/".join(r["actual"].get(c, "–") for c in TOP3_COLS)
            hit_txt = (" ✅ " + "; ".join(
                f"{x['number']} ({x['tier']})" for x in r["hits"])) if r["hits"] else "—"
            rand_txt = (" ✅ " + ", ".join(x["number"] for x in r["random_hits"])) \
                if r["random_hits"] else "—"
            a(f"| {r['for_date']} | {picks_md} | `{top3_actual}` | {hit_txt} | {rand_txt} |")
        a("")
        a(f"**Replay tally:** model hit on {rp_md}/{n_r} draws"
          f" ({rp_md / n_r * 100:.2f}%) · random on {rp_rd}/{n_r} draws"
          f" ({rp_rd / n_r * 100:.2f}%) · expected ~2.27% per draw."
          f" Slot hits across all 23 slots: model {rp_ms}, random {rp_rs}.")
        a("")
    a("## Corpus & method notes")
    a("")
    a("- **Current training corpus** — tracked `data/damacai.csv` plus Da Ma Cai"
      " results retained in `predictions.json`. See `data/provenance.json` for"
      " snapshot coverage and validation limits. Magnum and the unverified"
      " Excel series are excluded. Local sync caches cannot silently change"
      " the scheduled experiment.")
    a("- **Versioning** — original predictions and legacy replay are preserved."
      f" Future predictions record `{PIPELINE_VERSION}`, row count, actual"
      " training cutoff and corpus SHA-256. Scores across versions are not"
      " evidence that the new pipeline has improved; it needs its own future results.")
    a("- **Prediction method** — `predictor_suite.run_all(include_ml=False)`"
      " (8 statistical lenses: frequency, 90-day hot, overdue-gap, digit"
      " probabilities, permutation families, Markov, ARIMA digit-sum,"
      " Bayesian Dirichlet) aggregated by cross-method consensus.")
    a("- **Verification** — Da Ma Cai public endpoint, polite 1 req/sec"
      " fetches, no scraping of protected pages.")
    a("")
    a("_Reminder: independent uniform outcomes are the null model. Nothing here is a")
    a("betting strategy — this log is an open experiment whose expected outcome")
    a("is that both columns match the ~2.27% baseline._")
    a("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
