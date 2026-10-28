# Magnum 4D Analytics — Honest Edition

A multi-method statistics + ML analytics suite for the Magnum 4D historical
draw dataset, with a Streamlit UI.

## Up-front honesty

This tool has **no demonstrated lottery prediction advantage**. Under an
independent uniform draw mechanism, historical results cannot improve the
next-number probabilities over `0000–9999`. Randomness tests can reveal
departures from a specified null; passing them does not prove independence.
The Big-bet simulator uses Magnum's fixed payout assumptions and an expected
house edge of roughly 36%; it is not a payout model for every selectable feed.

What this tool is good for:

1. Visualizing 30+ years of draw history.
2. Running a battery of randomness diagnostics, with sample-size limitations.
3. Seeing how different descriptive lenses (frequency, gap, Markov, LSTM,
   ARIMA, Bayesian Dirichlet, XGBoost, …) rank the 10,000 possible numbers.
4. Checking what history says about *your* favorite 4 digits.
5. Watching a Monte Carlo simulation lose your money — slowly,
   stochastically, and inevitably.

## 🎯 Live prediction tracker

Since no method beats random on i.i.d. draws, the repo does the only honest
next-best thing: **it logs its predictions publicly and verifies them against
the real results.**

A scheduled GitHub Action checks daily after normal Malaysian draw hours:

1. syncs the latest published Da Ma Cai result,
2. verifies any pending prediction against the real 23 prize numbers,
3. logs a fresh 10-number consensus prediction for the next draw — commit-timestamped **before** the draw happens,
4. pairs each prediction with a seeded **random control** of 10 numbers. Under independent uniform four-digit slots, `1 − (1 − 10/10000)^23 ≈ 2.27%` (about one hit per 44 draws). This is a suffix-draw approximation: Da Ma Cai derives four-digit winners from six-digit draws, so the earlier assumption of 23 distinct four-digit winners was not justified.

The tracker trains on **Da Ma Cai only**: the tracked `data/damacai.csv`
snapshot plus observed results retained in `predictions.json`. New predictions
record the actual latest training date, training row count, pipeline version,
and SHA-256 of the training inputs. Machine-local caches do not affect this
scheduled experiment. See `data/provenance.json` for snapshot coverage gaps
and the limits of the three official-endpoint spot checks.

Regular targets are Wednesday/Saturday/Sunday. Daily polling observes Tuesday
special results, but a special draw needs an explicit, announced future target
(`--for-date YYYYMMDD`) while no other prediction is pending. An unpublished
special forecast is never reconstructed as a prospective prediction.

`--replay N` evaluates the same current prediction function with strictly earlier
Da Ma Cai data, adding earlier observed outcomes between folds. Add `--no-sync`
to replay the newest N locally recorded draws offline. Previous replays are
archived when a new one is explicitly requested.

**2026-10-28 audit correction:** the original 40-row replay used the same
5,568 Magnum training rows ending **2025-12-28** for every 2026 target. It
therefore produced identical picks; it was a cross-operator fixed-corpus
comparison. The original live tracker also advanced its displayed cutoff
without adding fetched results to training. Original picks, timestamps, and
scores remain intact and are labeled as the legacy pipeline in the log.
The corrected pipeline is `damacai-v2`; improvement requires future evidence.

➡️ **[Read the full log → PREDICTION_LOG.md](PREDICTION_LOG.md)** — upcoming
prediction, verified draws with actual prizes, and the running
model-vs-random tally.

Reproduce locally (`--dry-run` writes nothing; `--no-sync` skips the network):

```powershell
python scripts/track_predictions.py --dry-run
```

## Install

```powershell
python -m pip install -r requirements.txt
```

Optional TensorFlow + XGBoost methods:

```powershell
python -m pip install -r requirements-ml.txt
```

Leave "Use ML models" off to run the statistical suite. Statistical and
Markov methods do not import either heavy runtime. The Action installs only
`requirements-tracker.txt`.

## Run the UI

```powershell
streamlit run app.py
```

Streamlit opens in your browser at `http://localhost:8501`.

## CLI entry points

Each module is runnable standalone:

```powershell
python data_loader.py            # sanity-check data load
python randomness_tests.py       # print all randomness-test p-values
python predictor_suite.py        # run every method, print top-10 per method
python consensus.py              # cross-method consensus table
python user_guess_analyzer.py 1234   # full report for a 4-digit number
python backtest.py --years 2020-2023 # walk-forward backtest vs random baseline
python ev_simulator.py           # quick EV demo
```

## Modules

| File | Purpose |
|------|---------|
| `data_loader.py` | Normalize all prize cells and dates, load one operator at a time, deduplicate by operator + date; Excel is a separate explicit series |
| `feature_engineer.py` | Per-number / per-digit / temporal features. Vectorized |
| `train_model.py` | `StatisticalModel`, `MarkovChainModel`, `XGBoostModel`, real `LSTMModel` |
| `predict_final.py` | Original CLI ensemble (frequency / overdue / patterns / ...) |
| `predict.py` | Earlier CLI runner (kept for reference) |
| `randomness_tests.py` | Chi-square, runs test, Ljung-Box, serial-pairs, Benford |
| `predictor_suite.py` | Uniform `MethodResult` interface for every prediction method |
| `consensus.py` | Cross-method agreement scoring |
| `user_guess_analyzer.py` | Report on user-input 4-digit number |
| `backtest.py` | Walk-forward validation vs random baseline (with Bonferroni) |
| `ev_simulator.py` | Monte Carlo bankroll simulator (Big bet payouts) |
| `app.py` | Streamlit UI tying it all together |
| `scripts/track_predictions.py` | Versioned Da Ma Cai tracker, complete-result verification, shared live/replay prediction function, immutable published pending picks |
| `.github/workflows/prediction-tracker.yml` | The Action that runs the tracker on draw evenings |
| `data/damacai.csv`, `data/provenance.json` | Reproducible Da Ma Cai training snapshot and coverage / verification metadata |
| `tests/` | Regression tests for ingestion, latest-seen index, caching, malformed values, chronological replay and tracking |
| `docs/MODEL_AUDIT.md` | Code audit, method map, research sources and ranked model roadmap |

## What the methods are doing

- **frequency_alltime** — counts of each number over full history.
- **recent_hot_90d** — counts within last 90 days.
- **overdue_gap** — days since last appearance, ranked descending.
- **digit_probability** — product of per-position empirical digit probs.
- **permutation_family** — numbers belonging to most-seen digit multisets.
- **markov_chain** — first-order Markov over within-number digit transitions.
- **arima_digit_sum** — ARIMA(2,0,2) forecast of next digit-sum.
- **bayesian_dirichlet** — Dirichlet-multinomial posterior over per-position
  digit distributions (closed-form, no training).
- **xgboost** — legacy historical appeared-vs-never classifier. Its `freq`
  feature encodes its label. Kept as a diagnostic; these scores are **not
  next-draw probabilities**. A future classifier needs chronological labels
  and features built strictly before each target draw.
- **lstm** — real `tf.keras.layers.LSTM(64)` over per-digit one-hot stream;
  samples next-digit distributions.

## The data

- `4D.txt` — 5,568 Magnum draws, 1992-05-06 through 2025-12-28, 23 prize slots.
- `MDTdec28.xlsx` — separate, provenance-unverified series. All 5,231
  overlapping dates disagree with the TXT top-three results; select it
  explicitly rather than pooling it with Magnum or Da Ma Cai.
- `data/damacai.csv` — 3,633 Da Ma Cai draws, 2005-01-01 through 2026-10-28,
  with coverage gaps documented in `data/provenance.json`.
- `data_cache/<operator>.csv` — populated by the live-sync feature (see below).

The UI's **Analyze draw series** selection controls the analyzed operator.
Its default is Magnum. Loader callers can use
`load_combined_data(operator='damacai')`; the Excel series is
`operator='unverified_excel'`. Syncing one operator does not mix it into
another operator's model.

## Live data sync

The app can pull fresh draws from Malaysian operators in two ways:

| Operator | Method | Auth |
|---|---|---|
| Da Ma Cai (PMP) | Public undocumented JSON endpoints | none (uses `cookiesession: 363` header) |
| Magnum | Bundled `4D.txt` historical file | none |
| 4dyes3.com | Config-driven adapter via `data_sources/fouryes3_config.yaml` | API key (env `FOURYES3_API_KEY` or sidebar input) |

From the Streamlit sidebar, pick an operator and click **⬇ Sync latest
results**. A polite 1 req/sec rate limit is enforced. New draws land in
`data_cache/<operator>.csv` and merge into that operator's selected dataset.

CLI equivalent:

```powershell
python -m data_sources.manager sync damacai --recent 30
python -m data_sources.manager sync all --recent 30
python -m data_sources.manager status
```

### Legal note

Live-sync scraping uses public-but-undocumented operator endpoints
(technique credit: Hong's Tech Blog). You are responsible for compliance
with each operator's terms of service in your jurisdiction. The app
ships polite defaults (1 req/sec) and never stores credentials on disk —
API keys live in `st.session_state` or environment variables only.

### 4dyes3.com note

`https://4dyes3.com/en/api-widget` is bot-protected (HTTP 403 to
unauthenticated fetches). The `fouryes3.py` adapter is therefore
config-driven via `data_sources/fouryes3_config.yaml` — paste a working
curl into that YAML and the adapter will use it. Without a working
config, the source reports itself as unavailable but does not crash the
app.

## Caveats baked into the code

The Backtest tab measures each method against a baseline using its actual
distinct pick count, with a binomial test and Bonferroni correction. It does
not force a negative result. A positive result calls for provenance checks,
out-of-sample replication and correction for all model/metric/window searches.
The annual backtest scores top-three prizes; the tracker scores 23 slots.
Their raw hit rates are different metrics. Consensus weights remain heuristic,
and correlated methods are not independent confirmations.

Run the regression checks with:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```
