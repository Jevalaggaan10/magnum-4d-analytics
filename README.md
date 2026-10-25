# Magnum 4D Analytics — Honest Edition

A multi-method statistics + ML analytics suite for the Magnum 4D historical
draw dataset, with a Streamlit UI.

## Up-front honesty

This tool **cannot predict the lottery**. Magnum 4D draws are i.i.d. uniform
over `0000–9999`. The published payout schedule has a ~36% house edge per
RM1 Big bet. No model on Earth — including every model in this repo —
beats random chance on i.i.d. data.

What this tool is good for:

1. Visualizing 30+ years of draw history.
2. Running a battery of randomness tests to confirm the data is random.
3. Seeing how different descriptive lenses (frequency, gap, Markov, LSTM,
   ARIMA, Bayesian Dirichlet, XGBoost, …) rank the 10,000 possible numbers.
4. Checking what history says about *your* favorite 4 digits.
5. Watching a Monte Carlo simulation lose your money — slowly,
   stochastically, and inevitably.

## Install

```powershell
python -m pip install -r requirements.txt
```

TensorFlow + XGBoost are heavy; if you don't want them, leave the "Include
ML methods" checkbox off in the UI sidebar and the suite skips them.

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
| `data_loader.py` | Load `4D.txt` and `MDTdec28.xlsx`, deduplicate, return clean DataFrame |
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
- **xgboost** — binary classifier (appeared vs never-appeared) over engineered
  features. Ranks all 10,000.
- **lstm** — real `tf.keras.layers.LSTM(64)` over per-digit one-hot stream;
  samples next-digit distributions.

## The data

- `4D.txt` — Magnum 4D draws (one row per draw, 23 prize slots per row).
- `MDTdec28.xlsx` — supplementary draw data.
- `data_cache/<operator>.csv` — populated by the live-sync feature (see below).

## Live data sync

The app can pull fresh draws from Malaysian operators in two ways:

| Operator | Method | Auth |
|---|---|---|
| Da Ma Cai (PMP) | Public undocumented JSON endpoints | none (uses `cookiesession: 363` header) |
| Magnum | Bundled `4D.txt` historical file | none |
| 4dyes3.com | Config-driven adapter via `data_sources/fouryes3_config.yaml` | API key (env `FOURYES3_API_KEY` or sidebar input) |

From the Streamlit sidebar, pick an operator and click **⬇ Sync latest
results**. A polite 1 req/sec rate limit is enforced. New draws land in
`data_cache/<operator>.csv` and merge automatically into the dataset.

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

Every prediction surface in the UI shows a disclaimer. The Backtest tab
will (by design) show that no method significantly beats the random
baseline after Bonferroni correction. That negative result is the most
important number in this whole repo.
