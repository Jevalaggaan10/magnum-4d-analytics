# 🎯 Live Prediction Tracker

> Updated automatically by a scheduled GitHub Action on draw evenings (Wed / Sat / Sun), after results publish. Predictions are generated and logged **before** the next draw, then verified against the real result from Da Ma Cai's public endpoint.

## How to read this

Every draw gets two 10-number sets, published in advance:

- **Model picks** — consensus of the suite's statistical methods.
- **Random control** — 10 purely random numbers, seeded by the draw date (`random.Random(YYYYMMDD).sample(range(10000), 10)`), so anyone can reproduce the luck-only baseline.

A pick *hits* if it appears among the draw's 23 prize slots (1st/2nd/3rd prize + 10 special + 10 consolation). Per-draw hit probability for *any* 10-number set:

```text
P(hit) = 1 - C(9990, 10) / C(10000, 10) ≈ 2.27%  →  ~1 hit per 44 draws
```

Over time both columns should hover at that baseline. If the model column ever drifts meaningfully above the random column on this i.i.d. stream, that would be extraordinary — the log makes that self-checkable.

## 🔮 Upcoming draw — 20261028 _(generated 2026-10-25, corpus cutoff 2026-10-25, 8 methods: frequency, recent_hot_90d, overdue, digit_probability, permutation_family, markov, arima, bayesian)_

| # | Consensus pick |
|---|----------------|
| 1 | `3845` |
| 2 | `1830` |
| 3 | `3830` |
| 4 | `1840` |
| 5 | `3840` |
| 6 | `1833` |
| 7 | `1834` |
| 8 | `3833` |
| 9 | `1831` |
| 10 | `1835` |

**Random control:** `9150`, `9562`, `0744`, `9473`, `3475`, `0615`, `8179`, `8858`, `2325`, `4306`

## ✅ Verified draws

_No draws verified yet — the first scheduled prediction is pending._

## 📊 Lifetime tally

| Set | Draws tracked | Draws with a hit | Slot hits | Hit frequency | Expected |
|-----|--------------:|-----------------:|----------:|--------------:|---------:|
| Model picks | 0 | 0 | 0 | — | ~2.27% |
| Random control | 0 | 0 | 0 | — | ~2.27% |

## Corpus & method notes

- **Corpus** — bundled Magnum `4D.txt` history (up to 2026-10-25) + recent Da Ma Cai live syncs. The bundled `MDTdec28.xlsx` is a separate draw series that conflicts with `4D.txt` on shared dates, so it is excluded here until its provenance is sorted.
- **Prediction method** — `predictor_suite.run_all(include_ml=False)` (8 statistical lenses: frequency, 90-day hot, overdue-gap, digit probabilities, permutation families, Markov, ARIMA digit-sum, Bayesian Dirichlet) aggregated by cross-method consensus.
- **Verification** — Da Ma Cai public endpoint, polite 1 req/sec fetches, no scraping of protected pages.

_Reminder: Malaysian 4D draws are i.i.d. uniform. Nothing here is a
betting strategy — this log is an open experiment whose expected outcome
is that both columns match the ~2.27% baseline._
