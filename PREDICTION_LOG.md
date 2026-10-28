# 🎯 Live Prediction Tracker

> Checked daily by a scheduled GitHub Action after normal draw hours. Regular targets are Wed / Sat / Sun; announced special draws require an explicit target date. Predictions are generated and logged **before** the next draw, then verified against the real result from Da Ma Cai's public endpoint.

## How to read this

Every tracked target gets two 10-number sets, published in advance:

- **Model picks** — consensus of the suite's statistical methods.
- **Random control** — 10 purely random numbers, seeded by the draw date (`random.Random(YYYYMMDD).sample(range(10000), 10)`), so anyone can reproduce the luck-only baseline.

A pick *hits* if it appears among the draw's 23 prize slots (1st/2nd/3rd prize + 10 special + 10 consolation). Per-draw hit probability for *any* 10-number set:

```text
P(hit) ≈ 1 - (1 - 10/10000)^23 ≈ 2.27%  →  ~1 hit per 44 draws
```

This uses independent uniform four-digit slots as an approximation. Da Ma Cai derives four-digit results from six-digit draws; four-digit suffixes need not be distinct. The earlier combination formula assumed 23 distinct four-digit winners. Duplicate prize slots remain separate when counting slot hits.

Under that null, both columns should approach the baseline. If the model column ever drifts meaningfully above the random column on this i.i.d. stream, that would be extraordinary — the log makes that self-checkable.

## 🔮 Upcoming draw — 20261031 _(generated 2026-10-28, corpus cutoff 2026-10-28, 8 methods: frequency, recent_hot_90d, overdue, digit_probability, permutation_family, markov, arima, bayesian)_

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

**Random control:** `5009`, `3963`, `6154`, `1192`, `8397`, `0669`, `3419`, `4048`, `7796`, `3623`

> **Audit correction:** this prediction belongs to the original pipeline and is preserved as published. Its displayed corpus cutoff advanced with fetched results even though those results were not added to training. In a fresh Actions checkout, training actually ended on **2025-12-28** in the bundled Magnum history. New predictions will use the separately versioned Da Ma Cai corpus.

## ✅ Verified draws

| Draw | 10 picks | Actual 1st/2nd/3rd | Model hits | Random-control hits |
|------|----------|--------------------|------------|---------------------|
| 20261028 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `2694/6382/3450` | — | — |

## 📊 Lifetime tally

| Set | Draws tracked | Draws with a hit | Slot hits | Hit frequency | Expected |
|-----|--------------:|-----------------:|----------:|--------------:|---------:|
| Model picks | 1 | 0 | 0 | 0.00% (draws) · 0 slots | ~2.27% |
| Random control | 1 | 0 | 0 | 0.00% (draws) · 0 slots | ~2.27% |

### Results by pipeline version

| Pipeline | Draws | Model hit draws | Random hit draws |
|---|---:|---:|---:|
| `legacy-v1` | 1 | 0 | 0 |

## 🔁 Legacy fixed-corpus comparison over 40 draws (retrospective)

_Replay window: 20260801 → 20261027 • run 2026-10-27._

> **Audit correction:** all these replay dates trained on the same 5,568 Magnum draws ending **2025-12-28**, then scored against Da Ma Cai. There was no new 2026 training data between folds, explaining the repeated picks. This is a cross-operator, fixed-corpus comparison; it is not evidence of an updating same-operator forecast. Original picks and scores are retained for transparency.

| Draw | Model picks | Actual 1st/2nd/3rd | Model hits | Random hits |
|------|-------------|--------------------|------------|-------------|
| 20260801 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `2131/2803/8821` | — | — |
| 20260802 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `0195/2811/3801` | — | — |
| 20260805 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `7113/0942/9833` | — | — |
| 20260808 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `2150/7945/6293` | — | — |
| 20260809 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `6101/1265/4303` | — | — |
| 20260812 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `6960/7038/8223` | — | — |
| 20260815 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `3041/8111/3228` | — | — |
| 20260816 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `8974/5604/7811` | — | — |
| 20260819 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `0416/7463/0523` | — | — |
| 20260822 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `8805/9155/6209` | — | — |
| 20260823 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `4083/5421/7449` | — | — |
| 20260826 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `6179/1390/2155` | — | — |
| 20260829 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `2050/1211/2507` | — | — |
| 20260830 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `7342/1432/5587` | — |  ✅ 5201 |
| 20260901 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `7554/4949/3516` | — | — |
| 20260902 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `9732/6930/2191` | — | — |
| 20260905 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `7212/2926/1062` | — | — |
| 20260906 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `0694/8811/4027` | — | — |
| 20260909 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `3723/9524/9020` | — | — |
| 20260912 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `1840/6197/9429` |  ✅ 1840 (1st prize) | — |
| 20260913 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `6692/6261/4194` | — | — |
| 20260916 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `3961/3368/0852` | — | — |
| 20260919 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `0322/3250/3765` | — | — |
| 20260920 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `4671/3897/8018` | — | — |
| 20260923 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `6085/3437/1339` | — | — |
| 20260926 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `8315/3229/2597` | — | — |
| 20260927 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `2371/3396/3151` | — | — |
| 20260930 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `5069/9701/4368` | — | — |
| 20261003 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `9936/1196/5850` | — | — |
| 20261004 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `1293/2645/2519` | — | — |
| 20261007 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `4790/4925/3106` | — | — |
| 20261010 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `9612/1233/2270` | — | — |
| 20261011 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `7952/4847/9772` | — | — |
| 20261014 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `6906/7100/6516` | — | — |
| 20261017 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `5434/6342/7652` | — | — |
| 20261018 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `2454/1240/6521` | — | — |
| 20261021 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `3145/4268/4566` | — | — |
| 20261024 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `9250/5348/2726` | — | — |
| 20261025 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `2963/8780/4243` | — | — |
| 20261027 | `3845`, `1830`, `3830`, `1840`, `3840`, `1833`, `1834`, `3833`, `1831`, `1835` | `7750/3964/8263` | — | — |

**Replay tally:** model hit on 1/40 draws (2.50%) · random on 1/40 draws (2.50%) · expected ~2.27% per draw. Slot hits across all 23 slots: model 1, random 1.

## Corpus & method notes

- **Current training corpus** — tracked `data/damacai.csv` plus Da Ma Cai results retained in `predictions.json`. See `data/provenance.json` for snapshot coverage and validation limits. Magnum and the unverified Excel series are excluded. Local sync caches cannot silently change the scheduled experiment.
- **Versioning** — original predictions and legacy replay are preserved. Future predictions record `damacai-v2`, row count, actual training cutoff and corpus SHA-256. Scores across versions are not evidence that the new pipeline has improved; it needs its own future results.
- **Prediction method** — `predictor_suite.run_all(include_ml=False)` (8 statistical lenses: frequency, 90-day hot, overdue-gap, digit probabilities, permutation families, Markov, ARIMA digit-sum, Bayesian Dirichlet) aggregated by cross-method consensus.
- **Verification** — Da Ma Cai public endpoint, polite 1 req/sec fetches, no scraping of protected pages.

_Reminder: independent uniform outcomes are the null model. Nothing here is a
betting strategy — this log is an open experiment whose expected outcome
is that both columns match the ~2.27% baseline._
