# Lottery model audit and research

Research and code review cutoff: **28 October 2026**. Scope: the first-party lottery repository, its datasets, adapters, user interfaces, scheduled tracker, and relevant primary research. This is a targeted literature review, not a claim to have inspected every publication or installed dependency.

**The highest-value next model is a calibrated Bayesian categorical model, followed by a small regularized classifier.** The audit found concrete bugs that made the previous comparison misleading: every 2026 replay fold used the same 5,568 Magnum rows ending on 28 December 2025, while the live tracker advanced its displayed cutoff without adding fetched results to training. Those boundaries have been corrected and the published predictions preserved. A corrected retrospective now updates its training history and produces 17 different prediction sets across 40 targets, but still records **one model hit-draw and one random-control hit-draw**. Today's single miss is unsurprising under the stated null, which predicts a miss about 97.7% of the time. The research reviewed here establishes no independently replicated, prospective Malaysian 4D advantage from historical winning numbers alone. The project can become a credible tool for detecting and testing possible signal; a stronger winning claim requires evidence that does not yet exist.

## The repeated picks came from unchanged training data

The old replay filtered the bundled `4D.txt` to dates before each target. That sounds like an expanding historical evaluation, but all 40 targets were in August–October 2026 and the file stopped at **2025-12-28**. The filter therefore returned exactly the same 5,568 training rows every time. Repeated outputs were the expected consequence of unchanged inputs. Independently, the live path used newly fetched result dates to update the displayed `corpus_cutoff`, although those result rows did not enter training. The earlier description of the replay as an updating forecast overstated what it tested. The actual mechanism is documented in `scripts/track_predictions.py::load_corpus`, `walk_forward_replay`, and `training_metadata`; the original records remain in `predictions.json`.

The corrected tracker uses a **Da Ma Cai-only** snapshot plus observed Da Ma Cai results retained in the ledger. It excludes Magnum, the unverified Excel series, and machine-local caches from the scheduled experiment. Future predictions record `damacai-v2`, the actual latest training date, training row count, and SHA-256 of the training inputs. The already published **31 October prediction remains the original prediction**. A new implementation does not justify quietly replacing that forecast after publication; version 2 takes effect when a new prediction is generated.

| Data source | Audited contents | Correct treatment |
|---|---|---|
| `4D.txt` | 5,568 Magnum draws, 1992-05-06 through 2025-12-28 | Separate Magnum series; valid top-three values in every row |
| `MDTdec28.xlsx` | 6,695 rows; 5,231 dates overlap TXT and every overlapping top-three tuple differs; two duplicate dates | Preserve as an explicitly selected, provenance-unverified series |
| Original local Da Ma Cai cache | 3,592 draws, 2005-01-01 through 2026-05-23 | Operator-specific historical input, retained locally |
| New `data/damacai.csv` | 3,633 complete observed draws through 2026-10-28 | Reproducible tracker snapshot with documented coverage gaps |

The snapshot combines the existing Da Ma Cai cache with 41 later observed ledger dates. All 23 prize slots matched the official adapter on **2005-01-01, 2024-05-01, and 2026-05-23**. Three checks support provenance but do not validate every historical row. The archive has gaps after May 2026; missing results were not manufactured. The last Magnum TXT row also lacks its tenth special and all ten consolation values; the loader now preserves those as missing. `data/provenance.json` records the snapshot's source, coverage, checks, and file hash.

The corrected 40-target analysis is a new, exploratory retrospective over the original target window, not a replacement for the original ledger:

| Comparison | Training behavior | Distinct ten-pick sets | Model hit-draws | Random hit-draws |
|---|---|---:|---:|---:|
| Original retrospective | Same 5,568 Magnum rows; cutoff 2025-12-28 throughout | 1 | 1/40 | 1/40 |
| Corrected retrospective | Earlier Da Ma Cai observations added; 3,592 → 3,631 training rows | 17 | 1/40 | 1/40 |

The corrected folds ran without skipped targets. Their actual training cutoffs move from **2026-05-23 to 2026-10-25**, always before the target date. This proves the repaired chronology advances; it does not prove the model improved. `statsmodels` emitted ARIMA convergence warnings during these folds, which are another reason to treat the rankings as exploratory. Measured fold details and overlap calculations are saved in the accompanying research notes' `implementation_evidence.json`.

## The repository is connected, and its ten methods answer different questions

The main architecture is a shared data and ranking pipeline with two interfaces: a Streamlit explorer and an advance-logged Da Ma Cai experiment. It also contains useful diagnostics and older runnable CLIs. Those entrypoints were retained because overlapping code is not sufficient evidence that a public feature is unnecessary.

```text
lottery/
├── app.py                         Streamlit explorer and per-series selection
├── data_loader.py                 Dates, prize validation, provenance and dedup
├── feature_engineer.py            Digit, frequency, gap and rolling features
├── predictor_suite.py             Eight core + two optional method wrappers
├── train_model.py                 Statistical, Markov, XGBoost and LSTM classes
├── consensus.py                   Agreement and reciprocal-rank aggregation
├── backtest.py                    Yearly forward evaluation of top-three hits
├── randomness_tests.py            Uniformity, runs, pairs, Ljung–Box and Benford
├── user_guess_analyzer.py         Historical report for a supplied number
├── ev_simulator.py                Synthetic Magnum Big-bet payout simulator
├── predict.py                     Earlier training/ensemble CLI, retained
├── predict_final.py               Six-lens statistical CLI, retained
├── scripts/
│   └── track_predictions.py       Live verification and chronological replay
├── data_sources/
│   ├── __init__.py                Adapter exports
│   ├── base.py                    Canonical draw and source interfaces
│   ├── damacai.py                 Public Da Ma Cai result adapter
│   ├── magnum.py                  Bundled TXT adapter; not a live Magnum feed
│   ├── fouryes3.py                Configurable, provenance-unverified feed
│   ├── fouryes3_config.yaml       Endpoint and field configuration
│   └── manager.py                Per-operator incremental synchronization
├── data/
│   ├── damacai.csv                Tracked Da Ma Cai training snapshot
│   └── provenance.json           Coverage and verification metadata
├── tests/
│   ├── test_data_and_features.py
│   ├── test_predictors.py
│   ├── test_tracker.py
│   └── test_backtest.py
├── docs/MODEL_AUDIT.md             This audit and research report
├── 4D.txt                         Separate Magnum history
├── MDTdec28.xlsx                  Separate unverified history
├── predictions.json              Published predictions, actuals and versions
├── PREDICTION_LOG.md              Human-readable ledger
├── requirements.txt              Application dependencies
├── requirements-ml.txt           Optional heavy ML dependencies
├── requirements-tracker.txt      Scheduled experiment dependencies
├── .github/workflows/prediction-tracker.yml
├── .streamlit/config.toml
├── .gitattributes                 Pins data CSV/JSON to LF for stable file hashes
├── .gitignore
└── README.md

Local material retained: data_cache/, .venv/, __pycache__/, logs.
```

```text
One selected operator's files/cache
  → data_loader → validated chronological rows
      ├─ predictor_suite → MethodResult lists → consensus → UI / CLI
      ├─ randomness_tests → diagnostic tables
      ├─ user_guess_analyzer → historical number report
      └─ backtest → forward yearly top-three comparison

Official Da Ma Cai results
  → adapter → complete result → score immutable pending prediction
  → persist observations → next training corpus → new future prediction
  → predictions.json → PREDICTION_LOG.md

ev_simulator → synthetic draws + fixed Magnum payout assumptions
```

The scheduled tracker runs the **eight core methods**, with heavy ML disabled. The optional methods appear in the UI/CLI when requested; their existence does not establish predictive validity. There are no random-forest or MLP implementations in this repository.

| Method | Actual calculation | Main limitation |
|---|---|---|
| All-time frequency | Counts exact top-three codes | Historical popularity is not a conditional forecast |
| Recent hot, 90 days | Counts within a trailing calendar window | The window and persistence hypothesis need forward validation |
| Overdue gap | Time since latest appearance across top-three columns | Independent draws do not make absent numbers become due |
| Digit probability | Product of four empirical positional digit probabilities | Assumes positional independence; small frequency differences can dominate rankings |
| Permutation family | Counts sorted digit multisets; returns observed members | Family sizes differ: 24 arrangements for four distinct digits, one for `0000` |
| Markov chain | Initial digit and adjacent-digit transitions inside a code | This is within-number structure, not transitions between draw dates |
| ARIMA digit sum | Forecasts first-prize digit sum, then ranks codes near that sum | Many codes share a sum; aggregate prediction is not exact-code prediction |
| Bayesian Dirichlet | Smoothed positional digit marginals | Already closely resembles digit probability; not yet calibrated by forward scoring |
| XGBoost, optional | Historical ever-seen versus never-seen classification | `freq > 0` exactly reveals its label; not a next-draw target |
| LSTM, optional | Autoregressive digits from a flattened first-prize stream | Overlapping windows and digit accuracy do not supply an independent exact-draw test |

`train_model.StatisticalModel` combines historical frequency, positional sampling, and permutation sampling. In normal use its frequency candidates already fill the requested output, so the audit added a direct return instead of doing unused sampling work. `predict_final.py` offers frequency, recent-hot, overdue, digit sampling, permutation, and random pattern demonstrations. `predict.py` runs the training classes and an older weighted ensemble. The older CLIs still contain unvalidated “confidence” and prize-tier suggestion language: assigning rank groups to first/second/third prizes does not train a tier-specific model.

The audit corrected the following reproduced defects while retaining datasets, adapters, public entrypoints, and published results:

| Reproduced defect | Implemented correction |
|---|---|
| Frozen replay inputs and misleading live cutoff | Same-operator observed history, shared live/replay prediction function, actual cutoff/count/hash/version |
| TXT, conflicting Excel, and other operators pooled together | Explicit series selection; deduplication by canonical operator and date |
| Numeric cache inference lost leading zeros; missing values became malformed codes or `0000` | One validator across all 23 slots; missing remains missing |
| Older third-prize occurrence overwrote a newer first-prize date | Last-seen index retains the maximum date across columns |
| Streamlit ignored `_results` in cache keys; row counts hid same-length corrections | Actual result/DataFrame contents determine predictor and consensus cache keys |
| Missing adapter slots shifted position; numeric zero could disappear | Preserve slot positions, reject malformed required prizes, retain genuine `0000` |
| `--no-sync` could still fetch a pending result | Offline execution avoids that network path |
| Corrupt ledger could silently reset; metadata-only changes could fail to save | Fail explicitly on corrupt state; persist state independently of rendered Markdown |
| Partial results could prematurely resolve a prediction | Require complete 23-slot results before verification |
| Duplicate votes, invalid candidates, and insertion-dependent consensus ties | One valid vote per method/code; deterministic tie ordering |
| ARIMA missing-column result used wrong dataclass fields; LSTM omitted its final target window | Correct result construction and sequence bounds |
| Backtest assumed requested rather than actual unique pick count | Baseline uses the valid distinct candidates actually scored |
| User-guess repeat metric used top three but baseline used 23 slots | Match the null to the analyzed columns |
| Sparse chi-square and short/degenerate diagnostic inputs gave unreliable results or crashed | Mark unsupported p-values unassessed; validate codes, lags and contingency tables |
| Global warning suppression and unnecessary heavy imports hid problems or slowed statistical use | Remove global suppression; load optional ML runtimes only when needed |
| Hardcoded Windows paths and legacy empty/sparse-input crashes | Derive paths from modules; guard empty ensembles and impossible unique sampling |

Removed code was limited to verified unused imports, variables, pattern counts, and unnecessary computation. Public features remain available. **33 unit tests passed**, including actual Streamlit cache regressions, and full UI smoke checks passed for Magnum and Da Ma Cai. Compilation and diff checks passed, and loading also worked from an unrelated working directory. A regression verifies platform-independent corpus hashes using explicit YYYY-MM-DD dates and LF line endings; `.gitattributes` pins tracked data CSV/JSON line endings to preserve file hashes. Optional full TensorFlow/XGBoost training was not rerun; lazy import and LSTM sequence preparation were tested. The original pending prediction, verified history, and legacy replay were checked for exact preservation. The rendered log separates pipeline versions, and the tracker avoids fetching future pending results.

## Anti-correlation means three different things here

In simple terms, **anti-correlation means one outcome tends to move opposite another**. It does not mean that a wrong prediction can be inverted into a winning number. Three relationships must be kept separate.

Across draw dates, independence means yesterday's result does not change tomorrow's probabilities. An overdue rule assumes compensation; a hot-number rule assumes persistence. Those are testable hypotheses, not consequences of randomness. On the corrected snapshot, first-prize digit values had the following descriptive lag-one Pearson correlations, ordered by draw rather than equal calendar spacing:

| Digit position | Previous-draw correlation |
|---|---:|
| Thousands | −0.0059 |
| Hundreds | +0.0196 |
| Tens | +0.0132 |
| Units | −0.0131 |

These small coefficients do not establish exploitable negative dependence. No adjusted significance claim follows from this exploratory view, and it does not test every nonlinear relationship. The randomness battery also mixes within-code and cross-boundary digits in some tests; its output needs a stated null and multiple-testing treatment.

Within a single draw, sampling constraints can create negative dependence. In a six-of-49 game without replacement, including one ball leaves fewer places for another; inclusion indicators for two specified balls have correlation −1/48. That is a property of that mechanism, not evidence that a number appearing today is less likely in the next independent draw. Covariance-aware lottery auditing has a formal literature; it is an audit method, not a demonstrated profitable forecaster. ([Coronel-Brizio and colleagues, 2008](https://arxiv.org/abs/0806.4595))

Da Ma Cai is different from a simple six-of-49 game. Its official procedure generates six-digit outcomes and uses their last four digits for 1+3D. The snapshot itself contains **93 draws with repeated four-digit codes across prize slots**. The original “23 distinct winning codes” assumption therefore does not describe this dataset. Duplicates must remain when counting prize-slot hits. ([Official Da Ma Cai procedure](https://www.damacai.com.my/general-information/))

For ten distinct picks, an independent-uniform-slot approximation gives:

```text
P(any hit in 23 slots) ≈ 1 − (1 − 10/10000)^23 = 2.2749%
P(miss on one draw)   ≈ 97.7251%
Expected hit-draws/40 ≈ 0.9100
P(no hits in 40)      ≈ 39.83%
```

The old combination formula is correct for a different null with 23 distinct four-digit winners. Both round to 2.27%, so the practical one-miss conclusion barely changes, but the mechanism should be stated correctly. For a particular observed draw with `u` distinct winning suffixes, a uniformly sampled ten-code control has exact conditional any-hit probability `1 − C(10000−u,10)/C(10000,10)`. One hit in 40 gives an approximate exact-binomial 95% interval from **0.063% to 13.16%** under constant independent hit probabilities—far too broad to resolve a small advantage.

Between models, the central issue is redundancy. On the 3,633-row snapshot, **digit probability and Bayesian Dirichlet selected exactly the same top 50 codes**, giving Jaccard overlap 1.000. Their vote counts should not be interpreted as independent confirmations. Digit probability and Markov shared four of their top 50, giving Jaccard 4/96 = 0.0417. Many pairs shared none. Two independently chosen 50-code subsets of 10,000 have expected intersection only 0.25, so zero overlap alone is not evidence of anti-correlation. Jaccard measures shared selections, not serial correlation, calibration, or prediction accuracy. Deliberately forcing methods to disagree cannot manufacture information about an independent draw.

## Current papers do not validate a stronger lottery predictor

The strongest recent lottery-specific source located was Dinu, Modan and Frunzete's *Experimental Limits of Machine Learning on Near-Random Data: A Case Study of the Romanian Lottery System*. First published online **30 June 2026**—despite a 2027 bibliographic copyright year—its accessible abstract describes 2,510 Romanian 6/49 draws and frequency, Markov, MLP and random-forest comparisons with temporal separation. It reports no statistically significant improvement over chance. The chapter was paywalled, so this review checked the publisher's abstract, not the full implementation or an independent replication. It is evidence about that study, not proof about every Malaysian lottery. ([Publisher chapter](https://link.springer.com/chapter/10.1007/978-3-032-30518-3_23))

A contrary **May 2026 self-publication** on Kerala lottery book-level outcomes claims an 11.55% hit rate against a 10% null. It is a different task, over 91 days and many model/window combinations, without an independently verified replication in this review. Its timestamped archive and treatment of observations sharing a draw would need reproduction; the paper's simulated financial outcomes still lose money. The claim should be examined rather than hidden, but it does not establish a ten-code Malaysian 4D advantage. ([Original Kerala paper](https://blog.iamnk.com/wp-content/uploads/2026/05/Lucky_Keralam_Research_Paper.pdf))

A 2026 arXiv study of repeated LLM-generated six-of-49 tickets found concentrated outputs across 1,200 attempted calls. That evaluates output diversity, not forecasting skill. Its displayed submission date conflicts with the identifier's month, so its metadata deserves caution. It supports retaining an actual seeded PRNG for the control, rather than asking an LLM to “be random.” Lottery-sales forecasting and neural-network “lottery ticket hypothesis” papers address different problems. ([Primary LLM preprint](https://arxiv.org/abs/2610.00052))

Real exploitable failures can exist when the mechanism is compromised: Iowa Lottery's account of the Eddie Tipton case describes insider-installed code making certain computer-generated draws predictable. That is evidence about a compromised generator, not a universal pattern learnable from fair public draw histories. Statistical tests also cannot certify unpredictability; passing a battery is not a proof of fairness, and failing one is not automatically a usable forecast. ([Iowa Lottery statement](https://ialottery.com/Pages/Pressroom/OtherNews_Detail.aspx?otherNewsID=20), [NIST's testing limits](https://csrc.nist.gov/pubs/sp/800/22/r1/upd1/final))

Modern general-purpose models are substantially more capable than this repository's experiments, but their benchmark results concern other data-generating processes:

| Current candidate, verified by the cutoff | What the primary source establishes | Priority here |
|---|---|---|
| TabPFN-3.5, announced 2026-09-15 | Current tabular foundation model; vendor technical report includes temporal/grouped benchmarks | Optional offline challenger after the target and evaluation are fixed; downloaded-weight licensing restricts commercial/production use |
| TimesFM-3, announced 2026-08-31 | Google's multivariate/covariate time-series foundation model | Low priority for exact categorical lottery codes |
| Chronos-2, paper 2025-10-17 and release 2025-10-20 | Univariate, multivariate and covariate-informed forecasting | Optional structured-data experiment; no lottery advantage demonstrated |
| Datadog Toto 2.0, dated announcement 2026-05-14 | Time-series/quantile models, observability focus, Apache 2.0 release | Low priority; unrelated to the Sports Toto lottery operator |

These dates and scopes come from the developers' primary publications, not a claim of independent lottery validation. ([TabPFN-3.5 report](https://priorlabs.ai/technical-reports/tabpfn-3-5), [weight license](https://huggingface.co/Prior-Labs/tabpfn_3_5/blob/main/LICENSE), [TimesFM-3](https://www.research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/), [Chronos-2 paper](https://arxiv.org/abs/2510.15821), [Toto 2.0](https://www.datadoghq.com/blog/ai/toto-2/))

## Upgrade the probability experiment in a fixed order

**First improve the existing Bayesian model.** Produce a normalized probability distribution with shrinkage toward uniform, choose the shrinkage only inside earlier training folds, and compare forward log loss and calibration against uniform. Do not add another almost identical digit counter and call the extra vote a new model. A hierarchical extension can test shared information across prize tiers while retaining operator separation. Dirichlet pseudo-counts give a transparent starting point: `(count_k + alpha_k)/(N + sum(alpha))`. ([Primary derivation](https://www.cs.mcgill.ca/~isabeau/COMP551/F23/slides/2-maximumlikelihood.pdf))

Then add **regularized multinomial logistic classification**, followed by **one shallow CatBoost or LightGBM challenger** with a small, recorded tuning budget. They can test lagged nonlinear relationships without immediately committing to a deep network. CatBoost's ordered boosting addresses a particular prediction-shift problem; it cannot repair labels or features that already contain the target. Switching the current ever-seen XGBoost task to another library would preserve the central error. Rebuild examples at historical cutoffs, with features available before each draw and labels taken from the next draw. ([CatBoost paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html), [LightGBM paper](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html), [Leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html))

Treat `0000`–`9999` as **categorical codes**, not magnitudes to regress. A 10,000-class distribution has 9,999 free parameters; four independent ten-class digit distributions have 36. That reduction is useful, but the digit-independence assumption must be stated and checked. A code one integer away from a winner is not a meaningful near miss. A next-first-prize task, a next-top-three task and an any-of-23 task also require different labels and nulls. All slots belonging to one draw must remain in the same evaluation partition.

| Phase | Concrete deliverable | Acceptance condition |
|---|---|---|
| Data foundation | Fill documented archive gaps; preserve operator, source, retrieval and draw identity; freeze dataset version | Reproducible data hash and no cross-operator mixing or invalid prize manufacture |
| Probability baseline | Uniform plus calibrated Dirichlet shrinkage | Complete normalized distributions, chronological scoring, explicit slot target |
| Compact challengers | Regularized logistic model and one shallow boosting library | Better untouched outer-fold probability scores than uniform and the simpler model, with uncertainty |
| Optional foundation model | TabPFN-3.5 under suitable access/license | Same frozen task and folds; improvement beyond benchmark reputation |
| Live promotion | Freeze selected model/weights; timestamp predictions before results | Independent future evidence under declared checkpoints or valid sequential monitoring |

The current annual backtest fixes a set for a year and scores top-three hits; the tracker updates sets and scores 23 slots. Their percentages are not interchangeable. Neither currently implements the full normalized-probability, nested-tuning protocol above. Use inner chronological splits for hyperparameters and outer untouched periods for comparison, group by complete draws, and account for archive gaps and irregular calendar spacing. Proper log loss and Brier scores assess probabilities; uniform single-code log loss is `ln(10000) ≈ 9.2103` nats. Keep exact hit metrics as secondary, predeclared outcomes. ([Chronological splitting guidance](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [2026 review of proper scoring rules](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-042424-050626))

Keep the date-seeded control for transparency and add many correctly simulated controls for uncertainty. Record every tried model, window, feature and metric; adjust for the full selection process. Repeatedly checking ordinary 5% tests until one passes is not valid monitoring. Use a predeclared fixed checkpoint or a suitable confidence sequence. Illustrative exact-binomial calculations under the **old distinct-slot null**, one model and one fixed checkpoint need about 372 draws to detect a doubled hit probability, 1,289 for a 50% lift, and 7,216 for a 20% lift at 80% power. These are planning illustrations, not exact Da Ma Cai requirements; multiple models, repeated looks and nonstationarity change them. ([Exact binomial testing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html), [Time-uniform confidence sequences](https://arxiv.org/abs/1810.08240))

The remaining weaknesses are explicit: XGBoost's historical target and leakage; LSTM's overlapping-window validation; permutation-family size bias; heuristic correlated consensus weights; ARIMA's aggregate target and fit warnings; diagnostic multiple testing; incomplete historical coverage; and the simulator's fixed Magnum mechanism. Daily result polling now catches special results, but a Tuesday special still needs an announced future target to be prospectively covered. A small hit-rate improvement would also not automatically imply positive financial expectation: payout by tier and stake cost matter, and shared-jackpot advice does not transfer automatically to fixed-payout 4D. The useful next milestone is a trustworthy, falsifiable probability experiment whose outcome can be either a validated signal or a well-supported null result.
