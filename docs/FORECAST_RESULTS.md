# Forecasting implementation results — 2026-11-03

Uniform remains champion. No practical probability improvement or exact-hit
advantage was established. Better software and a falsifiable probability experiment
are delivered; future confirmation has not happened.

## Development (345 complete draws, 2018–2019, exploratory)

Uniform marginal log loss: **9.210340372** nats/slot. Best observed nonuniform
candidate: positional digits, pseudocount 100, 50% mixture with uniform;
**9.210254937**. Gain **0.000085435**, below declared **0.001** threshold.
Paired difference (challenger minus uniform), 98.75% interval, 13-draw blocks:
**[-0.000407713, +0.000230556]**; 26-draw blocks:
**[-0.000396232, +0.000226596]**. Both include zero. Historical exposure and
bootstrap dependence/stationarity assumptions further limit inference.

Uniform had 8 exact hit draws; that candidate had 4, each out of 345. These sparse
secondary counts do not demonstrate an advantage or justify choosing another seed.
Calibration diagnostics are pooled and cannot validate precise rare-code risks.

18 configurations attempted over three rounds. Twelve smoothed/shrinkage baselines
and one logistic configuration completed healthily; five logistic configurations
reported convergence warnings and were ineligible. Search stopped after three
completed consecutive rounds without qualifying gain. Boosting/ensembles were
therefore not searched; XGBoost received an actual four-head/three-seed correctness
training smoke only. LSTM, TabPFN and Chronos-2 were not trained/evaluated.

Raw source: experiments/damacai-v3/run-002, with baseline attempts reused from
run-001 after a NumPy-boolean round-review serialization failure. Failed output
and original predictions retained. User interruption stopped one outer attempt;
partial evidence retained and that attempt marked interrupted, not a complete fit.
Resumption did not change candidate grids or look at outer outcomes to tune.
A subsequent correctness fix propagates cached fit warnings across all targets
sharing that estimator; original run health records retain their original values.
This affects health reporting, not probability values or losses.

## Correctness, data and operational behavior

Fresh Python 3.12 and 3.11/pandas 3.1 reproduced <NA>-versus-None contract failure;
object-backed string-or-None normalization fixes it while preserving genuine 0000.
CLI missing-sys crash fixed; actual default legacy CLI completed. Optional legacy
ML training now requires --ml. All-slot hashes, saved full-support distributions,
strict prior-draw features, grouped labels, artifact integrity and visible fit
health replace unsupported probability/confidence claims in the primary path.

Current published ledger/log hashes remain identical to original evidence.
Pending Nov 4 v2 forecast remains unchanged; no probabilities reconstructed for it.
Nov 1 missed forecast remains historical coverage failure. New v3 records use
18:00 MYT publication deadline for official 19:00 draws, after-push confirmation,
and independent target records so unresolved older results cannot block later ones.
Special announcements require explicit official source metadata; the application
validates metadata, not the truth of an arbitrary announcement page's content.

69 tests passed on Python 3.12 and 3.13; constrained Python 3.12 tracker passed with
five optional UI/ML tests skipped. Actual UI AppTest passed in full environment.
Existing ARIMA/Streamlit warnings retained in logs. No TensorFlow training evidence.

Local ARM64 macOS tracker environment: warm/cold corpus preparation plus three
uniform forecasts ~0.09s, isolated tracker generation/artifact write ~1.08s,
process peak RSS ~87.5 MB. Tracker dependency install from warm package cache 0.66s;
regression suite ~3s. These are local measurements, not cold-network Linux CI
measurements. Hosted 15-minute job margin is configured but must be observed on its
first real run; no hosted job was triggered or deployment claimed.

## Continuation

Freeze settings in forecast_config.json before publication. Uniform champion;
best observed shrinkage candidate shadow; simplest digit baseline saved separately.
Development qualification is false, so this candidate cannot be promoted by a lucky
future hit sequence. First series starts no earlier than Nov 7, after existing v2
target. Require 1000 scheduled targets, not 1000 successfully selected outcomes;
missing/late publication prevents promotion. No interim aggregate comparisons.

Future probability gain and exact-hit gain have separate fixed simultaneous
contrasts. Newly declared models/procedures need newly reserved future data, not
retuning against this historical evaluation or recycling a failed lockbox.
Remaining hypotheses: more informative provenance/corrections, verified special
schedule ingestion, alternative categorical joint constructions only under a new
bounded protocol. Chronos real-valued forecasts cannot be rounded into code risks.

## Frozen current-consensus comparator

Same 345 development dates, original v2 top-three training/all-23 scoring: 4 hit draws, 4 slot hits; 176.32s total. ARIMA degraded/fallback on 38 dates. Probability metrics unavailable. Per-draw picks, actual training rows and method health saved in experiments/verification/consensus-baseline.jsonl. This baseline reproduction did not select or modify challengers.

## Frozen historical evaluation

All outer results remain exploratory because historical data had prior exposure.

| Outer year | Complete scored draws | Selected model | Marginal log loss | Hit draws |
|---|---:|---|---:|---:|
| 2020 | 126 | uniform | 9.210340372 | 4 |
| 2021 | 112 | uniform | 9.210340372 | 2 |
| 2022 | 179 | uniform | 9.210340372 | 4 |
| 2023 | 165 | uniform | 9.210340372 | 5 |

Already-inspected 2024–2026-11-01: 437 draws, uniform loss 9.210340372, 13 hit draws. No outer result changed the development selection. All outer scored folds completed; one interrupted inner attempt remained ineligible.

Measured physical model attempts: 1548.46s. Conservatively charging full interrupted allocation gives upper bound 1686.52s, below 7200s cap. This excludes frozen consensus baseline reproduction and software checks as declared.

## Release verification — 2026-11-04

Release preparation added a regression for the actual saved control-metric shape
at the 1000-target checkpoint and corrected its lookup. This changes no historical
probability forecasts or model selection. A read-only Linux Python 3.12 validation
workflow now checks both tracker and full environments before deployment. Original
experiment manifests and published ledger files remain preserved.
