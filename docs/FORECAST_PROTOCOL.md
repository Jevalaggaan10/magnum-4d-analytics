# Frozen Da Ma Cai forecasting protocol v3.0

Written before candidate comparisons on 2026-11-03. Machine-readable defaults are
in FORECAST_PROTOCOL.yaml; every run records its SHA-256, code revision and exact
configuration. Changes create a new protocol, never overwrite an evaluated run.

## Target and evidence

Ten distinct categorical ASCII codes 0000–9999, scored against all 23 slots.
Select by expected matched slots, not claimed probability of any hit. Positional
products assume within-code digit independence; pooling assumes exchangeable slots
within declared tiers. No independence across prize slots is asserted. Any-hit
probabilities are unavailable unless explicitly requested as an independence
approximation. Repeated suffix slots remain separate. Uniform has log loss
9.210340371976184 and sum-over-classes Brier 0.9999 per slot.

All existing history is exploratory, including nominal outer folds. Snapshot plus
ledger observations is frozen through 2026-11-01. Eligible retrospective targets
are complete recorded draws; missing/invalid slots, conflicting sources and
unrecorded regular dates remain in coverage/provenance reports. An absent regular
date is not proof that a draw happened. Current snapshot gaps are not filled.
Magnum, unverified Excel and yearly fixed-pick/top-three backtests stay separate.
Legacy consensus has no categorical probability metrics.

## Chronology and selection

Initial history: 2005–2017. Inner development validation: 2018 and 2019, expanding
history before each target. Fit transformations only on earlier target rows.
Compute calibration OOF forecasts in earlier years (2016–2017 for 2018); weights
for any year use only earlier OOF forecasts. All slots and digits of a draw stay
together. Preserve past context across boundaries. Evaluate one draw, then update.
Known target calendar features are allowed; outcome features use prior draws.

Outer years: 2020–2023. After development freezes the procedure, each outer year
selects using its previous two inner validation years only, under the same fixed
candidate procedure. Outer scores cannot choose models, grids or new hypotheses.
Already inspected 2024–2026-11-01 is a separately labeled retrospective of the
frozen development selection, not a new test. No retrospective is prospective.

Counts update each target. Learners refit at multiples of 32 observed complete
draws, anchored to the first corpus draw, using the previous 1024 target draws.
Feature construction uses the full prior prefix, including target-date calendar
windows and elapsed gaps. Fitted scalers use only that refit's training rows.
Report both parameter-fit cutoff and actual available-history cutoff. Calibration
weights are frozen before each evaluated year and before future confirmation.

Five rounds: 5 baseline configs (uniform, full-code total prior mass 10000/100000,
digit pseudocount 10/100); 7 shrinkage/decay/tier variants of the best nonuniform
baseline; 6 logistic configs (C .01/.1/1 × history/calendar); 6 XGBoost configs
(depth 1/2/3 × rounds 25/75, eta .05, lambda 10, alpha 1); 6 ensemble subsets
(uniform plus baseline, linear, boost, baseline+linear, baseline+boost, all three).
Absent families make their subsets ineligible, not substitute new configurations.
Round 2 changes one baseline assumption per variant: uniform weights .5/.9/.99;
calendar half-lives 365/1095 at uniform weight .9; five-tier hierarchical prior
strengths 1000/10000 at uniform weight .9. Tier prior mean is the earlier global
baseline. Logistic uses four multinomial heads. Boosting uses four ten-class
softprob heads, no negative sampling or class weighting. Features: smoothed
all-time/30/90-draw positional frequencies, known target tier, elapsed gap;
calendar extension adds weekday and annual sine/cosine. Missing digit classes
receive numerical uniform support. Stochastic seeds 17/29/43 are averaged.
Ensembles fit nonnegative simplex weights with .001 squared-distance penalty
toward uniform, using earlier OOF actual-slot probabilities, never outer outcomes.
At least 100 earlier OOF draws are required; otherwise uniform is used explicitly.

A round advances complexity only for development loss improvement >=.001 nats
per slot; compare paired draw uncertainty and prefer simpler within uncertainty.
Stop at five rounds, 30 attempted configs, 7200 fitting/evaluation seconds, or
three consecutive completed rounds without qualifying gain. Failed/time-limited
attempts count. No target subsampling to finish a slow candidate. Per-round caps
600/900/1500/1500/600 seconds; 2100 seconds reserved for frozen evaluation. Each
attempt includes hypothesis, falsification, configuration, seeds, errors and time.
No paid services, seed picking, k changes, outcome-driven exclusions or extra grids.

## Scores, controls and uncertainty

Primary: mean draw-level mean marginal categorical log loss over 23 slots. Brier:
sum squared error over 10000 classes, then mean slots and draws. Calibration bins
[0,.00005,.0001,.0002,.0005,.001,.01,1], digit marginal diagnostics and ten-pick
mass; draw-block intervals. Sparse per-code events cannot establish precise rare
probabilities. Secondary exact hit-draw, slot-hit, top-three and first-prize counts.
Report denominators, unique picks, failures, missing forecasts and data coverage.

Keep date-seeded random.Random(YYYYMMDD) ten-code control. Also use 1000 independent
local seeded controls per target, averaged rather than choose a lucky seed.
Independent-slot null: 1-(1-10/10000)^23. Conditional random-ticket control given u
distinct observed suffixes: 1-C(10000-u,10)/C(10000,10). These are different nulls.

Paired moving-block bootstrap, 10000 replicates, seed 17029, block length 13 draws;
26-draw sensitivity required, length 1 diagnostic only. This assumes sufficiently
short dependence and approximately stationary local loss differences; overlapping
training histories do not create independent replicates. Mark structural regime
changes and uncertain dependence as inference limitations. Report tier and year
slices descriptively, with no slice-selection promotion claims. Use Bonferroni
98.75% two-sided intervals for four fixed contrasts (two losses vs uniform and
simplest nonuniform baseline; hit rate vs uniform picks and conditional control),
providing simultaneous >=95% family coverage under bootstrap assumptions.

## Future confirmation and publication

Freeze one procedure, config, calibration, dataset lineage and code hash before
future confirmation. First 1000 scheduled targets after freeze constitute one
checkpoint, including late/missing forecasts. Never extend until 1000 successes.
Special targets require announcement URL and retrieval time recorded before their
deadline. No interim aggregate comparisons or promotion. A substantive model change
ends that confirmation series; start a newly declared future series, not a pooled
continuation. Correctness fixes retain version history and require impact review.

Uniform is initial probability champion. Model failures produce a visible uniform
fallback; complete-procedure metrics include it. Incomplete coverage/invalid
forecasts prevent promotion at the checkpoint. Practical gain >=.001 nats over
uniform and frozen simplest baseline, with paired intervals below zero under both
block lengths, plus all software/provenance/runtime gates, is required. Exact-hit
advantage needs its own two positive paired contrasts. A loss improvement is not
a financial return claim. Confirmation failure keeps uniform; lockbox not recycled.

Use Asia/Kuala_Lumpur; verified official regular draw time 19:00, engineering
publication deadline 18:00 (one-hour safety margin). Publication must finish before
deadline. No target/current outcomes allowed in forecasts. Preserve Nov 1 2026 miss
and pending Nov 4 v2 picks; never reconstruct prospective forecasts. CI confirms
publication time, logs missed coverage, and commits artifacts with ledger. Model
search and replay write separate experiment directories, never prospective ledger.

## Power and continuation

Estimate paired-loss minimum detectable effect using development variance and
predeclared block bootstrap. Hit power uses independent-uniform approximation and
conditional-control simulations, with assumptions stated. 1000 targets are roughly
six years of regular draws and may be insufficient for a small exact-hit edge.
Future results cannot exist at implementation time. Save unfinished hypotheses,
required future targets and access/compute limits. LSTM/TabPFN/Chronos are deferred;
any new categorical construction needs a new budget/protocol before testing.
