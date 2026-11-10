# Magnum 40-draw audit and research roadmap

Checked 9 November 2026. Repository code: `cbf189b`. Latest sampled draw: 8 November 2026. User-requested window: latest **40 complete Magnum 4D draws**, 15 August–8 November 2026, with **920 prize slots**.

## Verdict

No verified Magnum predictive advantage is established. The live tracker publishes **Da Ma Cai** forecasts; it has **zero eligible pre-draw Magnum forecasts** for this window. Prospective Magnum prediction accuracy is therefore **not available**, rather than 0%.

The current Magnum adapter reads the bundled `4D.txt`, whose latest date is 28 December 2025. A fresh interface or a successful software test does not make that feed current or establish forecast skill.

I ran an isolated, strict-prefix retrospective comparison on newly obtained Magnum results. Uniform reference matched two draws, the digit shadow one draw, the simpler digit baseline one draw, and legacy consensus one draw. All matches were in Special prizes; no method matched a first, second or third prize. These are exploratory replays, not forecasts actually published before those draws.

## Results and source confidence

[Lottochaos archive](https://lottochaos.com/en/magnum-4d/archive/) supplied 141 complete 2026 rows. All prize tiers for all 40 target draws agreed with corresponding [STARBUCKS/STAR4D archive](https://starbucks4d.com/4d-result/magnum/) pages. Top-three order was compared exactly; Special and Consolation numbers were compared as multisets because publication order can differ. Two archives may share upstream data, so agreement is corroboration, not independent certification.

Direct requests to Magnum's official website returned HTTP 403. Its [official draw-results page](https://www.magnum4d.my/results/draw-results) was readable through search, but result fields were empty dynamic placeholders. I could verify its mechanical draw description, not certify these 40 result boards directly. Treat the numerical audit as provisional secondary-source evidence.

| Method replayed with ten distinct picks per draw | Draws with any exact match | Matched prize slots | Draws with top-three match | Mean marginal log loss |
|---|---:|---:|---:|---:|
| Current uniform reference, `0000`–`0009` | 2/40, 5.0% | 2/920 | 0/40 | 9.210340372 |
| Digit shadow, pseudocount 100, 50% uniform mixture | 1/40, 2.5% | 1/920 | 0/40 | 9.210046946 |
| Digit baseline, pseudocount 100 | 1/40, 2.5% | 1/920 | 0/40 | 9.209828899 |
| Existing legacy descriptive consensus | 1/40, 2.5% | 1/920 | 0/40 | Unavailable: ranks are not probabilities |

Concrete matches:

- Uniform reference: `0005`, Special prize, 16 September; `0001`, Special prize, 27 October.
- Digit shadow and digit baseline: `1141`, Special prize, 16 August.
- Legacy consensus: `3845`, Special prize, 3 October.

A uniformly random ten-code ticket has conditional any-hit probability
`1 - C(10000-u,10)/C(10000,10)`, where `u` is that draw's number of distinct winning codes. This uses the observed board and does not require independence between its prize slots. Repeated codes were preserved: three sampled boards contain 22 distinct codes, the other 37 contain 23.

Across these boards, random tickets average **0.908 hit-draws per 40**. A separate 1,000-replicate control averaged 0.963, with an empirical central 95% range of **0–3 hit-draws**. The exact conditional random-ticket distribution assigns about **23.0%** probability to two or more hit-draws. This is a descriptive chance comparator, not a formal prospective significance test of models reconstructed after outcomes were known.

The shadow's mean loss gain is **0.000293426 nats/slot**, below the repository's 0.001 practical threshold. Its candidate-minus-uniform 98.75% moving-block intervals are:

- 13-draw blocks: **[-0.001275340, +0.000525935]**.
- 26-draw blocks: **[-0.000712897, +0.000313201]**.

Both include zero. With only 40 draws, long-block intervals are themselves fragile; this is diagnostic evidence, not confirmation. The simpler digit baseline also fails the practical threshold and both interval checks. Neither result justifies changing champion.

### What was actually tested

`compare_magnum.py` copied existing model settings before evaluating targets; it did not tune candidates on this window. Training combined the bundled Magnum history with earlier rows from the 2026 archive. Every target used only rows dated strictly earlier than itself, with explicit cutoff assertions. The seed remains an incomplete historical archive; unknown missing dates were not fabricated.

Digit configurations were selected previously for Da Ma Cai and transferred here only as hypotheses. They are not qualified Magnum procedures. The current forecast implementation was developed after many target dates, so strict row chronology alone cannot turn this into a historical deployment claim. Later pretrained foundation models can also contain public historical results, another reason future timestamped evaluation is necessary.

Files:

- `latest-40-magnum-draws.csv`: all 23 actual prize fields for each requested draw; preserve leading zeros when opening.
- `draw-comparison.csv`: each date's top-three actuals, all methods' ten picks and matched codes.
- `comparisons.jsonl`: full per-draw scores, training cutoffs, corpus hashes and picks.
- `source-crosschecks.json`, `source-manifest.json`: archive checks and captured-file hashes.
- `summary.json`, `null-distribution.json`: aggregate results and control calculations.
- `analysis-spec.json`: fixed methods, seeds, scope and source limitations.

No live ledger, model configuration, frozen source file or production feed was changed.

## Current primary research: what matters for this project

Survey cutoff: 9 November 2026. This is a focused review, not an exhaustive literature claim or independent reproduction of vendor benchmarks. Full-text access below means methods/results/limitations sections were inspected; it does not mean every appendix experiment was reproduced.

### 1. TabPFN-3.5 Technical Report — Jäger et al., v3, 3 November 2026

[Latest primary report](https://arxiv.org/html/2609.17895v3). Vendor preprint; full-text architecture, evaluation and availability sections inspected. Reports stronger tabular prediction and temporal/grouped benchmark results. Thinking mode uses additional tabular-model computation, not an LLM. Code and model weights have different licenses.

**Use here:** worthwhile optional categorical challenger, with ten-class digit heads or a normalized autoregressive construction. Freeze exact checkpoint and environment before future testing. Reported general-purpose rankings establish no Magnum advantage; more compute is not evidence of more lottery signal. Check applicable weight/use terms before adopting it.

### 2. Chronos-2: A foundation model for multivariate time series forecasting — Ansari et al., 30 October 2026

[Primary extended paper](https://arxiv.org/html/2610.41956v1). Full-text target formulation, architecture and limitations inspected. Handles multivariate real-valued series and covariates. Categorical variables are covariates, not supported target outcomes; its quantiles do not supply a coherent joint trajectory distribution.

**Use here:** low priority for raw four-digit outcomes. Treating `0037` and `0038` as nearby real values is the wrong task. Forecasting properly defined digit-frequency series could be an exploratory construction, but would require a defensible conversion to normalized code probabilities and proof of forward gain.

### 3. Beyond IID: How General Are Tabular Foundation Models, Really? — Purucker et al., v2, 30 October 2026

[Primary paper](https://arxiv.org/html/2606.30410v2). Full-text curation, split ablations and conclusion inspected. Uses 142 datasets and 11 models; strengths depend on data scale and grouping. Inappropriate split choices distort rankings. Traditional methods remain important comparators on large/grouped tasks.

**Use here:** keep each complete draw grouped in train/test, preserve chronological inner and outer evaluation, and compare foundation models with regularized trees. Twenty-three slots from one draw are not 23 independent validation draws. Benchmark success on other domains cannot substitute for the same Magnum task.

### 4. TabReliBench — Zhao et al., 9 October 2026

[Primary paper](https://arxiv.org/html/2610.13654v1). Full-text evaluation definitions, results and discussion inspected. Predictive rankings, proper scores and calibration rankings differ. Larger ensembles do not consistently repair calibration; marginal coverage does not establish conditional coverage.

**Use here:** retain full categorical log loss/Brier and digit/tier calibration alongside exact hits. Learn ensemble weights and calibrators from earlier out-of-fold predictions only. A sharp, confident forecast or a large model is not automatically reliable.

### 5. Agentic Forecasting with Structured Linguistic Beliefs — Kevin Murphy, v5, 27 September 2026

[Current primary paper](https://arxiv.org/html/2604.18576v5). Full-text tool design, baseline comparisons, leakage discussion and limitations inspected. Structured evidence memory, search and numerical tools help binary event forecasting. Authors distinguish Bayesian-inspired linguistic updates from exact Bayesian inference. Its evidence is retrospective and does not establish live lottery skill.

**Use here:** useful pattern for an evidence-audit assistant: record probability claims, source URLs, timestamps, opposing evidence and unresolved questions. Delegate numerical work to tested statistical tools. This paper's useful external event information has no demonstrated analogue for independently drawn lottery codes.

### 6. Ask a Language Model for Lottery Numbers — Żatuchin, 2026 preprint

[Primary note](https://arxiv.org/html/2610.00052v1). Methods, tables and limitations inspected. Repeated six-of-49 requests produced concentrated, recurring tickets rather than uniform coverage. It tests output diversity, not winning-number forecast performance; provider caching and differing tool configurations were not resolved.

**Use here:** an LLM should not serve as a random-number generator or be credited with validated picks merely because its responses vary. Metadata caveat: the arXiv page reports a 4 September submission despite an October-prefixed identifier; I have not resolved that inconsistency, so this note receives limited evidential weight.

### 7. LLM as Forecasting Planner — Nguyen et al., 27 July 2026

[Primary paper](https://arxiv.org/html/2607.24892v1). Full-text planning method, context relevance, results and cost discussion inspected. An LLM ranks candidate trajectories generated by a numerical foundation model; it does not create or revise values directly. Gains depend on useful contextual information and are bounded by candidate support.

**Use here:** a reasonable architecture for research planning and checking source/schedule context. No demonstrated reason that selecting lottery trajectories through language reasoning creates information about future balls. Its many agent calls make blind adoption an expensive hypothesis, not an obvious upgrade.

### 8. Experimental Limits of Machine Learning on Near-Random Data — Dinu, Modan and Frunzete, first online 30 June 2026

[Publisher preview](https://link.springer.com/chapter/10.1007/978-3-032-30518-3_23). Conference chapter; **abstract only**, full chapter subscription-restricted. The abstract reports no significant advantage for frequency, Markov, MLP and random forest methods on 2,510 Romanian 6/49 draws.

**Use here:** relevant negative evidence, with different game mechanics. I could not audit the full methodology or reproduce it. High empirical entropy and passing randomness tests do not by themselves prove independence or mathematical impossibility of prediction.

### Supporting evaluation literature

[Proper scoring rules — Waghmare and Ziegel, v4, 10 May 2026](https://arxiv.org/html/2504.01781v4): inspected categorical/log-score definitions and evaluation discussion; supports assessing distributions through proper losses. [ForecastBench — Karger et al., v5, February 2025](https://arxiv.org/abs/2409.19839): abstract/protocol rationale inspected; unresolved future questions avoid known-answer leakage. [Are Language Models Actually Useful for Time Series Forecasting? — 2024](https://arxiv.org/html/2406.16964v1): inspected ablations; removing an LLM need not reduce forecasting performance. These supporting sources are not all newly published papers.

None of the reviewed papers demonstrates a reproducible, prospective Magnum 4D prediction advantage. Papers about the neural-network “lottery ticket hypothesis” concern pruning, not lottery outcomes, and were excluded.

## Recommended improvement path

The mathematical boundary is conditional: if each next draw is independent and uniform given available information, history and extra model capacity cannot improve its true conditional probabilities. That assumption is testable only imperfectly; do not replace it with an unsupported claim that every real draw mechanism has been proven ideal. A useful research program looks for stable, available-before-deadline information and accepts a null result.

### Priority 1 — Establish a real Magnum experiment

Replace the static-only Magnum feed with a validated, provenance-aware result source. Seek an official export or supported feed; use corroborated archives as provisional inputs with explicit status. Capture draw dates/IDs, all prize tiers, leading zeros, source timestamps and corrections. Verify Magnum-specific scheduling and a publication deadline rather than copying Da Ma Cai assumptions automatically.

Create an independent Magnum scientific series with a frozen procedure and pre-draw publication evidence. Keep the ongoing Da Ma Cai series separate. The 40 draws inspected here are now exploratory data and cannot be relabeled an untouched test set.

### Priority 2 — Test for usable structure before increasing complexity

Audit chronology, missing draws, duplicate handling and changes in game/collection conventions. Study positional digits and tiers with strong shrinkage toward uniform, and test whether findings survive later periods and negative controls. Diagnose collection/publishing-order artifacts separately from physical draw effects.

If available, evaluate machine/ball-set or maintenance metadata only when genuinely observable before forecast publication. Later video details or result-time metadata may explain outcomes retrospectively but cannot be used as pre-draw features. High historical entropy is not a substitute for conditional, forward prediction tests.

### Priority 3 — A bounded categorical challenger ladder

Start with uniform; hierarchical Bayesian digit/tier models; the repository's existing categorical regularized logistic/boosting path; then one CatBoost/LightGBM-style challenger if appropriate. Avoid the old historical appeared-versus-never XGBoost task as a next-draw probability estimator.

A potentially useful structural extension is an autoregressive categorical code model:

`P(d1,d2,d3,d4 | x) = P(d1|x) P(d2|d1,x) P(d3|d1,d2,x) P(d4|d1,d2,d3,x)`.

This permits within-code digit interactions while preserving a normalized distribution over all 10,000 codes. It still requires enough data, strong shrinkage and successful forward evaluation. Do not assume arbitrary sorted Special-prize positions are physical draw order.

Only then allocate one controlled TabPFN-3.5 experiment, with exact checkpoint, license/access, parameters, feature schema and compute recorded. Direct integer-code regression and rounding are poor substitutes for categorical probabilities. Do not fine-tune a general LLM merely because it is larger.

### Priority 4 — Calibrated combination and a conservative decision layer

Fit probability calibration and nonnegative ensemble weights on strictly earlier out-of-fold forecasts. Compare every component and ablation with uniform on identical eligible dates. Uncertainty, health and runtime count; no method should win by excluding failed draws.

A useful decision model chooses among **qualified challenger**, **uniform reference**, and **abstain/block** when source integrity, freshness or timing fails. For a tree-based decision model, evaluate categorical gradient-boosted heads as above. Reinforcement learning on sparse lottery hits is low priority unless there is an independently demonstrated state-dependent signal and a valid reward/evaluation design.

### Priority 5 — Give an LLM evidence work

Use an LLM to extract official schedule announcements into a validated schema, identify source discrepancies, maintain a cited research register and propose falsifiable hypotheses. Require deterministic validation and dated evidence before these outputs affect records. Keep outcome scoring and probability arithmetic in tested code.

An optional LLM challenger must be blind to target outcomes, with prompt, model/checkpoint, retrieval cutoff, seeds and failures logged. Evaluate it against a no-LLM tool pipeline. New weights may already know old public draw results; fresh prospective targets are the cleanest safeguard.

### Priority 6 — Freeze selection and demand future evidence

Use the existing practical loss threshold and paired 13/26-draw uncertainty checks for a newly preregistered Magnum procedure. Separate marginal-loss improvement from exact-hit advantage. Do not promote a candidate because it wins this 40-draw replay, and do not silently expand the original Da Ma Cai search after its stopping rule.

An illustrative calculation using this window's random-ticket rate shows why 40 is weak: at a one-sided 1.25% test level, five hit-draws would be needed to reject a constant-rate null; even doubling that rate gives only about **3.4% power** under a binomial approximation. This is not a sample-size prescription for every model, but it rules out treating a handful of matches as reliable validation. Use a preregistered power calculation and genuinely future evidence; small gains may take substantial time to resolve.

Suggested first engineering deliverable: a fresh Magnum feed plus immutable prospective scoreboard. Suggested first model experiment: a strongly regularized categorical baseline with a uniform mixture. Optional second model experiment: TabPFN-3.5 under the same frozen task. The highest-value LLM role is an auditable research/data assistant until a blinded trial proves additional forecasting value.

No model weights were downloaded, no paid prediction API was invoked, no production procedure was replaced, and no new betting recommendation is supported by this audit.
