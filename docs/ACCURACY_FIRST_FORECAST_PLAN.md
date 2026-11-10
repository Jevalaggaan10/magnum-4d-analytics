# Accuracy-first lottery forecasting: comprehensive audit and execution plan

Drafted 10 November 2026. Status: **plan only; implementation and new model qualification remain pending**. Scope: Da Ma Cai live forecasting, a separate Magnum research track, all previously reported model/data/evaluation issues, and the November research review.

## 1. Recommendation and achievable result

Make the primary research objective **the probability that one fixed set of ten distinct four-digit codes matches at least one of the next draw's 23 prize slots**. Keep first-prize and top-three accuracy as separately reported outcomes. Fix operator, target, pick count and publication deadline before comparing methods.

The strongest route is better evidence and correctly formulated categorical models: verified operator-specific inputs, strict pre-draw features, regularized probability estimates, then a decision rule aligned with any-hit accuracy. Try a tabular foundation model only as a controlled challenger. Use an LLM first for cited research and source checking; no reviewed paper establishes that an LLM can predict Malaysian lottery outcomes.

**There is no supported way to ensure that the next draw contains even one predicted code.** If outcomes are independent and uniform conditional on all information available before the deadline, a more complex model cannot improve their true probabilities. Real-world departures remain research hypotheses; passing randomness tests does not prove independence, and finding historical patterns does not establish a predictive departure.

Ten picks also cannot cover a board containing more than ten distinct winning codes. Predicting every prize requires a separately defined full-board task; it is not achieved by matching one code, reporting near misses, or increasing the number of picks after seeing results. A correct hit in one future draw would be a successful prediction for that draw, but would not by itself establish repeatable model skill.

This plan optimizes exact-hit accuracy. Payout optimization, EV simulation, visual redesign and adding more descriptive ranking lenses are outside its implementation priorities.

## 2. Why GitHub says Magnum while the live tracker uses Da Ma Cai

The repository began as `magnum-4d-analytics`. Its README still starts with **“Magnum 4D Analytics — Honest Edition”** and describes the Magnum historical dataset. The current GitHub About description already mentions **both Magnum and Da Ma Cai**, but does not clearly identify which operator supplies the live experiment. The live procedure in `forecast_identity.py`, configuration and tracker is Da Ma Cai-only. The Magnum adapter still reads the bundled historical TXT file.

This is documentation and product-identity drift. There was also a real, earlier cross-operator training error, documented in the 28 October audit correction. That historical error must remain visible; a rename must not imply that old forecasts were produced by the corrected procedure.

Proposed future README/product title:

> Malaysian 4D Analytics — Da Ma Cai Live, Magnum History

Proposed GitHub About description:

> Malaysian 4D research: live Da Ma Cai probability forecasts with pre-draw audit logs; Magnum historical analytics. No demonstrated prediction advantage.

Keep the existing repository URL for now. Changing its slug is unnecessary for accuracy. When implementing the naming correction, inspect README, app headings, tracker log labels, CLI help and workflow descriptions together. Every prediction must show operator, game, draw date, procedure version, training cutoff and pick count. Historical reports retain their original operator and scope.

## 3. Evidence baseline: what is established and what remains open

| Item | Established evidence | Consequence for this plan |
|---|---|---|
| Current live operator | Da Ma Cai; champion uniform; digit shadow with pseudocount 100 and 50% uniform mixture | Continue the frozen series; no evidence-based champion upgrade is available today |
| Scientific procedure | `v3-fixed-1000`, ten picks, all 23 slots; current primary metric is marginal log loss and selector objective is expected matched slots | A new any-hit-primary experiment needs its own preregistered identity |
| Existing development | No qualifying practical improvement; `development_qualified` and `correctness_verified` remain false | Passing software checks must not be relabeled model qualification |
| Latest requested Magnum study | Forty complete draws, 15 August–8 November 2026; 920 prize slots | Separate retrospective research, not Da Ma Cai live evidence |
| Magnum prospective coverage | No eligible pre-draw Magnum forecasts in that window | Prospective Magnum accuracy is unavailable, not zero |
| Magnum replay | No predictive advantage established; inspected digit-model loss gains fail the existing threshold/uncertainty checks | Do not select a new champion from this inspected window |
| Magnum result provenance | All 40 boards corroborated by two secondary archives; direct official board certification unavailable | Preserve provisional source status and possible shared upstream dependence |
| Magnum freshness | `4D.txt`/current adapter ends on 28 December 2025 | A fresh Magnum screen does not constitute a live feed |
| Repairs released on 9 November | Identity isolation, frozen-source enforcement, overdue-result recovery, development uncertainty selection, special-schedule persistence | Preserve these fixes; audit their integration before new experiments |
| Release verification | Historical local 100-test run and successful hosted full/tracker validation are recorded | Software evidence, not evidence of winning-number accuracy; rerun appropriate checks when implementation changes |
| Documentation consistency | Earlier continuation/results notes still describe hosted verification as pending; later repair evidence records it | Reconcile dated evidence rather than silently setting qualification flags |
| Working copy at planning time | Local `main` at `cbf189b`; `origin/main` at `0153382`, one automation commit ahead; existing research directory untracked | Inspect/preserve both before future implementation; this draft does not merge or push |

The remote-only commit adds an integrity-status field to `predictions.json`; inspected published target metadata is unchanged. The 11 November Da Ma Cai forecast is already marked published, with publication timestamp **8 November 2026, 22:39:41 MYT**. Preserve its picks, vectors, training cutoff, original publication evidence and identity.

Evidence to carry forward:

- [Latest Magnum audit, methods, matches, source limits and paper review](../experiments/research/20261109-magnum/MAGNUM_40_DRAW_RESEARCH.md).
- [Model/data audit and older research findings](MODEL_AUDIT.md), [research-to-change register](FORECAST_RESEARCH.md), and [frozen development results](FORECAST_RESULTS.md).
- [Completed repair plan](FORECAST_REPAIR_PLAN.md), [repair impact and verification](FORECAST_REPAIR_IMPACT.md), and [continuation constraints](FORECAST_CONTINUATION.md).
- Existing [scientific protocol](FORECAST_PROTOCOL.md), [machine-readable protocol](FORECAST_PROTOCOL.yaml), [development selection policy](FORECAST_SELECTION_POLICY.json), and [hosted repair validation](https://github.com/Jevalaggaan10/magnum-4d-analytics/actions/runs/42612674111).

The research directory currently contains local evidence that is not committed. Preserve raw captures, manifests, scripts and checks when making it durable; do not link a public report to missing local-only artifacts.

## 4. Define accuracy before selecting another model

For target draw `t`, publish exactly ten distinct codes `S_t`, each an ASCII string from `0000` through `9999`. Let `W_t` contain all 23 actual prize slots, preserving repeated codes.

Primary outcome:

`H_t = 1 if any published code appears in any actual prize slot; otherwise 0`.

Primary reporting unit is a **complete draw**, not a prize slot or a digit. Report `sum(H_t) / scheduled eligible targets`, with publication and result coverage displayed alongside it. Define missingness before launch:

- A missed, late or invalid publication counts as no successful forecast in the operational scheduled-target denominator. It must not disappear from model comparisons.
- An unresolved actual result remains unresolved. Show the unresolved count and possible accuracy bounds; do not invent an observed miss or promote while required outcomes are missing.
- A conditional-on-valid-publication hit rate may be shown descriptively, but cannot replace the scheduled-target result.

Secondary outcomes: matched prize slots, distinct winning codes covered, top-three hit draws, first-prize hit draws, and precision/recall with their explicit denominators. A repeated code can match two slots while still producing only one hit-draw. Do not count digit matches, permutations, reversed codes or last-three matches as exact four-digit hits.

Use full categorical log loss, Brier score and calibration as supporting evidence about probability quality. They are useful diagnostics and can expose weaknesses earlier than sparse hits. They cannot substitute for the requested exact-hit result. Avoid binary classification “accuracy” dominated by the thousands of codes that lose.

### Reference probability and the limit of a next-draw promise

Under the illustrative independent-uniform four-digit-slot model, ten distinct picks have any-hit probability `1 - (1 - 10/10000)^23`, about **2.27%**. This is a mechanism approximation, not a validated probability for a particular forecast or a guarantee of one hit every 44 draws.

For a completed board containing `u` distinct codes, an independently sampled uniform ten-code control has exact conditional probability `1 - C(10000-u,10)/C(10000,10)`. This describes the random-ticket comparator conditional on that board; it does not certify a retrospectively fitted model.

Da Ma Cai's official procedure uses four-digit suffixes of six-digit outcomes, so duplicates are legitimate. Its horse-race mechanism assigns main-prize tiers among drawn codes; it does not establish that publicly available race context predicts the set of all 23 suffixes. Verify actual information availability before considering any such feature. Sources: [official draw procedure and regular schedule](https://www.damacai.com.my/general-information/), [official 1+3D target definition](https://www.damacai.com.my/how-to-play).

### Correct the objective mismatch explicitly

`forecasting.py::_result` ranks codes by the sum of their marginal slot probabilities. This is appropriate for maximizing **expected matched slots**. It does not generally maximize **probability of at least one matched slot** under dependence or heterogeneous slot distributions. Under identical independent slot distributions, the objectives select the same top codes; a mismatch is not proof that today's uniform forecast is wrong.

Audit whether the proposed model supplies enough information to optimize any-hit probability. Marginal probabilities alone generally do not identify the joint event. Start with a declared independence approximation and evaluate it; only add a joint model when data and mechanism justify it. A simulator-based coverage optimizer maximizes its assumed model's probability, not actual lottery certainty.

Do not change `selection_objective`, primary metric, model configurations or pick budget inside the existing `v3-fixed-1000` series. The new objective requires a new procedure, separate artifacts and separately reserved future targets.

## 5. Comprehensive audit register

Create one issue/evidence register during implementation. Each entry records status, affected paths, reproduction or source, severity, proposed fix, validation, and whether scientific identity changes. “Previously fixed” requires preserved evidence; “not reproduced” is not “verified absent.”

| ID / priority | Audit and paths | Required evidence / acceptance |
|---|---|---|
| A01 / P0 | Operator and experiment routing: `data_sources/`, `data_loader.py`, `forecast_identity.py`, tracker, UI, README/About | A Da Ma Cai target cannot consume Magnum rows, wrong-operator results, local caches or another series' forecast; all displays name the operator |
| A02 / P0 | Coverage and freshness: `4D.txt`, `data/damacai.csv`, `data/provenance.json`, observed ledger | Inventory gaps, incomplete rows, latest complete date and source availability; never fabricate missing draws or treat stale data as current |
| A03 / P0 | Provenance and correction history: adapters, observation acceptance, source captures | Reproduce captured hashes; retain requested/returned date and draw ID; quarantine conflicts; record retrieval/correction times as unknown when unavailable |
| A04 / P0 | Schema and prize semantics: normalization, suffix extraction, tier mapping | Preserve leading zeros and legitimate repeated codes; reject wrong dates/operators/incomplete boards; compare unordered published tiers as multisets when appropriate |
| A05 / P0 | Calendar/deadline: `probability_tracker.py`, CLI, workflow | Check official regular schedule and each special announcement's actual content; timezone-aware generation and after-push evidence precede deadline; metadata-only validation is insufficient proof of announcement truth |
| A06 / P0 | Labels and leakage: `predictor_suite.py`, `train_model.py`, legacy CLIs, new categorical path | Separate historical appeared-versus-never XGBoost from next-draw learning; rebuild historical examples only from earlier observations; no full-corpus target-derived features |
| A07 / P0 | Windowing, preprocessing and cache identity: LSTM preparation, feature builders, training caches | No shared target observations across train/validation; preprocessing and calibration fitted only on earlier folds; cache/resume keys include corpus, cutoff, feature, code, environment, policy and fold identity |
| A08 / P0 | Objective and denominators: `forecasting.py`, evaluator, backtests, log/UI | Distinguish any-hit, expected slots, first/top-three and marginal loss; ten-code budget shared by all comparisons; no post-result union of model picks |
| A09 / P0 | Published evidence: `predictions.json`, `PREDICTION_LOG.md`, workflow commits | Reproduce each eligible saved vector and exact pick set; verify publication push evidence; local file creation or a replay timestamp cannot establish prospective publication |
| A10 / P0 | Identity/source guards: `forecast_identity.py`, compatibility manifest, configuration | Recheck isolation and approved execution fingerprints; reject mixed identities and unapproved bytes; operational flags cannot manufacture scientific compatibility |
| A11 / P0 | Outcome recovery: tracker and `data_sources/damacai.py` | Preserve bounded, fair overdue recovery; test wrong-date responses, old targets beyond latest five, outages, retries, idempotence and no requests for future targets |
| A12 / P1 | Legacy ranking claims: consensus, Markov, gaps, permutations, ARIMA | Ranks are not calibrated probabilities; “overdue” numbers are hypotheses, not a law; family-size and correlated-method agreement cannot inflate evidence; retain fit/fallback warnings |
| A13 / P1 | Probability contracts and fit health: `forecasting.py`, artifacts | All 10,000 classes normalized with full support; stable tie-breaking; seed/environment reproducibility; failed/degraded fits remain ineligible rather than silently winning |
| A14 / P1 | Splits and historical exposure: evaluator and experiment manifests | Complete-draw grouping and nested chronological selection; all previously inspected histories labeled exploratory; “untouched” means genuinely uninspected future evidence |
| A15 / P1 | Model search and selection: runner, selection policy, attempt logs | Record every family/window/seed/failed run; apply declared uncertainty/simplicity rule; no best-seed selection, stopping on a lucky hit, or reuse after policy changes |
| A16 / P1 | Nulls, controls and uncertainty: `forecast_evaluation.py`, `randomness_tests.py` | Match mechanism, horizon and pick count; evaluate the entire search on synthetic null data; group uncertainty by draw; adjust multiple tests; do not treat 1,000 simulated controls as 1,000 real draws |
| A17 / P1 | Calibration and combinations | Earlier out-of-fold fitting only; report coarse digit/tier reliability with sample sizes; no precise rare-code calibration claim from a handful of outcomes; ensemble ablations expose duplicated signal |
| A18 / P1 | Runtime/dependencies and failure accounting: requirements, workflows | Clean pinned environment, source/hash checks, completion before deadline and visible failure modes; no excluding slow/failed dates to improve accuracy |
| A19 / P1 | Artifact portability and reproducibility: saved NPZ/manifests, experiment stores | Distinguish tracked live artifacts from ignored local research stores; copy/hash required artifacts and verify reload on a clean machine; a path in a manifest is insufficient |
| A20 / P1 | Protocol/state/document consistency: configuration, continuation/results/repair docs | Reconcile dated hosted evidence and stale notes; keep software correctness, deployment evidence, development skill and prospective skill as distinct statuses |
| A21 / P1 | LLM/foundation contamination and source trust | Freeze checkpoint/prompt/retrieval cutoff; prevent target-result retrieval; validate extracted official announcements deterministically; future testing addresses unknown pretraining exposure |
| A22 / P2 | Research validity and contrary claims | Verify versions, full-text access and relevance; record paywalls and anomalous metadata; distinguish Malaysian 4D from six-of-49 or Kerala book-ending tasks |
| A23 / P2 | Simulator and unrelated metrics: `ev_simulator.py`, old annual backtests | Keep Magnum payout/target assumptions explicitly separate; top-three annual hit rates cannot become all-23 live accuracy; do not spend accuracy work on cosmetic financial outputs |

P0 means required before trusting a new live forecast or changing a scientific procedure. P1 means required before accepting a candidate's accuracy claim. P2 means document or isolate misleading ancillary claims; escalate if the primary forecast depends on them.

## 6. Research findings translated into experiments

Survey basis: the 9 November report, refreshed official schedule and current TabPFN/Beyond IID metadata on 10 November. This is a focused review, not an exhaustive latest-paper claim. Proposed lottery applications below are engineering hypotheses, not findings established by the cited papers.

| Research / access limit | What transfers | Planned experiment and rejection condition |
|---|---|---|
| [TabPFN-3.5, v3, 3 November 2026](https://arxiv.org/html/2609.17895v3); vendor preprint; relevant full-text sections reviewed | A strong tabular challenger is plausible; its extra computation is not an LLM, and code/weight licenses differ | Freeze one legally usable checkpoint and categorical construction; reject adoption without aligned forward gain beyond simple baselines |
| [Beyond IID, v2, 30 October 2026](https://arxiv.org/html/2606.30410v2); methods/split/results sections reviewed | Benchmark performance depends on temporal/group structure and scale | Same draw-grouped chronological folds for traditional/foundation models; reject random row splits and benchmark-rank arguments |
| [Chronos-2 extended paper, 30 October 2026](https://arxiv.org/html/2610.41956v1); formulation/limitations reviewed | Real-valued multivariate forecasting is not a categorical four-digit target | Low priority; never regress and round raw codes. Any frequency-series construction must yield valid normalized code risks and survive forward testing |
| [TabReliBench, 9 October 2026](https://arxiv.org/html/2610.13654v1); evaluation/results reviewed | Predictive ranking and calibration need separate checks; bigger ensembles are not automatically reliable | Calibrate on earlier out-of-fold probabilities; compare component/ensemble hit rate and proper scores on identical dates |
| [Agentic Forecasting with Structured Linguistic Beliefs, v5, 27 September 2026](https://arxiv.org/html/2604.18576v5); tools/baselines/leakage/limits reviewed | Structured cited evidence plus numerical tools can support audits | Evaluate source extraction/error detection against deterministic tooling; reject any claim that retrospective binary-event forecasting proves lottery skill |
| [LLM as Forecasting Planner, 27 July 2026](https://arxiv.org/html/2607.24892v1); method/context/limits reviewed | Language models can rank numerical candidates when context is useful | Optional, lower-priority blinded ablation; require real pre-deadline context and improvement over the same tool pipeline without an LLM |
| [Ask a Language Model for Lottery Numbers, 2026 preprint](https://arxiv.org/html/2610.00052v1); short paper reviewed | Repeated LLM tickets can concentrate; this is an output-diversity study, not a winning-number test | Never use conversational output as a uniform RNG or assume varied answers are independent forecasts; low evidential weight because metadata inconsistency remains unresolved |
| [Romanian near-random lottery study, first online 30 June 2026](https://link.springer.com/chapter/10.1007/978-3-032-30518-3_23); abstract only, chapter paywalled | Negative evidence for tested methods in a different game | Retain null baselines and accept failure to find signal; full-method replication remains unavailable, so do not generalize it into an impossibility proof |
| [Kerala near-random systems self-publication, May 2026](https://blog.iamnk.com/wp-content/uploads/2026/05/Lucky_Keralam_Research_Paper.pdf); earlier PDF review, no independent archive replication | A contrary claim deserves examination; book-ending target and many trials differ from Malaysian 4D | Replicate timestamped inputs and complete-draw dependence/multiplicity treatment before transfer; reject imported headline accuracy as evidence here |
| [Proper scoring rules, v4, May 2026](https://arxiv.org/html/2504.01781v4), [selection bias, Cawley/Talbot](https://www.jmlr.org/papers/v11/cawley10a.html), [ForecastBench protocol](https://arxiv.org/abs/2409.19839), [confidence sequences](https://arxiv.org/abs/1810.08240) | Honest probabilities, explicit model selection, unresolved future targets and valid monitoring | Retain proper-score diagnostics; predeclare the exact-hit primary outcome and checkpoint. A sequential design requires a new validated protocol, not repeated ordinary tests |

The earlier [LLM time-series ablation study](https://arxiv.org/html/2406.16964v1) also supports a no-LLM baseline. Network-pruning “lottery ticket” papers and lottery-sales forecasting are unrelated to predicting winning codes.

## 7. Model and decision priorities

Run this ladder only in a newly declared research procedure, separate from the current confirmation series. Reuse working categorical infrastructure before adding libraries.

1. **Uniform reference and strong shrinkage.** Reproduce current behavior; distinguish arbitrary lexical tie-breaking (`0000`–`0009`) from a claim that these codes are more likely. Compare fixed, predeclared shrinkage strengths and shared versus tier-specific parameters. Avoid selecting arbitrary tie seeds because they happened to hit history.
2. **Hierarchical categorical baselines.** Test partial pooling across digit positions and prize tiers. Study stable changes in digit bias and historical regimes, with strict prior-only rolling features and uniform mixtures. Avoid many unrecorded “hot/cold” windows.
3. **Regularized logistic and shallow categorical boosting.** Use next-target labels and the existing probability path. Audit convergence and compare one implementation before adding CatBoost/LightGBM. A new tree library cannot repair leaked labels in legacy XGBoost.
4. **Within-code dependence.** Test an autoregressive categorical factorization: `P(d1|x) P(d2|d1,x) P(d3|d1,d2,x) P(d4|d1,d2,d3,x)`. Enumerating all 10,000 codes produces a normalized distribution. Strong regularization and ablation must justify extra interactions; digit values are categories, not code magnitudes.
5. **One bounded TabPFN challenger.** Verify class support, preprocessing, weight terms and practical inference; use ten-class digit heads or the declared autoregressive construction. Freeze checkpoint/hash, availability date, seeds and environment. No claim that new weights were blind to old public results.
6. **Calibrated ensemble only after evidence.** Fit nonnegative weights and calibration on earlier out-of-fold predictions; compare against the best simpler component. Correlated models do not supply independent confirmations. Discard an ensemble that only adds confidence without exact-hit benefit.
7. **Any-hit selection layer, only if justified.** First evaluate the simple declared slot model. If genuine, validated dependence changes decisions, sample coherent full boards respecting operator rules, estimate `P(S intersects W)`, and compare greedy coverage/bounded swap selection with marginal top-ten. Use fixed seeds, held-out simulations and exact enumeration on small toy spaces to verify the optimizer. No optimizer is needed to distinguish uniform codes.

For physical/operational covariates, investigate source availability before model work: machine/ball-set identifiers, documented procedure changes, maintenance or venue. Include only information actually public before the publication deadline. Post-draw video details, result-time identifiers and later corrections cannot become historical pre-draw features without evidence of earlier availability. Sorting in published Starter/Special lists is not physical draw order.

Defer raw-code ARIMA/Chronos regression, general LLM fine-tuning, deep LSTM expansion, reinforcement learning on sparse hit rewards, forced model disagreement and large unconstrained searches. Reconsider only when a falsifiable pre-deadline signal and a stronger baseline comparison justify them. More computation cannot manufacture absent information.

An LLM research assistant may draft hypotheses, extract official announcements into a schema, flag conflicting sources and maintain citations. Tested code handles dates, source validation, probability arithmetic and outcome scoring. An LLM number-picker, if later tested, is a separate blinded challenger with frozen prompt/model/tool access; count all failures and compare it with identical tools without the LLM.

## 8. Phase-by-phase implementation and audit gates

Effort estimates below describe engineering work, not time needed to prove a rare-event accuracy advantage. Phases may conclude with **no qualified model**. That is a valid result.

### Phase 0 — Preserve baseline and establish one current state

Scope: A01, A09, A10, A19, A20. Estimated effort: half a day.

- Inspect the local/remote diff and untracked research before synchronizing; preserve original published files and research captures.
- Record repository SHA, dependency environment, configuration, protocol, compatibility manifest, source hashes, live/research artifact hashes and official source checks.
- Inventory all legacy, current and proposed series separately. Record published targets and unresolved outcomes without producing interim aggregate comparisons for the frozen series.
- Reconcile dated documentation with hosted evidence; explain each gate separately. Do not flip scientific qualification or frozen hashes to make checks green.

Deliverables: baseline manifest and audit register in a new dated experiment directory. Acceptance: no original publication/artifact changes, approved execution check passes or a specific blocker is recorded, and every future edit has an identity-impact decision. Stop model work if source/artifact integrity fails.

### Phase 1 — Fix naming, task specification and source contract

Scope: A01–A05, A08. Estimated effort: one to two days, excluding unavailable upstream data.

- Apply the proposed operator labels and About/README text when implementation is requested. Audit all callers and user-visible target labels.
- Write a new accuracy protocol specifying Da Ma Cai 1+3D, all 23 slots, ten distinct exact codes, any-hit primary outcome, missingness and publication rules. Keep the current protocol unchanged.
- Inventory Da Ma Cai source gaps and corrections; validate complete boards and date-specific responses. Preserve unknown provenance honestly.
- For a future Magnum track, implement a fresh, provenance-aware adapter before describing it as live. Prefer an official supported source; provisional corroborated archives must remain visibly provisional. Do not postpone Da Ma Cai checks while waiting for Magnum access.

Deliverables: operator/task contract, source audit, new protocol draft. Acceptance: wrong operator/date, malformed codes and stale/incomplete inputs fail closed; duplicates and leading zeros survive round-trip; each forecast identifies its correct game. An official source being unavailable is an unresolved dependency, not a passed source audit.

### Phase 2 — Close leakage and scoring gaps

Scope: A06–A08, A12–A14, A17. Estimated effort: two to four days.

- Trace raw data through labels, features, caching, training, calibration, selection and reporting.
- Isolate descriptive/legacy estimators from the qualified probability path; fix shared functions only where necessary, retaining explicit legacy labels for preserved outputs.
- Verify strict earlier-draw training, draw-grouped folds, prior-only preprocessing and time-aware metadata availability.
- Implement independently checked any-hit scoring and denominator reporting for the new research path. Unit examples must cover duplicate winning codes, missing publication, unresolved outcomes, wrong operator, leading zeros and no near-match credit.
- Reproduce valid saved forecasts; fail on corrupted arrays/manifests or misleading fallback health.

Deliverables: leakage map, targeted regressions, scoring/feature contracts. Acceptance: feature invariance under changes to future outcomes, no cross-draw split contamination, exact independent score recount and reproducible artifacts. Use test-first regressions for substantive bugs; do not write tests that merely repeat implementation arithmetic.

### Phase 3 — Preregister experiment selection and power

Scope: A14–A16, A20–A22. Estimated effort: one to two days.

- Declare initial candidate families, feature schema, training/refit schedule, fixed seeds, search budget, failure policy and selection rule before the new run. Initial proposal: at most 30 complete configurations across the ladder; any expansion is a separately logged experiment with new reserved evidence.
- Treat the existing 40 Magnum draws, prior development/outer folds and all inspected histories as exploratory. A new date split does not undo human or pretrained-model exposure.
- Use nested chronological, complete-draw folds for development. Choose calibration and ensemble weights from earlier out-of-fold predictions only. Never inspect outer results to revise an inner search.
- Use any-hit accuracy as the new procedure's primary selection/confirmation objective. Proper scores support diagnosis and a predeclared material-degradation guard, not a substitute success criterion. Keep the old procedure's `.001` loss gate unchanged.
- Freeze a multiplicity plan. Proposed maximum family-wise error budget: 5% across at most four declared contrasts, with 1.25% allocated to each; unused comparisons cannot become post-hoc extra tries.
- Calculate required future targets using paired candidate/control simulations under the operator-specific null and plausible dependence. Show 20%, 50% and 100% relative-lift scenarios, targeting at least 80% power for the declared planning alternative. A significant positive effect need not reach 50%; planning alternatives are not fabricated minimum real effects.
- Freeze the new checkpoint only after that calculation. Do not copy the current 1,000-target number without checking power. Rare-event confirmation may require years of scheduled draws. Forty draws cannot resolve small gains reliably.
- Predeclare one primary candidate and comparator, or an explicitly adjusted set. Include paired draw-level uncertainty and sensitivity to dependence; retain 13/26-draw block diagnostics where appropriate, while acknowledging fragile intervals for short samples. Validate the proposed method's null behavior rather than assuming a bootstrap is exact.

Deliverables: new protocol/selection manifest, power report and research register. Acceptance: parameters and stopping rules fixed before new evaluation; all outcome/feature availability assumptions documented; every proposed statistical claim has a calibrated error-rate interpretation.

### Phase 4 — Test for signal, then run the bounded challenger ladder

Scope: A12–A18, A21. Estimated effort: several days of controlled work; compute bounded by the new manifest.

- Reproduce uniform and simple shrinkage baselines on identical dates.
- Run the minimum complete candidate ladder above. Report all failed, timed-out and rejected attempts with actual reasons. Reuse existing correct code; isolate new model environments rather than destabilizing the tracker.
- Test the entire feature/search/selection pipeline on operator-appropriate synthetic null data. Under a null, complexity must not repeatedly produce false “winners.”
- Inject known small synthetic signals to show that the pipeline can detect available information without leakage; report power and failure cases. These are software/statistical checks, not real lottery evidence.
- Run dependence-preserving or explicitly qualified permutation/negative controls, feature ablations and regime sensitivity. A date-shuffle that destroys the structure being tested is not a universal null.
- For a joint decision rule, validate probability calculations against small exhaustive examples, independent simulation draws and an unchanged marginal-top-ten baseline. Report optimization uncertainty separately from estimated model uncertainty.

Deliverables: per-draw predictions, attempt/health logs, complete search manifest, null/injected-signal checks and a candidate decision report. Acceptance: candidate meets the frozen development rule on aligned dates, all integrity checks pass, and any apparent gain survives the registered comparisons. Otherwise retain uniform; do not restart a search until it finds a hit.

### Phase 5 — Operational qualification and prospective publication

Scope: A05, A09–A11, A18–A21. Estimated effort: one to two engineering days plus future observation.

- Assign a distinct scientific identity to the new accuracy experiment. Preserve the existing series and its 1,000 scheduled-target checkpoint without pooling or replacing records.
- In isolated paths, run full artifact generation, deterministic validation, deadline checks, upload/push confirmation and subsequent scoring. Clean-machine restore must work for all artifacts required to verify publication.
- Run focused regressions, full relevant suite, offline tracker/replay and categorical smoke evaluation in pinned environments; inspect optional-dependency skips. Perform hosted validation of the actual proposed code, not only an older release.
- Publish one fixed ten-code set per model/target before the declared deadline. New shadows remain research; do not combine their pick unions into one ten-pick accuracy claim.
- Count late/missed/failed targets as declared. Source conflicts or bad hashes block the affected new publication. Uniform is a fallback for an unqualified model, not a way to bless invalid source or timing evidence.

Deliverables: qualification report, frozen identity/configuration, immutable prospective artifacts, publication evidence and recovery runbook. Acceptance: scientific and operational gates pass independently; no altered historical picks; no claim of prospective skill before future evaluation.

### Phase 6 — Confirm accuracy and decide continuation

Scope: the new series' frozen checkpoint. Duration determined by its power calculation.

- Continue the existing series under its original no-interim-aggregate-comparison rule. A new series follows its own preregistered monitoring rule; individual observations do not justify repeated unplanned significance tests.
- At the new checkpoint, reconcile all scheduled targets, missing publications, unresolved results and source corrections; evaluate declared contrasts without pooling operators or incompatible procedures.
- A repeatable exact-hit advantage requires a positive adjusted lower uncertainty bound under the declared method, complete qualification evidence, and no unexplained integrity failures. Distinguish this from a probability-score improvement or one lucky winning draw.
- Report estimates and uncertainty, including null/inconclusive results. A real measured gain still does not guarantee the following draw. Changes after checkpoint start a newly declared procedure with newly reserved future evidence.

Deliverable: confirmation report with advance/retain/reject decision. Acceptance: every conclusion matches its metric and evidence; no changing the objective, number of picks or stopping rule after outcomes.

## 9. Next-draw runbook: realistic actions before the next result

As checked on 10 November, the existing ledger targets **Wednesday, 11 November 2026, 19:00 MYT (11:00 UTC)**. The repository's publication deadline is **18:00 MYT (10:00 UTC)**; this is an internal forecast deadline, not a claim about ticket-sales rules. The official regular schedule supports that date. A special draw requires a separately verified official announcement; this review does not certify the absence of every possible special announcement. [Official schedule](https://www.damacai.com.my/general-information/).

The 11 November forecast is already published. Do not replace or refresh it after seeing this plan, and do not claim it incorporates later information. Newly accepted results can train later targets, not rewrite this one.

| Time / step | Action | Required result |
|---|---|---|
| Before any change | Read the latest remote ledger and compare original publication fingerprints/artifacts | Identify the exact published target and preserve it |
| Before deadline | Check official target/special announcements, operator, all required prior inputs, source conflicts, approved hashes and artifact availability | Record readiness and blockers; do not manufacture a new forecast for a past or already locked target |
| Before deadline | Reproduce the already-published ten picks from its saved distribution and verify push evidence | Confirm what was actually predicted, under which data/procedure, and when |
| If a defect is discovered | Append a dated integrity note; block unsupported new publication/claims; keep original evidence | An invalid or missed forecast is recorded honestly, without a replacement timestamp |
| At draw start | Stop all target-dependent model/selection changes | Published forecast remains fixed |
| After complete official results | Accept the correct dated 23-slot board with provenance; score original saved artifacts | Report exact matched codes and tiers, or an explicit miss/unresolved result |
| Later outage recovery | Use existing bounded overdue fetching and conflict checks | Result recovery never reconstructs or improves the original forecast |
| Next future ungenerated target | Continue the frozen procedure; publish any new research candidate only after its own qualification and identity are ready | No overnight unvalidated model promotion |

Before an implementation release, use the repository's established verification entry points in the intended pinned environment:

```sh
python -m unittest discover -s tests -p 'test_*.py'
python scripts/track_predictions.py --no-sync --dry-run
python scripts/track_predictions.py --replay 3 --no-sync --dry-run
python scripts/evaluate_forecasts.py --protocol docs/FORECAST_PROTOCOL.yaml --phase smoke --offline --output experiments/accuracy-audit-smoke
git diff --check
```

The smoke output path must be new, and smoke success does not qualify a model. New-procedure evaluation uses its new protocol once supported by the runner. Do not run mutating tracker commands against the live ledger merely to draft or test this plan. Record exact code SHA, environment and outputs when these checks are actually executed; the above is a future validation runbook, not a claim of a fresh test run today.

## 10. Deliverables, ownership and stopping rules

The implementer owns reproducible artifacts and fixes; the reviewer checks source/chronology/identity evidence and statistical claims; the maintainer records any new scientific procedure and GitHub metadata changes. These are review responsibilities, not instructions to spawn agents. Keep each completed phase reviewable before proceeding to a dependent phase.

Proposed implementation outputs, **not files created by this planning task**:

```text
experiments/accuracy-audit/<run-id>/
  baseline-manifest.json
  audit-register.json
  source-audit.json
  leakage-audit.md
  power-analysis.json
  candidate-manifest.json
  attempts.jsonl
  per-draw-predictions.jsonl
  negative-controls.json
  candidate-decision.md
  qualification-report.md
docs/ACCURACY_PROTOCOL.md
docs/ACCURACY_PROTOCOL.yaml
docs/ACCURACY_CONFIRMATION.md
```

Stop or block advancement when operator/date/source identity is uncertain, pre-deadline availability is unproven, artifacts cannot be reproduced, the search exhausted its declared budget, required outcomes are unresolved, or a candidate fails qualification. Continue the existing declared uniform procedure where operationally valid, or record a publication failure as applicable; no predictive advantage has been qualified. Do not change seeds, add picks, extend the checkpoint or rewrite history to force a success.

The priority order is **protect the next published forecast's validity; close operator/data/leakage/metric gaps; test compact categorical models; optionally test TabPFN and justified joint selection; demand future exact-hit evidence**. These steps maximize the chance of discovering a real edge if one exists. They cannot make a next-draw exact match certain.

## 11. Scope of this draft

This task creates this plan only. It does not change GitHub metadata, repository slug, production code, configuration, published picks, ledger, probability artifacts or existing research evidence. It does not download model weights, buy API access, qualify a candidate, commit or push.

Document verification checks local references, baseline factual assertions and preservation of existing files. Historical test/research evidence is cited with its date; the full model/release suite is not rerun for a Markdown-only draft. Official certification of the 40 Magnum boards, exhaustive special-announcement coverage and the paywalled Romanian chapter remain unresolved research limits.
