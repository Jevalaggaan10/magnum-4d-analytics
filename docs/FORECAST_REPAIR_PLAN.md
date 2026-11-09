# Forecasting reliability: review and repair plan

Status: phases 1–5 implemented; phase 6 release validation in progress. See
`FORECAST_REPAIR_IMPACT.md` and `experiments/repairs/20261109` for evidence.

Baseline: `eed96e731f96b55bcec3306cf770ad4853798714`, reviewed 2026-11-09. Scope covers three issue groups from the recent review. Experiment identity enforcement is split into two implementation tasks because record compatibility and runtime source validation fail at different boundaries.

## Intended result

1. One confirmation checkpoint evaluates one declared scientific procedure and its first 1,000 scheduled targets. Incompatible procedures cannot be pooled.
2. Live generation checks frozen code/protocol evidence before publishing forecasts under an existing experiment identity.
3. Published v3 forecasts can retrieve delayed results even after those dates leave the recent-results window.
4. Development selection follows a declared, deterministic rule for practical improvement, paired uncertainty and simplicity.

Keep published forecasts, original publication times, saved probability vectors, historical selection outputs, missing-target denominators and legacy v2 records intact. Corrections add evidence and version history; they do not reconstruct prospective forecasts.

## Evidence established before planning

| Work item | Reproduced behavior | Root cause | Current production implication |
|---|---|---|---|
| A. Confirmation isolation | 500 verified records from series A plus 500 from series B can return both probability promotion and exact-hit advantage when gates are enabled. | `confirmation_status` and `promotion_decision` sort/truncate all records without validating experiment identity. | Current gates remain false; invalid pooling is possible after later changes. |
| B. Frozen-source enforcement | Supplying an incorrect frozen `forecasting.py` hash still permits generating another forecast under `v3-fixed-1000`. | Frozen hashes are configuration metadata, not a runtime precondition. | Current configured source hashes match actual files. Missing enforcement remains a future integrity risk. |
| C. Delayed-result recovery | Real CLI flow with a published November 7 forecast and latest-five results starting November 14 makes zero targeted requests; old forecast stays `published`. | Targeted fetching exists only for legacy `pending`; v3 scoring requires a result already in the corpus. | Current ledger has no observed stranded v3 result, but an extended outage can cause one. |
| D. Selection uncertainty | Deterministic candidate outputs select `full_code` despite a paired loss interval crossing zero. | `run_search` changes champion on mean gain alone. Later intervals cannot influence that earlier decision. | Current champion is uniform. Future development and annual inner searches can violate the protocol. |

Reproductions used temporary directories, fixed dates and synthetic outcomes. Network calls were mocked only at the transport boundary. No live prediction records were changed. Existing evidence includes 71 passing tests, successful offline tracker/replay checks and valid live artifact hashes; passing tests do not cover these failure cases.

## Phase 1 — Lock down evidence and write failing regressions

Before implementation:

- Fetch and record the implementation starting SHA. Inspect local changes before editing.
- Record byte hashes for `forecast_config.json`, `predictions.json`, `PREDICTION_LOG.md`, live probability artifacts, frozen protocol, selection manifests and affected source files. Retain this inventory with repair evidence.
- Audit all current forecast records: series label, settings hash, forecast configuration, artifact integrity, publication status, unresolved result dates and duplicate/missing scheduled targets. Record findings separately; do not rewrite ledger metadata merely to satisfy a validator.
- Reproduce each issue through the actual shared function or CLI caller. Convert the reproduction into a focused regression test before its fix. Confirm failure is the intended behavior assertion rather than missing fixture fields or setup errors.
- Freeze time, seeds, filesystem paths and remote responses. Use real artifact save/load and scoring code. Small bootstrap replicate counts are acceptable for deterministic plumbing tests; final statistical validation uses protocol settings.

Affected call-site inventory:

| Area | Review/repair paths |
|---|---|
| Confirmation | `forecast_evaluation.py`: `confirmation_status`, `promotion_decision`; `probability_tracker.py`: advancement and log rendering; `tests/test_forecast_evaluation.py` |
| Scientific identity/source evidence | `forecast_config.json`; `probability_tracker.py`; settings loading/publication confirmation in `scripts/track_predictions.py`; saved-forecast/preview integration in `app.py`; `tests/test_probability_tracker.py`; protocol and continuation documents |
| Recovery | `scripts/track_predictions.py`: recent sync, targeted fetching, corpus merge; `probability_tracker.py`: observation acceptance and artifact scoring; `data_sources/damacai.py`: date-specific fetch/rate limiting; `tests/test_tracker.py`, `tests/test_probability_tracker.py` |
| Development selection | `scripts/evaluate_forecasts.py`: search, annual inner selection, resume and decision artifacts; `forecast_evaluation.py`: paired comparison helpers; `tests/test_experiment_runner.py` |
| Release | `.github/workflows/forecast-validation.yml`, `.github/workflows/prediction-tracker.yml`, README, repair impact evidence |

## Phase 2 — A. Isolate confirmation by scientific identity

### Define compatibility before changing counters

Use one shared compatibility rule across status reporting, promotion, generation and publication confirmation. At minimum, validate declared series, operator/target definition, champion/shadow/reference configurations, feature/calibration versions, protocol, dataset-lineage definition and approved code revision evidence.

Preserve the existing `settings_sha256` as a provenance hash. Do not use the entire settings object as the sole scientific identity: it currently includes operational fields such as qualification flags, correctness notes, requested dates and announcement inputs. Changing an administrative flag must not silently create a new statistical procedure. Conversely, changing a shadow model must not be accepted merely because the champion remains uniform.

Keep two concepts explicit:

- **Scientific series identity:** stable declared procedure and checkpoint schedule.
- **Execution revision evidence:** exact source/protocol/dependency evidence used for a particular forecast, with explicitly reviewed compatibility where appropriate.

Ordinary new observations legitimately change each forecast's training-corpus hash. Validate chronology and declared data lineage rather than requiring every forecast to share identical training bytes. Special-draw announcements belong to individually evidenced scheduled targets; adding a valid announcement must not erase earlier regular targets.

### Required behavior

- Evaluate an explicitly selected series; archived series remain separately inspectable.
- Reject incompatible records claiming the same series. Do not silently filter them away and replace them with later successes.
- Do not combine different series to reach the checkpoint. Duplicate target dates and missing identity evidence cannot add successful coverage.
- Build the checkpoint from the declared schedule, including missing/late targets. A placeholder with no forecast remains a failed scheduled target; it does not require invented artifact metadata.
- Validate identity before calculating comparative intervals or enabling either promotion flag. Missing identity or compatibility evidence returns a clear non-promotable status.
- Retain the existing gates: incomplete coverage/scoring, insufficient development qualification and unverified correctness prevent promotion. Under 1,000 scheduled targets, do not expose interim aggregate comparisons.
- Recompute status after adding scheduled missing-target records or changing forecast states, so stored confirmation and rendered log describe the same state snapshot.
- Route log rendering through the same selected-series validation as promotion. Update affected fixtures with real identity evidence rather than weakening validation to keep old tests green.

### Backward compatibility

Existing published records cannot be assigned a new identity merely from today's configuration. Create a small, explicit compatibility/impact manifest tied to original configuration/source/artifact evidence if the existing series is retained. Preserve original record fields and hashes; any legacy mapping must identify the source evidence supporting it. Unknown records remain unverified.

For these repairs, recommend retaining the current series only if compatibility review proves generation/scoring equivalence for existing forecasts and classifies the changes as correctness fixes. If the procedure changes substantively, terminate the old series transparently and declare a separate series using new future targets. Do not reset the old checkpoint or recycle missing outcomes to improve its result.

### Regression acceptance

- Mixed 500/500 series cannot yield an evaluated pooled checkpoint or either promotion flag.
- Same series label with incompatible model/protocol identity is rejected.
- Matching valid series still reaches the existing scoring/promotion checks.
- Missing identity, duplicate targets, archived series and missing-forecast placeholders behave explicitly and deterministically.
- Harmless runtime flag changes preserve scientific identity; model/calibration changes do not.
- Missing/late targets remain in the original denominator; no interim comparative scores leak into status/log output.

## Phase 3 — B. Enforce frozen execution evidence

### Generation boundary

Before generating a new prospective forecast, compare actual files and protocol against the approved revision evidence for the selected series. Validate required paths, missing entries, digest format and contents. The current three-file source list is the starting point; review its dependency closure, including normalization and orchestration code that can change training inputs, targets or scores. Document coverage and limitations of the frozen file/dependency list.

A source/protocol mismatch must block new publication under that identity with a precise diagnostic. Do not automatically update expected hashes to whatever files are present, silently create a replacement series, or set correctness/development gates to true. Validate series/revision compatibility again when confirming publication so old pending records cannot inherit unrelated approval.

Artifact-backed verification of already published forecasts must use their original saved vectors. Separate that operation from permission to generate new forecasts: a source mismatch must not silently refit old records. Define and test failure persistence: accepted results and integrity-block information are saved through the CLI's normal state/log path; the workflow can commit audit updates while declining new forecast confirmation. If this needs workflow changes, keep the invalid-generation path visibly unsuccessful and ensure it cannot mark coverage as successful publication.

Use a single validator shared by CLI/advancement and relevant UI preview paths. A saved, hash-valid published forecast remains inspectable; a new preview under mismatched code must not appear to be an approved frozen forecast. Avoid separate ad hoc hash checks at each caller.

### Handling the repair itself

The fixes change files listed in `source_hashes`, so enforcement cannot be introduced by blindly retaining the old digest list or silently overwriting it. Commit reviewed revision evidence containing old/new source hashes, reason, impact classification, equivalence checks and affected records. Freeze the final implementation revision after tests pass. Only an explicitly compatible correctness revision may continue an existing scientific series.

### Regression acceptance

- Altered/missing source or protocol bytes block new generation before forecast artifacts are written.
- Missing/invalid manifest evidence is not treated as implicit approval.
- Compatible declared revision passes; undeclared revision fails.
- Published records retain original picks, vectors, publication evidence and timestamps.
- Recovery/scoring and audit persistence follow the documented mismatch path without granting publication coverage.
- CLI, publication confirmation and UI agree about approved execution identity; dry-run remains write-free.

## Phase 4 — C. Recover overdue published v3 results

### Recovery flow

After recent sync, inspect v3 records with `status='published'` whose scheduled draw start has passed and whose complete result is absent from the accepted corpus. Fetch those dates explicitly using the existing date-specific adapter. Process dates deterministically, deduplicate requests and merge accepted outcomes before calling existing saved-artifact verification.

Review the actual semantics of the current endpoint and adapter response: the returned outcome must be tied to the requested date, not merely be a complete latest result. If that cannot be established, keep the forecast unresolved and record the limitation.

- Route recovered results through `accept_observations`, complete-slot normalization, future-time rejection, conflict quarantine and provenance hashing. Do not add a bypass path straight into `observed_draws`.
- Keep original artifacts and picks. Recovery supplies outcomes; it never generates a prediction for a past target or changes publication success.
- Respect `--no-sync`, transport timeouts and existing polite rate limits. A recent-index failure should not prohibit an independent date-specific attempt when the transport remains usable.
- Use a bounded per-run request budget within the existing 15-minute workflow limit. Distinguish recent-fetch status from recovery attempts. A missing/failed date remains unresolved, with concise diagnostic evidence; failures must not erase successful recoveries from other dates.
- Prevent backlog starvation: persist minimal retry/cursor evidence or an equivalent deterministic schedule so one permanently unavailable old date cannot consume every run's budget. Avoid a new queue framework.
- Skip verified/locally complete targets and all future targets. Do not apply the legacy 60-day expiry policy to v3 unless a separate declared rule justifies it; expiry must never count as successful verification.
- Maintain legacy pending behavior and conflict handling. Recovery can enrich training data for later forecasts; it cannot alter already published forecast artifacts.

### Regression acceptance

- Real CLI integration: an overdue target outside latest five dates receives a targeted fetch, gains accepted provenance and becomes `verified` using its original artifact.
- Offline mode and future targets cause zero network calls.
- Incomplete, conflicting, wrong-date and unavailable responses never become verified results.
- Repeated runs are idempotent: no duplicate observation/score records and no forecast mutation.
- One failed recovery does not discard another successful recovery; backlog progresses within the request budget.
- Recent-results failure and targeted-fetch success are independently represented.
- Dates around 19:00 MYT and announced special targets obey the same time boundary.

## Phase 5 — D. Make uncertainty affect development selection

### Declare the rule before a new search

The existing protocol requires comparing paired uncertainty and preferring simpler models within uncertainty, but does not fully specify an executable development decision rule. Document an explicit interpretation before implementation is used for new selection. Proposed conservative rule:

1. Compare a healthy, complete candidate with the current selected procedure on identical eligible draw dates.
2. Require mean log-loss improvement of at least `.001` nats per slot.
3. Require paired candidate-minus-incumbent intervals to have upper bounds below zero under both declared block lengths, 13 and 26 draws. Reuse the declared seed/interval method; record the confidence level and its exploratory development status.
4. Otherwise retain the incumbent; deterministic complexity ordering resolves equivalent candidates. Define ordering for model families and within-family settings explicitly, rather than relying on candidate arrival order.

This is a proposed precise policy, not a claim that the historical protocol already specified this exact significance gate. Record any formalization as a versioned protocol amendment for future runs. It must not be chosen after inspecting outer-year results. Distinguish development selection from the separate, fixed future-confirmation contrasts and promotion requirements.

### Apply the rule consistently

- Retain incumbent per-draw rows, including an explicit uniform comparator. Do not select against only a scalar loss with no aligned baseline.
- Centralize the comparison used by development and annual inner searches; keep outer outcomes out of all selection decisions.
- Use qualified selection decisions for champion changes, round advancement and no-gain counters. Descriptive best-observed/nonuniform references may still be recorded, but cannot imply qualification.
- Keep fallback/degraded/partial candidates in attempt evidence and ineligible for selection as currently intended. Incomplete comparator dates must not be silently intersected to produce an easier comparison.
- Save decision evidence: candidate/incumbent identifiers, paired date count, practical gain, both intervals, complexity preference and acceptance/rejection reason.
- Charge comparisons to existing wall-clock budgets. A timeout records an unfinished decision and cannot promote a candidate using partial bootstrap work.
- Resume/reuse must check data, protocol, code and fold identity before using prior rows or selection decisions. A changed decision policy requires a new output directory and versioned run manifest.
- Do not overwrite `experiments/damacai-v3/run-002` or retrofit historical qualification. Reproduce historical inputs only in a separate impact run; keep the original conclusion and artifacts auditable.

### Regression acceptance

- Candidate with mean gain `.002` and interval crossing zero retains uniform.
- Clearly improved, healthy candidate with both intervals below zero can be selected.
- Subthreshold gain, contradictory block-length outcomes, mismatched dates, fallback/degraded candidates and equal-loss ties cannot create an unsupported promotion.
- Same inputs/seeds produce the same selected model and recorded reason.
- Round/no-gain behavior reflects qualified decisions; annual inner search uses the same logic without consulting its outer target outcomes.
- Changed protocol or incompatible resume evidence cannot silently reuse old selection decisions.

## Phase 6 — Integration, evidence preservation and release

Implement in reviewable units: A (series isolation), B (execution evidence), C (result recovery), D (selection policy). A and B share identity/migration decisions and should land together if an intermediate commit would block all valid live generation. C and D can remain separate fixes. No unrelated UI redesign, model-family expansion or dependency additions are required.

Validation sequence for implementation:

```bash
# Run focused regressions first, then the full suite once changes are integrated.
.venv/bin/python -m unittest discover -s tests -p 'test_forecast_evaluation.py'
.venv/bin/python -m unittest discover -s tests -p 'test_probability_tracker.py'
.venv/bin/python -m unittest discover -s tests -p 'test_tracker.py'
.venv/bin/python -m unittest discover -s tests -p 'test_experiment_runner.py'
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

# Read-only checks against production-shaped records.
.venv/bin/python scripts/track_predictions.py --no-sync --dry-run
.venv/bin/python scripts/track_predictions.py --replay 3 --no-sync --dry-run
git diff --check
```

Also run isolated end-to-end tracker scenarios with `--state-file`, `--log-file` and `--artifact-dir` pointing to temporary paths. Compare original and resulting ledger/artifact byte hashes. Expected outcome/status additions must be itemized; existing prospective inputs, publication evidence and vectors must remain unchanged.

Run the existing offline smoke benchmark in a new disposable experiment directory and both hosted Python 3.12 environments (tracker and full) through the read-only validation workflow. Include regressions in the scheduled tracker's existing suite. Exercise the source-mismatch workflow path and successful publication path in isolation before relying on live scheduled publication. No paid model search or live prospective write is necessary for repair validation.

Before release, produce a short impact record covering source/protocol revisions, old-record compatibility, unchanged forecast vectors, new regression results and remaining operational limitations. Update `docs/FORECAST_CONTINUATION.md`, protocol version/evidence and README where behavior changes. Keep both promotion gates false until their stated evidence requirements are actually satisfied.

Publish code only after verification and revision evidence agree. During implementation, fetch again before pushing and preserve concurrent scheduled ledger commits through a normal merge. Avoid replacing live ledger files with copied test fixtures or older snapshots.

Rollback: revert defective code through a normal commit while retaining observations and publication history. Use an explicitly approved source revision if restoring previous code. Do not reset ledger history, delete failures or bypass integrity checks to restart publication.

## Completion checklist

- [x] Each reported failure has a regression observed failing before its implementation fix.
- [x] Scientific identity is shared by generation, confirmation, promotion and status rendering.
- [x] Incompatible/missing identity and source evidence prevent new or mixed-series promotion/publication.
- [x] Compatibility of current published records is documented without rewriting their original evidence.
- [x] Overdue v3 targets recover outside the recent-results window, with validated date/provenance and bounded retries.
- [x] Selection decisions implement a documented, versioned uncertainty/simplicity rule.
- [ ] Full regression suite, dry-runs, artifact checks, offline smoke and hosted validation pass.
- [x] Production ledger changes, if any, are explicitly explained; no prospective forecast was reconstructed.
- [x] Uniform remains champion unless a separately qualified procedure satisfies all existing evidence gates.
- [ ] Final handoff names commit(s), checks run, identity/compatibility decision and outstanding limitations.

Implementation limits: no new model-search grid or forecast/ledger migration was performed. Read-only official archive checks verified response dates against two retained outcomes. The uncertainty rule is now declared in `FORECAST_SELECTION_POLICY.json` for new searches; original protocol/selection artifacts remain unchanged. Hosted release evidence is tracked separately in phase 6.
