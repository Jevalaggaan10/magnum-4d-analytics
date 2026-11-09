# Forecast reliability repair impact — 2026-11-09

Baseline: `eed96e731f96b55bcec3306cf770ad4853798714`. The repair plan is
`FORECAST_REPAIR_PLAN.md`; raw evidence is in `experiments/repairs/20261109`.

## Classification and series continuity

Retain `v3-fixed-1000` as a compatible live correctness revision. Champion,
shadow/reference configurations, target definition, original frozen protocol,
calibration, training-prefix rules, probability generation and saved-vector
scoring remain unchanged. Both development and correctness promotion gates remain
false. Missing outcomes/publications remain in the same scheduled denominator.

An exact equivalence check regenerated every one of the three saved targets under
its original configuration and strict historical prefix. All nine model vectors,
slot mappings, picks, training/feature hashes and available stored scores matched.
No existing forecast, ledger, artifact or publication log was edited. Original
record fingerprints, configuration/source hashes and baseline commit are pinned
in `forecast_compatibility.json`; original records are not assigned invented fields.

The current execution manifest expands frozen evidence to normalization,
orchestration, official-result parsing, dependency constraints and the tracked
snapshot/provenance. Scientific identity excludes administrative flags and the
growing per-target corpus hash. New code requires explicit compatible revision
evidence; runtime never refreshes its own expected hashes.

## Behavior changes

- Confirmation rejects incompatible/missing identity and duplicate targets before
  comparisons. Selected series exclude archives but retain missing scheduled dates.
- Generation/publication validate frozen evidence. On an integrity block, accepted
  outcomes and diagnostics can be committed, publication is skipped, and CI fails.
- Published v3 forecasts recover overdue results independently of the recent-date
  index. Five attempts per run, least-attempted-first retry ordering, per-date error
  evidence, complete-slot checks and official `drawDate` validation bound recovery.
- Announced special dates are retained even when network failure prevents generation.
- A separately versioned development policy formalizes uncertainty and complexity
  preference for new offline searches. This changes future search procedure, not
  the frozen historical selection or current production models. It is not evidence
  of predictive advantage. Old searches cannot be silently resumed under this policy.

## Verification

Four original failure regressions were observed failing before implementation.
Additional red evidence covers missing/duplicate identities, wrong-date responses
and loss of special-draw schedule declarations. Focused tests exercise shared
functions and real CLI/artifact paths; network/time are isolated at boundaries.

Final test, smoke, preservation and hosted validation results are recorded with
release evidence in `experiments/repairs/20261109`. Read-only archive checks on
November 7/4 responses confirmed date evidence is available; they did not modify
the prospective ledger. No new model-search grid or prospective publication was
run for this repair.

Local full regression suite: **100 tests passed**. Offline tracker/replay and the
probability smoke benchmark passed. Thirteen published/frozen files retained
their baseline byte hashes, including ledger, log, all seven live probability
artifacts, original protocol and frozen selection evidence.

Hosted Python 3.12 tracker and full environments both passed regression, offline
tracker/replay, probability smoke and publication preservation checks on code commit
`e90a39ca7a6661152cb2d2886c3bad7f837f643c`:
[forecast validation run](https://github.com/Jevalaggaan10/magnum-4d-analytics/actions/runs/42612674111).
The release evidence update changes documentation/results only; executable files
and approved source hashes remain identical to that tested commit.

## Limits and rollback

Hashes establish exact reviewed file bytes, not independent proof of every past
source observation or of the lottery model. Endpoint/schema/network failures remain
unresolved outcomes rather than fabricated verification. The fixed future checkpoint
still requires future draws; passing software checks does not establish an edge.

Rollback through a normal revert and explicitly approved execution revision.
Preserve observed outcomes, publication history and failure denominators. Never
reset the confirmation checkpoint or copy test state over production records.
