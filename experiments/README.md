# Evidence store

baseline-20261103: raw original commands, package/environment versions, failures,
leading-zero/null contract reproduction on Python 3.11/3.12, trained XGBoost smoke.
data-v3: frozen canonical corpus, dates, hashes and provenance gaps.
damacai-v3/run-001: preserved first five baseline attempts, followed by round-review
serialization failure. run-002 reused those forecasts rather than refitting.
damacai-v3/run-002: development, rejected candidates, freeze, nested outer selection,
inspected retrospective, calibration and loss intervals. Partial interrupted outer
attempt remains in place and ineligible. resume-frozen*.txt documents resumption.
verification: constrained/full regression outputs, actual isolated tracker, replay,
UI operator smoke, runtime and frozen-consensus comparator.

## Storage cleanup — 10 November 2026

Generated offline probability vectors (`experiments/**/*.npz`) were removed from
this working tree: 25,371 files, 5,124,949,651 bytes. Per-draw prediction records,
scores, hashes, attempt logs, frozen selection and source evidence remain. Their
historical vector paths describe the original runs; the files are no longer
available for direct reload or resume. Rebuilding vectors requires the recorded
historical code revision, frozen CSV, protocol and resolved environment in a new
output directory. Do not describe rebuilt artifacts as the original saved bytes.

Redundant generated `verification/offline-smoke/` and
`repairs/20261109/smoke/` output directories were also removed; their console
reports and hosted validation evidence remain. CI recreates its own ignored
`experiments/ci-smoke/` output. Keep new disposable smoke output outside the evidence
store or ignore its exact output path.

Live `probability_artifacts/`, `predictions.json` and `PREDICTION_LOG.md` are retained
unchanged: these saved forecasts are necessary for prospective verification.
Baseline console messages saying updated state/log refer only to regression-test
temporary files. Never replace a prospective forecast with a regenerated replay.
