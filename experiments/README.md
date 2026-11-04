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

Complete compressed probability vectors (*.npz) remain in this local working tree
but are ignored by Git (several GB). Per-draw prediction manifests carry exact
relative paths plus content/file SHA-256. Open original run directories locally;
regenerate using the frozen CSV, protocol and resolved packages on another machine.
No cloud upload or prospective ledger rewrite occurred. Baseline artifact console
messages saying updated state/log refer only to regression-test temporary files.
