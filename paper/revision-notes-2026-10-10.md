# Revision notes — 2026-10-10

Implemented the review corrections in the manuscript and analysis scripts.

- Added modified Knapp–Hartung sensitivity intervals; qualified significance and equivalence claims that depend on normal intervals.
- Corrected calibration counts, policy/strict ranking differences, baseline comparisons, and dataset-cutoff wording.
- Clarified target-label selection of raw probability readouts, transfer-group semantics, exploratory RAG comparisons, and configuration confounds in same-base controls.
- Reframed answer masking as a descriptive perturbation with a potential marker cue, rather than a causal mechanism test.
- Scoped downstream RAG conclusions to the tested bge baseline. A Qwen8B downstream generation experiment remains outstanding and was not run.
- Added a compact main ranking table, the full appendix ranking table, sensitivity table, and updated reliability figure.
- Added complete-grid and stale-cache checks, split/input provenance, targeted tests, and reproduction commands.

Validation: six unit tests passed; RAG summaries regenerated from complete cached grids; reported F1 values unchanged; revised 11-page PDF compiled and visually inspected. No paid inference was run. Historical model weight commits cannot be recovered from the original logs. One normalized pilot/final MuSiQue question overlap is disclosed; the historical splits are preserved.

Original manuscript backup: C:/Users/parvb/AppData/Local/Temp/choicerank-before-revision-20261010.

The paper directory is ignored by Git in the existing repository configuration. Manuscript edits are saved locally but are not included in the tracked-code diff.
