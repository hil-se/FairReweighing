# Result Folder Layout

- `main/`: manuscript-ready averaged tables, summary table, and statistical-test tables.
- `raw/`: seed-level raw experiment outputs used to regenerate averages and paired tests.
- `figures/`: generated result figures and heatmaps.
- `parameter_sensitivity/`: fixed/dynamic radius and bandwidth analyses, rank summaries, pairwise tests, and the near-zero-bandwidth classification consistency check.
- `baselines/fairicp/`: FairICP raw, summary, and comparison outputs.
- `baselines/fairbalance/`: FairBalance classification outputs.
- `synthetic/`: synthetic ground-truth density and prediction checks.
- `cache/`: local reusable caches, currently frozen SCUT VGG-Face features.
  This directory is intentionally excluded from version control and must be
  regenerated from separately obtained source assets.

Removed as unnecessary: checkpoint CSVs and smoke-test outputs whose final raw/summary files are already preserved.
