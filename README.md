# FairReweighing

Density-estimation-based reweighing for improving separation in fair regression.

## Environment

Use Python 3.10-3.12. The checked dependency set is intentionally conservative
because `fairlearn==0.9.0` and the original scikit-learn APIs are not reliable on
Python 3.13 yet.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducible JAIR Experiments

Run a quick smoke experiment:

```bash
python src/main.py \
  --datasets Synthetic \
  --models auto ridge \
  --methods none fair-reweighing discretized-reweighing \
  --density-models Neighbor Kernel \
  --repeat 3 \
  --tune-density \
  --output result/jair_runs.csv
```

The command writes:

- `result/jair_runs.csv`: one row per seed/dataset/model/method.
- `result/jair_runs_summary.csv`: means, standard deviations, and 95% CIs.
- `result/jair_runs_comparisons.csv`: paired Wilcoxon tests and effect sizes
  against the `none` baseline.
- `result/jair_weight_examples.csv`: high/low sample-weight examples for
  qualitative discussion.

Run all non-SCUT paper datasets:

```bash
python src/main.py --all-paper-datasets --repeat 30 --tune-density
```

## SCUT-FBP5500

The SCUT loader looks for the neighboring `../Comparable/Data` directory by
default. It uses `ImageExp/Selected_Ratings.csv` for targets and
`landmark_txt/*.txt` as direct image-derived regression features. You can override
paths explicitly:

```bash
python src/main.py \
  --datasets SCUT \
  --models ridge rf gbr \
  --methods none fair-reweighing discretized-reweighing \
  --density-models Neighbor Kernel \
  --repeat 30 \
  --tune-density \
  --scut-data-root ../Comparable/Data \
  --output result/jair_scut_runs.csv
```

If you have precomputed image embeddings, pass a CSV with `Filename` plus numeric
embedding columns via `--scut-embeddings-file`.

## Revision Artifacts

- `docs/reviewer_response_matrix.md` maps review comments to planned/implemented
  manuscript, experiment, and code changes.
- `docs/jair_revision_notes.md` gives manuscript-side changes for the JAIR
  resubmission.
