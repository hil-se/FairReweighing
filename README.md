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
python src/main.py
```

Edit the constants at the top of `src/main.py` to change datasets, models,
density estimators, repeat count, or SCUT target.

The command writes:

- `result/jair_runs.csv`: one row per seed/dataset/model/method.
- `result/jair_runs_summary.csv`: means, standard deviations, and 95% CIs.
- `result/jair_runs_metric_correlations.csv`: Pearson/Spearman correlations
  among `r_sep`, `i_sep`, the old `c_sep`, and cross-fitted `c_sep_xfit`.

Run all non-SCUT paper datasets:

Set `DATASETS = ["Synthetic", "LSAC", "Community", "Community_Con", "Insurance", "German", "Heart"]`
and `REPEAT = 30` in `src/main.py`, then run `python src/main.py`.

## SCUT-FBP5500

The SCUT loader is standalone by default and looks under `data/scut` for
`ImageExp/Selected_Ratings.csv`, `Images/`, and `vgg_face_weights.h5`. Its
feature source is raw image paths, and `MODELS = ["auto"]` selects the VGG-Face
single encoder for SCUT.

Set `DATASETS = ["SCUT"]`, `MODELS = ["vgg_face"]`,
`DENSITY_MODELS = ["Neighbor", "Kernel"]`, and `REPEAT = 30` in `src/main.py`,
then run `python src/main.py`.

## Revision Artifacts

- `docs/reviewer_response_matrix.md` maps review comments to planned/implemented
  manuscript, experiment, and code changes.
- `docs/EMSE2025.tex` contains the revised manuscript draft for JAIR reframing.
- `docs/jair_revision_notes.md` summarizes manuscript-side revision guidance.
- `docs/references_to_add.bib` contains new BibTeX entries to merge into the
  main bibliography source.
