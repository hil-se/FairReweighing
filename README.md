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
  --density-models Neighbor Kernel \
  --repeat 3
```

The command writes:

- `result/jair_runs.csv`: one row per seed/dataset/model/method.
- `result/jair_runs_summary.csv`: means, standard deviations, and 95% CIs.
- `result/jair_runs_comparisons.csv`: paired Wilcoxon tests and effect sizes
  against the `none` baseline.
- `result/jair_runs_weights.csv`: high/low sample-weight examples for
  qualitative discussion, when reweighing is used.

Run all non-SCUT paper datasets:

```bash
python src/main.py \
  --datasets Synthetic LSAC Community Community_Con Insurance German Heart \
  --repeat 30
```

## SCUT-FBP5500

The SCUT loader is standalone by default and looks under `data/scut` for
`ImageExp/Selected_Ratings.csv`, `Images/`, and `vgg_face_weights.h5`. Its
feature source is raw image paths, and `--models auto` selects the VGG-Face
single encoder used by the neighboring `../Comparable` project.

```bash
python src/main.py \
  --datasets SCUT \
  --models vgg_face \
  --density-models Neighbor Kernel \
  --repeat 30
```

## Revision Artifacts

- `docs/reviewer_response_matrix.md` maps review comments to planned/implemented
  manuscript, experiment, and code changes.
- `docs/EMSE2025.tex` contains the revised manuscript draft for JAIR reframing.
- `docs/jair_revision_notes.md` summarizes manuscript-side revision guidance.
- `docs/references_to_add.bib` contains new BibTeX entries to merge into the
  main bibliography source.
