# FairReweighing

This repository provides the code for the paper **FairReweighing: Density
Estimation-Based Reweighing Framework for Improving Separation in Fair
Regression**.

FairReweighing estimates the density ratio

```text
w(a, y) = rho(a) rho(y) / rho(a, y)
```

with either radius-neighbor local frequencies or Gaussian kernel density
estimation, normalizes the weights to mean one, and passes them to an otherwise
standard downstream learner.

## Repository Layout

- `src/`: experiment, metric, baseline, and figure-generation code.
- `data/`: tabular dataset snapshots and SCUT annotation files.
- `result/raw/`: unaggregated seed-level outputs used by the manuscript.
- `result/main/`: repeated-run means and statistical comparisons.
- `result/parameter_sensitivity/`: fixed-versus-Dynamic ablation outputs.
- `result/synthetic/`: analytic-weight and metric-calibration outputs.
- `result/figures/`: generated manuscript figures.
- `docs/JAIR/`: JAIR LaTeX source, bibliography, figures, and compiled PDF.

## Environment

The artifact uses Python 3.12 and [uv](https://docs.astral.sh/uv/). Install the
locked environment from the repository root:

```bash
uv sync --frozen
```

The reported runs used Python 3.12.12 on macOS 26.5.2, an Apple M4 system
(10 CPU cores), and 16 GB RAM. TensorFlow exposed only the CPU during these
runs; tensorflow-metal 1.2.0 was installed, but no supported GPU device was
available. The runtime reported 128 KB L1 instruction, 64 KB L1 data, and
4 MB L2 cache, with no L3 cache value. Principal package versions were NumPy
2.1.3, pandas 3.0.2, SciPy 1.17.1, scikit-learn 1.8.0, Matplotlib 3.10.9, and
TensorFlow 2.19.1. The lock file records the full software environment.

## Main Experiments

The experiment constants in `src/main.py` specify seeds 1 through 20, the
dataset grid, and the methods. Run the tabular, synthetic, and classification
grid with:

```bash
uv run python src/main.py
```

The main Dynamic density-parameter grid is
`{0.1, 0.2, 0.3, 0.5, 0.75, 1.0}`. For every outer split, the parameter is
selected only from the outer training partition using a 75:25 inner split and
the score

```text
mean component-wise Csep + 0.01 * prediction MSE
```

Insurance and SCUT use the fixed dataset-level scaling documented in the
manuscript; otherwise, the outer test split is used only for final reporting.
The rank rule is the two-sided paired Wilcoxon test with `p < 0.05` plus
directional Cliff's delta `> 0.33`.

Additional analysis commands are:

```bash
uv run python src/synthetic_density_check.py
uv run python src/continuous_metric_calibration.py
uv run python src/classification_exact_match_check.py
uv run python src/density_parameter_rank_graphs.py
uv run python src/generate_methodology_flow.py
```

The scripts checkpoint long analyses under `result/`. Existing aggregate and
seed-level outputs are included, so reproducing a table does not require
rerunning unrelated experiments. Local feature caches are excluded from the
repository.

## SCUT-FBP5500 and VGG-Face

SCUT-FBP5500 images and VGG-Face weights are not redistributed. Obtain them
from their original providers and place them as follows:

```text
data/scut/Images/<SCUT image files>
data/scut/ImageExp/Selected_Ratings.csv
data/scut/vgg_face_weights.h5
```

Alternatively set `SCUT_DATA_ROOT` to the SCUT directory and
`SCUT_VGG_FACE_WEIGHTS` to the weights file. Generate the local frozen-feature
cache through the FairICP probe, then run the 20-split SCUT grid:

```bash
uv run python src/fairicp_probe.py \
  --fairicp-root /path/to/FairICP \
  --datasets SCUT_Sex SCUT_Multi
uv run python src/extend_scut_repeats.py
```

The manuscript uses participant P1's rating, scaled by the fixed dataset-level
maximum before repeated splitting, and a
frozen VGG-Face encoder followed by a one-unit linear head. The parameter-rank
SCUT ablation uses cached features and a common reduced training budget for all
fixed and Dynamic treatments; it is separate from the main SCUT result table.

## FairICP

FairICP is not vendored. Clone its official repository and check out the exact
revision used here:

```bash
git clone https://github.com/yuhenglai/FairICP.git /path/to/FairICP
git -C /path/to/FairICP checkout 6a7662190486ae83357a2c3a521a341ad33e63de
uv run python src/fairicp_probe.py --fairicp-root /path/to/FairICP
```

The regression comparison uses the released linear learner with a seeded,
quantile-bin conditional-permutation adaptation. This is not the full released
MAF/R validation workflow, as stated in the manuscript.

## Data and Licensing

The code is released under the MIT License in `LICENSE`. Dataset files retain
the terms of their original providers; the repository license does not
relicense third-party data. The manuscript cites the Communities and Crimes,
LSAC, German Credit, Heart Disease, Insurance, and SCUT-FBP5500 sources.
SCUT images and VGG-Face weights must be obtained separately under their source
terms. Derived SCUT face-feature caches are also not redistributed.

## Manuscript

Compile from `docs/JAIR/`:

```bash
cd docs/JAIR
latexmk -pdf -interaction=nonstopmode -halt-on-error FairReweighing_JAIR.tex
```

The manuscript reads figures from the same directory. Seed-level CSV files,
not rounded table entries, are the source for all paired tests.

A validated JAIR upload archive is available at
`docs/JAIR/submission/FairReweighing_JAIR_submission.zip`.

## Citation

Please cite the JAIR article after publication. Until then, use the
machine-readable metadata in `CITATION.cff`, which identifies the repository,
manuscript title, and authors.
