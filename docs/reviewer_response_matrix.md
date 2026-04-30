# Reviewer Response Matrix

This matrix is an internal checklist for the JAIR revision. The current
manuscript draft is `docs/EMSE2025.tex`; code and artifact changes are
implemented in the repository, while long-running result regeneration remains a
separate execution step.

| Review concern | Response type | Concrete fix |
| --- | --- | --- |
| R1: Novelty relative to Reweighing | Manuscript | State that FairReweighing generalizes Reweighing to continuous/regression outcomes and reduces to original Reweighing in classification; remove novelty claims for pure classification. |
| R1: Proposed metric not sufficiently validated | Experiment + code | Add repeated-run metric columns for `continuous_mi_*`, `r_sep_*`, BGL, and paired summaries; validate continuous MI against older metrics where sensitive attributes are binary. |
| R1/R2/R3: Limited datasets | Experiment + code | Keep Synthetic, LSAC, Communities, Insurance, German, Heart; add SCUT-FBP5500 with landmark/image-derived features and sex/race sensitive attributes. |
| R1/R2/R3: Limited models | Experiment + code | Add CLI support for `linear`, `ridge`, `svr`, `rf`, `gbr`, `mlp`, and `logistic`/`auto` model selection. |
| R1/R2/R3: No statistical rigor | Experiment + code | Emit one row per seed and generate summary CSVs with means/std/95% CIs plus paired Wilcoxon tests and paired effect sizes vs no mitigation. |
| R1/R2: Missing sensitivity analysis | Experiment + code | Add `--tune-density`, `--radius-grid`, and `--bandwidth-grid`; select radius/bandwidth using only the training split and record selected values. |
| R1: No fairness-accuracy tradeoff | Experiment + manuscript | Use raw result CSV to plot MSE/MAE/R2 against `fairness_mean_abs_continuous_mi`, BGL, and weight-effective sample size. |
| R1: No qualitative analysis | Experiment + manuscript | Write high/low sample-weight examples to `result/jair_weight_examples.csv` for case-study discussion. |
| R1/R3: Runtime overhead unclear | Experiment + code | Add `fit_seconds` and `total_seconds` per run. |
| R1/R2: Baselines favor proposed method | Manuscript + experiment | Include `none`, FairReweighing-Neighbor, FairReweighing-KDE, and discretized/original Reweighing; discuss Chzhen and Chi conceptually unless exact implementations are added. |
| R2/R3: Weak SE positioning | Manuscript | Reframe for JAIR as an AI/fairness paper; avoid forcing an EMSE-specific software engineering contribution. |
| R3: Introduction confusing | Manuscript | Start with fair regression examples, then limitations of classification-centric fairness, then contributions. |
| R3: Proof hard to read | Manuscript | Add intuitive density-ratio explanation before the formal proof. |
| R3: Single vs multiple sensitive attributes unclear | Manuscript + code | Explain that `A` may be a vector; code passes all protected columns jointly into the density estimator. |
| Minor: Race vs Race% | Manuscript + code | Use `Community` for binary majority race and `Community_Con`/Race% for continuous `racepctblack`; document in experiment setup. |
| Minor: Typos and phrasing | Manuscript | Fix face-recognition wording, "metric evaluating", "separation concerning", "balances", and EOD capitalization. |
| Transparency/reproducibility | Code + docs | Replace hardcoded `community()` run with CLI and document commands in README. |

## Remaining manuscript-only tasks

- Replace placeholder legacy result tables with regenerated tables/figures after the full experiment grid completes.
- Add tables/figures from the new CSV artifacts once long-running experiments complete.
- Avoid unsupported "state-of-the-art superiority" claims unless statistical tests support them.
- Keep the BibTeX entries in `docs/references_to_add.bib` synchronized with `docs/mybib.bib` if the manuscript bibliography is maintained outside this repository.
