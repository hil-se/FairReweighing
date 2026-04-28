# JAIR Revision Notes

## Positioning

Frame the revision as an AI/fairness contribution. The opening should motivate
fair regression directly: predicted insurance premiums, credit limits, education
outcomes, risk scores, and facial-rating/age-estimation systems are all continuous
or ordinal predictions where separation violations can still harm protected groups.

## Contribution Boundary

Use precise claims:

- FairReweighing extends the density-ratio idea behind Reweighing from discrete
  class labels to continuous regression outcomes.
- In pure classification settings, the method reduces to original Reweighing and
  should be presented as a consistency check, not a new contribution.
- Empirical claims should be tied to the datasets, models, and statistical tests
  reported in the revised results.

## Method Intuition To Add Before The Proof

FairReweighing changes the training distribution, not the labels. Samples from
overrepresented `(A, Y)` regions receive smaller weights; samples from
underrepresented regions receive larger weights. The intended effect is that the
weighted joint density of sensitive attribute and label resembles the product of
their marginals, so the learner sees a training set where `A` is less informative
about prediction errors after conditioning on `Y`.

## Related Work To Add

- Kamiran and Calders: original Reweighing for classification.
- Berk et al., Agarwal et al., Narasimhan et al.: fair regression baselines and
  constraint/regularization approaches.
- Chzhen et al. 2020: plug-in estimator and recalibration with statistical guarantees.
- Chi et al. 2021: accuracy disparity in regression.
- OOD/distribution-shift fairness: discuss as limitation and future work rather
  than claiming current coverage.

## Discussion Section Outline

- When FairReweighing is appropriate: tabular or embedding-based regression where
  sample weights are supported by the learner.
- When assumptions are fragile: strong hidden confounding, severe distribution
  shift, sparse protected-label regions, or learners that ignore sample weights.
- Hyperparameter guidance: use train-only tuning over radius/bandwidth grids;
  report sensitivity curves rather than a single hand-picked value.
- Fairness-accuracy tradeoff: show CI/error bars and avoid interpreting tiny
  differences without paired tests.
- Practical reproducibility: include CLI command, seeds, runtime overhead, and
  data-processing details.
