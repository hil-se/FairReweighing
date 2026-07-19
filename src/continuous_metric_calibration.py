"""Calibrate the continuous-sensitive C-separation approximation on known cases."""

from pathlib import Path

import numpy as np
import pandas as pd

from metrics import Metrics


OUTPUT = Path("result/synthetic/continuous_metric_calibration.csv")
SEEDS = range(1, 21)
N_SAMPLES = 4000


def main():
    rows = []
    for seed in SEEDS:
        for scenario, y, prediction, sensitive in scenarios(seed, N_SAMPLES):
            metric = Metrics(y, prediction)
            raw = metric.continuous_mi_crossfit_raw(sensitive, seed=seed)
            rows.append({
                "seed": seed,
                "scenario": scenario,
                "c_sep_xfit_raw": raw,
                "c_sep_xfit": max(0.0, raw),
            })

    result = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    summary = result.groupby("scenario", sort=False).agg(
        mean_c_sep=("c_sep_xfit", "mean"),
        sd_c_sep=("c_sep_xfit", "std"),
        mean_raw=("c_sep_xfit_raw", "mean"),
    )
    print(summary.to_string())
    print(f"Wrote {len(result)} rows to {OUTPUT}")


def scenarios(seed, n_samples):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_samples)
    shared = rng.normal(size=n_samples)
    sensitive_noise = rng.normal(size=n_samples)
    prediction_noise = rng.normal(size=n_samples)

    yield (
        "Conditional-independence null",
        y,
        y + 0.5 * prediction_noise,
        0.8 * y + sensitive_noise,
    )
    yield (
        "Linear conditional dependence",
        y,
        y + 0.8 * shared + 0.5 * prediction_noise,
        0.8 * y + 0.8 * shared + 0.5 * sensitive_noise,
    )
    yield (
        "Nonlinear mean dependence",
        y,
        y + shared + 0.25 * prediction_noise,
        0.8 * y + 0.8 * (shared**2 - 1.0) + 0.5 * sensitive_noise,
    )
    yield (
        "Variance-only dependence",
        y,
        y + shared + 0.25 * prediction_noise,
        0.8 * y + np.exp(0.5 * shared) * sensitive_noise,
    )


if __name__ == "__main__":
    main()
