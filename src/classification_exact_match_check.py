"""Verify that near-zero-bandwidth KDE recovers discrete Reweighing."""

from pathlib import Path

import numpy as np
import pandas as pd

from experiment import Experiment


DATASETS = ["German", "Heart"]
SEEDS = range(1, 21)
BANDWIDTH = 1e-6
METRICS = ["accuracy", "f1", "aod", "eod", "r_sep", "i_sep", "c_sep_xfit"]
OUTPUT_DIR = Path("result/parameter_sensitivity")
RAW_OUTPUT = OUTPUT_DIR / "classification_exact_match_check.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "classification_exact_match_summary.csv"


def main():
    rows = []
    for dataset in DATASETS:
        for seed in SEEDS:
            reweighing = Experiment(
                data=dataset,
                regressor="logistic",
                balance="reweighing",
                density_model="Reweighing",
                seed=seed,
            ).run()
            kernel_experiment = Experiment(
                data=dataset,
                regressor="logistic",
                balance="fair-reweighing",
                density_model="Kernel",
                seed=seed,
            )
            kernel_experiment.select_density_param = lambda: BANDWIDTH
            kernel = kernel_experiment.run()

            for treatment, result in (("Reweighing", reweighing), ("Kernel-1e-6", kernel)):
                rows.append({
                    "dataset": dataset,
                    "seed": seed,
                    "treatment": treatment,
                    "bandwidth": np.nan if treatment == "Reweighing" else BANDWIDTH,
                    **{metric: result[metric] for metric in METRICS},
                })

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    verify_exact_match(summary)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote {RAW_OUTPUT} and {SUMMARY_OUTPUT}")


def summarize(raw):
    rows = []
    for dataset, group in raw.groupby("dataset", sort=False):
        reference = group[group["treatment"].eq("Reweighing")].set_index("seed")
        kernel = group[group["treatment"].eq("Kernel-1e-6")].set_index("seed")
        row = {"dataset": dataset, "seeds": len(reference)}
        for metric in METRICS:
            ref_values = pd.to_numeric(reference[metric], errors="coerce")
            kernel_values = pd.to_numeric(kernel[metric], errors="coerce")
            row[f"reweighing_{metric}"] = float(ref_values.mean())
            row[f"kernel_{metric}"] = float(kernel_values.mean())
            row[f"max_abs_diff_{metric}"] = float((ref_values - kernel_values).abs().max())
        rows.append(row)
    return pd.DataFrame(rows)


def verify_exact_match(summary):
    difference_columns = [column for column in summary if column.startswith("max_abs_diff_")]
    maximum = summary[difference_columns].max().max()
    if not np.isfinite(maximum) or maximum > 1e-10:
        raise RuntimeError(f"Near-zero KDE did not recover Reweighing; max difference={maximum}")


if __name__ == "__main__":
    main()
