from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from data_reader import load_dataset
from experiment import Experiment


# Run configuration. Edit these constants, then run `python src/main.py`.
#
# DATASETS:
#   "Synthetic", "LSAC", "Community", "Community_Con", "Insurance",
#   "German", "Heart", "SCUT"
#
# MODELS:
#   "auto"      -> linear for regression, logistic for classification, vgg_face for SCUT
#   "linear"    -> tabular regression
#   "ridge"     -> tabular regression
#   "rf"        -> tabular regression
#   "mlp"       -> tabular regression
#   "logistic"  -> classification
#   "vgg_face"  -> SCUT image regression
#
# DENSITY_MODELS:
#   "Neighbor", "Kernel"
#
# SCUT_TARGET:
#   "Average" or participant ratings "P1" through "P60"
#
# The methods are fixed paper baselines:
#   none, fair-reweighing, discretized-reweighing

PAPER_METHODS = ["none", "fair-reweighing", "discretized-reweighing"]
DATASETS = ["SCUT"]
MODELS = ["auto"]
DENSITY_MODELS = ["Neighbor"]
REPEAT = 1
SCUT_TARGET = "P5"
FIRST_SEED = 1
OUTPUT = Path("result/jair_runs.csv")
RESULT_COLUMNS = [
    "dataset",
    "model",
    "method",
    "density_model",
    "seed",
    "mse",
    "mae",
    "r2",
    "r_sep",
    "i_sep",
    "c_sep",
    "c_sep_xfit",
]
RUN_DETAIL_COLUMNS = [
    "fit_seconds",
    "total_seconds",
    "selected_radius",
    "selected_bandwidth",
]
SUMMARY_METRICS = [
    "mse",
    "mae",
    "r2",
    "r_sep",
    "i_sep",
    "c_sep",
    "c_sep_xfit",
]
SUMMARY_RUNTIME_METRICS = ["fit_seconds", "total_seconds"]

def main():
    result = run_grid()
    result = result[ordered_result_columns(result)]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT.with_name(OUTPUT.stem + "_summary.csv")
    correlation_path = OUTPUT.with_name(OUTPUT.stem + "_metric_correlations.csv")
    result.to_csv(OUTPUT, index=False)
    summarize_results(result).to_csv(summary_path, index=False)
    metric_correlations(result).to_csv(correlation_path, index=False)
    print(f"Wrote {len(result)} rows to {OUTPUT}, {summary_path}, and {correlation_path}")


def run_grid():
    rows = []
    scut_options = {"target": SCUT_TARGET}
    for dataset in DATASETS:
        task_type = task_type_for_dataset(dataset, FIRST_SEED, scut_options)
        for repeat_idx in range(REPEAT):
            seed = FIRST_SEED + repeat_idx
            for model in MODELS:
                for method in methods_for_dataset(dataset):
                    for density in densities_for_method(method, DENSITY_MODELS, task_type):
                        experiment = Experiment(
                            data=dataset,
                            regressor=model,
                            balance=method,
                            density_model=density,
                            seed=seed,
                            dataset_options=scut_options,
                        )
                        row = experiment.run()
                        rows.append(row)
    return pd.DataFrame(rows)


def methods_for_dataset(dataset):
    return PAPER_METHODS


def densities_for_method(method, density_models, task_type):
    if method.replace("_", "-").lower() in {"none", "discretized-reweighing"}:
        return ["Neighbor"]
    if task_type == "classification":
        return ["Reweighing"]
    return density_models


def task_type_for_dataset(dataset, seed, dataset_options):
    _, y, _ = load_dataset(dataset, seed=seed, **dataset_options)
    return "classification" if len(np.unique(y)) == 2 else "regression"


def summarize_results(result):
    group_cols = ["dataset", "model", "method", "density_model"]
    metrics = summary_metrics(result)
    rows = []
    for keys, group in result.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(group)
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def ordered_result_columns(result):
    per_sensitive = per_sensitive_metric_columns(result)
    columns = RESULT_COLUMNS + per_sensitive + RUN_DETAIL_COLUMNS
    return [column for column in columns if column in result.columns]


def summary_metrics(result):
    return [column for column in SUMMARY_METRICS + per_sensitive_metric_columns(result) + SUMMARY_RUNTIME_METRICS if column in result.columns]


def per_sensitive_metric_columns(result):
    prefixes = ("r_sep_", "i_sep_", "c_sep_", "c_sep_xfit_")
    base_metrics = set(RESULT_COLUMNS + RUN_DETAIL_COLUMNS)
    return sorted(
        column
        for column in result.columns
        if column.startswith(prefixes) and column not in base_metrics
    )


def metric_correlations(result):
    metric_pairs = [
        ("r_sep_gap", "i_sep"),
        ("r_sep_gap", "c_sep"),
        ("r_sep_gap", "c_sep_xfit"),
        ("i_sep", "c_sep"),
        ("i_sep", "c_sep_xfit"),
        ("c_sep", "c_sep_xfit"),
    ]
    rows = []
    for keys, group in result.groupby(["dataset", "model"], dropna=False):
        values = {
            "r_sep_gap": (group["r_sep"] - 1.0).abs(),
            "i_sep": group["i_sep"],
            "c_sep": group["c_sep"],
            "c_sep_xfit": group["c_sep_xfit"],
        }
        for left, right in metric_pairs:
            pair = pd.DataFrame({"left": values[left], "right": values[right]}).dropna()
            row = {
                "dataset": keys[0],
                "model": keys[1],
                "metric_x": left,
                "metric_y": right,
                "n_pairs": len(pair),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
            }
            if len(pair) > 1 and pair["left"].nunique() > 1 and pair["right"].nunique() > 1:
                pearson = pearsonr(pair["left"], pair["right"])
                spearman = spearmanr(pair["left"], pair["right"])
                row.update({
                    "pearson_r": pearson.statistic,
                    "pearson_p": pearson.pvalue,
                    "spearman_r": spearman.statistic,
                    "spearman_p": spearman.pvalue,
                })
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
