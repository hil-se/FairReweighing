from pathlib import Path

import numpy as np
import pandas as pd

from data_reader import load_dataset
from experiment import Experiment


# Run configuration. Edit these constants, then run `python src/main.py`.
#
# DATASETS:
#   "Synthetic", "LSAC", "Community", "Community_Con", "Insurance",
#   "German", "Heart", "SCUT"
#
# MODELS:
#   "linear"    -> tabular regression
#   "logistic"  -> classification
#
# DENSITY_MODELS:
#   "Neighbor", "Kernel"
#
# SCUT_TARGET:
#   "Average" or participant ratings "P1" through "P60"
#
# The methods are fixed paper baselines:
#   none, fair-reweighing, berk-pairwise, calders-effect-control,
#   chzhen-repair, agarwal-bgl

CORE_METHODS = ["none", "fair-reweighing"]
REGRESSION_BASELINES = ["berk-pairwise", "calders-effect-control", "chzhen-repair", "agarwal-bgl"]
DATASETS = ["SCUT"]
REGRESSION_MODELS = ["vgg_face"]
CLASSIFICATION_MODELS = ["logistic"]
DENSITY_MODELS = ["Neighbor", "Kernel"]
REPEAT = 20
SCUT_TARGET = "P10"
FIRST_SEED = 1
OUTPUT_DIR = Path("result")
RESULT_COLUMNS = [
    "dataset",
    "model",
    "method",
    "density_model",
    "mse",
    "mae",
    "r2",
    "r_sep",
    "i_sep",
    "c_sep_xfit",
    "pred_pearson_abs",
    "pred_spearman_abs",
    "pred_mean_gap",
    "residual_mean_gap",
    "group_mse_gap",
    "wasserstein_pred_gap",
]
RUN_DETAIL_COLUMNS = [
    "fit_seconds",
    "total_seconds",
    "selected_radius",
    "selected_bandwidth",
]
PER_SENSITIVE_PREFIXES = (
    "c_sep_xfit_",
    "pred_pearson_abs_",
    "pred_spearman_abs_",
    "pred_mean_gap_",
    "residual_mean_gap_",
    "group_mse_gap_",
    "wasserstein_pred_gap_",
    "r_sep_",
    "i_sep_",
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for dataset, result in run_grid().items():
        result = average_trials(result)
        result = result[ordered_result_columns(result, include_per_sensitive=has_multiple_sensitive_attributes(result))]
        path = OUTPUT_DIR / f"{safe_file_name(dataset)}.csv"
        result.to_csv(path, index=False)
        paths.append(path)
        print(f"Wrote {len(result)} rows to {path}")
    summary_path = OUTPUT_DIR / "summary.csv"
    summarize_dataset_files(paths).to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")
    print(f"Finished {len(paths)} dataset file(s).")


def run_grid():
    results = {}
    scut_options = {"target": SCUT_TARGET}
    for dataset in DATASETS:
        rows = []
        task_type = task_type_for_dataset(dataset, FIRST_SEED, scut_options)
        for repeat_idx in range(REPEAT):
            seed = FIRST_SEED + repeat_idx
            for model in models_for_task(task_type):
                for method in methods_for_dataset(dataset, task_type):
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
        results[dataset] = pd.DataFrame(rows)
    return results


def methods_for_dataset(dataset, task_type):
    if str(dataset).lower() in {"scut", "scut-fbp5500"} or task_type == "classification":
        return CORE_METHODS
    return CORE_METHODS + REGRESSION_BASELINES


def models_for_task(task_type):
    if task_type == "classification":
        return CLASSIFICATION_MODELS
    return REGRESSION_MODELS


def densities_for_method(method, density_models, task_type):
    method_key = method.replace("_", "-").lower()
    if method_key != "fair-reweighing":
        return ["Neighbor"]
    if task_type == "classification":
        return ["Reweighing"]
    return density_models


def task_type_for_dataset(dataset, seed, dataset_options):
    _, y, _ = load_dataset(dataset, seed=seed, **dataset_options)
    return "classification" if len(np.unique(y)) == 2 else "regression"


def average_trials(result):
    group_cols = ["dataset", "model", "method", "density_model"]
    value_cols = [
        column
        for column in ordered_result_columns(result)
        if column not in group_cols and column != "seed"
    ]
    rows = []
    for keys, group in result.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        for column in value_cols:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[column] = values.mean() if not values.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_dataset_files(paths):
    rows = []
    for path in sorted(paths):
        result = pd.read_csv(path)
        for (dataset, model), group in result.groupby(["dataset", "model"], dropna=False):
            baseline_rows = group[group["method"] == "none"]
            if baseline_rows.empty:
                continue
            baseline = baseline_rows.iloc[0]
            best_accuracy = group.loc[group["mse"].idxmin()]
            best_fairness = group.loc[group["c_sep_xfit"].idxmin()]
            rows.append({
                "dataset": dataset,
                "model": model,
                "baseline_mse": baseline["mse"],
                "baseline_r2": baseline["r2"],
                "baseline_c_sep_xfit": baseline["c_sep_xfit"],
                "best_accuracy_method": best_accuracy["method"],
                "best_accuracy_density": best_accuracy["density_model"],
                "best_accuracy_mse": best_accuracy["mse"],
                "best_accuracy_r2": best_accuracy["r2"],
                "best_fairness_method": best_fairness["method"],
                "best_fairness_density": best_fairness["density_model"],
                "best_fairness_c_sep_xfit": best_fairness["c_sep_xfit"],
                "c_sep_xfit_reduction_pct": percent_reduction(baseline["c_sep_xfit"], best_fairness["c_sep_xfit"]),
                "best_fairness_mse": best_fairness["mse"],
                "best_fairness_r2": best_fairness["r2"],
                "mse_change_pct_at_best_fairness": percent_change(baseline["mse"], best_fairness["mse"]),
            })
    return pd.DataFrame(rows)


def percent_reduction(baseline, value):
    if not np.isfinite(baseline) or baseline == 0:
        return np.nan
    return 100 * (baseline - value) / baseline


def percent_change(baseline, value):
    if not np.isfinite(baseline) or baseline == 0:
        return np.nan
    return 100 * (value - baseline) / baseline


def ordered_result_columns(result, include_per_sensitive=True):
    per_sensitive = per_sensitive_metric_columns(result) if include_per_sensitive else []
    columns = RESULT_COLUMNS + per_sensitive + RUN_DETAIL_COLUMNS
    return [column for column in columns if column in result.columns]


def has_multiple_sensitive_attributes(result):
    return len(sensitive_names_in_result(result)) > 1


def per_sensitive_metric_columns(result):
    base_metrics = set(RESULT_COLUMNS + RUN_DETAIL_COLUMNS)
    return sorted(
        column
        for column in result.columns
        if column.startswith(PER_SENSITIVE_PREFIXES) and column not in base_metrics
    )


def sensitive_names_in_result(result):
    names = set()
    for column in per_sensitive_metric_columns(result):
        for prefix in PER_SENSITIVE_PREFIXES:
            if column.startswith(prefix):
                names.add(column[len(prefix):])
                break
    return names


def safe_file_name(name):
    return str(name).replace(" ", "_").replace("/", "_")


if __name__ == "__main__":
    main()
