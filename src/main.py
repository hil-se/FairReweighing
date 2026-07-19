from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from data_reader import load_dataset
from experiment import Experiment


# Run configuration. Edit these constants, then run `python src/main.py`.
#
# DATASETS:
#   "Synthetic", "LSAC", "LSAC_Multi", "Community", "Community_Con",
#   "Community_Multi", "Insurance", "Insurance_Multi", "German", "Heart",
#   "SCUT_Sex", "SCUT_Multi"
#
# MODELS:
#   "linear"    -> tabular regression
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
#   none, fair-reweighing, reweighing, fairbalance, berk-pairwise,
#   calders-effect-control, chzhen-repair, agarwal-bgl

CORE_METHODS = ["none", "fair-reweighing"]
CLASSIFICATION_METHODS = ["none", "reweighing", "fair-reweighing", "fairbalance"]
REGRESSION_BASELINES = ["berk-pairwise", "calders-effect-control", "chzhen-repair", "agarwal-bgl"]
SCUT_METHODS = ["none", "fair-reweighing", "chzhen-repair"]
SCUT_DATASETS = {"scut-sex", "scut-multi"}
MULTI_SA_DATASETS = {"lsac-multi", "community-multi", "communities-multi", "insurance-multi", "scut-multi"}
DATASETS = [
    "Synthetic",
    "LSAC",
    "LSAC_Multi",
    "Community",
    "Community_Con",
    "Community_Multi",
    "Insurance",
    "Insurance_Multi",
    "German",
    "Heart",
]
REGRESSION_MODELS = ["linear"]
CLASSIFICATION_MODELS = ["logistic"]
DENSITY_MODELS = ["Neighbor", "Kernel"]
REPEAT = 20
SCUT_REPEAT = 20
SCUT_TARGET = "P1"
FIRST_SEED = 1
OUTPUT_DIR = Path("result")
RAW_OUTPUT_DIR = OUTPUT_DIR / "raw"
MAIN_OUTPUT_DIR = OUTPUT_DIR / "main"
RESULT_COLUMNS = [
    "dataset",
    "model",
    "method",
    "density_model",
    "accuracy",
    "f1",
    "mse",
    "mae",
    "r2",
    "aod",
    "eod",
    "r_sep",
    "i_sep",
    "c_sep_xfit",
]
RUN_DETAIL_COLUMNS = [
    "seed",
    "fit_seconds",
    "total_seconds",
    "selected_radius",
    "selected_bandwidth",
]
PER_SENSITIVE_PREFIXES = (
    "c_sep_xfit_",
    "r_sep_",
    "i_sep_",
    "aod_",
    "eod_",
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    raw_paths = []
    for dataset, result in run_grid().items():
        raw_result = result[ordered_result_columns(
            result,
            include_per_sensitive=has_multiple_sensitive_attributes(result),
            include_run_details=True,
        )]
        raw_path = RAW_OUTPUT_DIR / f"{safe_file_name(dataset)}.csv"
        raw_result.to_csv(raw_path, index=False)
        raw_paths.append(raw_path)
        print(f"Wrote {len(raw_result)} raw rows to {raw_path}")

        result = average_trials(result)
        result = result[ordered_result_columns(
            result,
            include_per_sensitive=has_multiple_sensitive_attributes(result),
        )]
        path = MAIN_OUTPUT_DIR / f"{safe_file_name(dataset)}.csv"
        result.to_csv(path, index=False)
        paths.append(path)
        print(f"Wrote {len(result)} rows to {path}")

    stats_path = MAIN_OUTPUT_DIR / "stat_tests.csv"
    statistical_tests(raw_paths).to_csv(stats_path, index=False)
    print(f"Wrote statistical tests to {stats_path}")

    pairwise_stats_path = MAIN_OUTPUT_DIR / "stat_tests_pairwise.csv"
    statistical_tests_pairwise(raw_paths).to_csv(pairwise_stats_path, index=False)
    print(f"Wrote pairwise statistical tests to {pairwise_stats_path}")

    summary_path = MAIN_OUTPUT_DIR / "summary.csv"
    summarize_dataset_files(result_paths_for_summary()).to_csv(summary_path, index=False)
    print(f"Wrote summary to {summary_path}")
    print(f"Finished {len(paths)} dataset file(s).")


def run_grid():
    results = {}
    scut_options = {"target": SCUT_TARGET}
    for dataset in DATASETS:
        rows = []
        task_type = task_type_for_dataset(dataset, FIRST_SEED, scut_options)
        for repeat_idx in range(repeats_for_dataset(dataset)):
            seed = FIRST_SEED + repeat_idx
            for model in models_for_dataset(dataset, task_type):
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
    dataset_key = str(dataset).replace("_", "-").lower()
    if task_type == "classification":
        return CLASSIFICATION_METHODS
    if dataset_key == "synthetic":
        return CORE_METHODS + ["groundtruth"] + REGRESSION_BASELINES
    if is_scut_dataset(dataset):
        return SCUT_METHODS
    if is_multi_sensitive_dataset(dataset):
        return CORE_METHODS + REGRESSION_BASELINES
    return CORE_METHODS + REGRESSION_BASELINES


def repeats_for_dataset(dataset):
    return SCUT_REPEAT if is_scut_dataset(dataset) else REPEAT


def models_for_dataset(dataset, task_type):
    if is_scut_dataset(dataset):
        return ["vgg_face"]
    if task_type == "classification":
        return CLASSIFICATION_MODELS
    return REGRESSION_MODELS


def is_scut_dataset(dataset):
    return str(dataset).replace("_", "-").lower() in SCUT_DATASETS


def is_multi_sensitive_dataset(dataset):
    return str(dataset).replace("_", "-").lower() in MULTI_SA_DATASETS


def densities_for_method(method, density_models, task_type):
    method_key = method.replace("_", "-").lower()
    if method_key == "reweighing":
        return ["Reweighing"]
    if method_key == "fairbalance":
        return ["FairBalance"]
    if method_key != "fair-reweighing":
        return ["Neighbor"]
    return density_models


def task_type_for_dataset(dataset, seed, dataset_options):
    _, y, _ = load_dataset(dataset, seed=seed, **dataset_options)
    return "classification" if len(np.unique(y)) == 2 else "regression"


def average_trials(result):
    group_cols = ["dataset", "model", "method", "density_model"]
    value_cols = [
        column
        for column in ordered_result_columns(result, include_run_details=False)
        if column not in group_cols
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
        result = read_result_csv(path)
        required_columns = {"dataset", "model", "method", "mse", "r2", "c_sep_xfit"}
        if not required_columns.issubset(result.columns):
            continue
        for (dataset, model), group in result.groupby(["dataset", "model"], dropna=False):
            baseline_rows = group[group["method"] == "none"]
            if baseline_rows.empty:
                continue
            baseline = baseline_rows.iloc[0]
            best_fairness = group.loc[group["c_sep_xfit"].idxmin()]
            rows.append({
                "dataset": dataset,
                "model": model,
                "baseline_mse": baseline["mse"],
                "baseline_r2": baseline["r2"],
                "baseline_c_sep_xfit": baseline["c_sep_xfit"],
                "best_fairness_method": best_fairness["method"],
                "best_fairness_density": best_fairness["density_model"],
                "best_fairness_mse": best_fairness["mse"],
                "best_fairness_r2": best_fairness["r2"],
                "best_fairness_c_sep_xfit": best_fairness["c_sep_xfit"],
                "c_sep_xfit_reduction_pct": percent_reduction(baseline["c_sep_xfit"], best_fairness["c_sep_xfit"]),
                "mse_change_pct_at_best_fairness": percent_change(baseline["mse"], best_fairness["mse"]),
            })
    return pd.DataFrame(rows)


def result_paths_for_summary():
    ignored = {"summary.csv", "stat_tests.csv", "stat_tests_pairwise.csv"}
    return [
        path
        for path in MAIN_OUTPUT_DIR.glob("*.csv")
        if path.name not in ignored
    ]


def statistical_tests(raw_paths):
    rows = []
    for path in sorted(raw_paths):
        raw = read_result_csv(path)
        if raw.empty or "seed" not in raw.columns:
            continue
        for (dataset, model), group in raw.groupby(["dataset", "model"], dropna=False):
            fair_density = best_fairreweighing_density(group)
            if fair_density is None:
                continue
            fair = group[
                (group["method"] == "fair-reweighing")
                & (group["density_model"] == fair_density)
            ]
            baselines = group[group["method"] != "fair-reweighing"]
            for (method, density_model), baseline in baselines.groupby(["method", "density_model"], dropna=False):
                for metric, source_column in test_metrics(group):
                    metric_rows = paired_metric_values(baseline, fair, source_column, metric)
                    if metric_rows is None:
                        continue
                    baseline_values, fair_values = metric_rows
                    test = paired_wilcoxon(baseline_values, fair_values)
                    delta = cliffs_delta(fair_values, baseline_values)
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "fair_density": fair_density,
                        "baseline_method": method,
                        "baseline_density": density_model,
                        "metric": metric,
                        "n": len(fair_values),
                        "baseline_mean": float(np.mean(baseline_values)),
                        "fairreweighing_mean": float(np.mean(fair_values)),
                        "wilcoxon_stat": test["statistic"],
                        "p_value": test["p_value"],
                        "cliffs_delta": delta,
                        "effect_size": cliffs_delta_size(delta),
                        "interpretation": test_interpretation(
                            baseline_values,
                            fair_values,
                            test["p_value"],
                            delta,
                        ),
                    })
    return pd.DataFrame(rows)


def statistical_tests_pairwise(raw_paths):
    rows = []
    for path in sorted(raw_paths):
        raw = read_result_csv(path)
        if raw.empty or "seed" not in raw.columns or "c_sep_xfit" not in raw.columns:
            continue
        for (dataset, model), group in raw.groupby(["dataset", "model"], dropna=False):
            treatments = treatment_series(group)
            if len(treatments) < 2:
                continue
            ranks = recursive_ranks(treatments)
            for row_key, row_values in treatments.items():
                for column_key, column_values in treatments.items():
                    if row_key == column_key:
                        continue
                    paired = paired_by_seed(row_values, column_values)
                    if len(paired) < 5:
                        continue
                    row_metric = paired["row"].to_numpy(dtype=float)
                    column_metric = paired["column"].to_numpy(dtype=float)
                    test = paired_wilcoxon(row_metric, column_metric)
                    delta = cliffs_delta(row_metric, column_metric)
                    rows.append({
                        "dataset": dataset,
                        "model": model,
                        "row_treatment": row_key[0],
                        "row_density": row_key[1],
                        "column_treatment": column_key[0],
                        "column_density": column_key[1],
                        "metric": "c_sep_xfit",
                        "n": len(row_metric),
                        "row_mean": float(np.mean(row_metric)),
                        "column_mean": float(np.mean(column_metric)),
                        "wilcoxon_stat": test["statistic"],
                        "p_value_two_sided": test["p_value"],
                        "cliffs_delta": delta,
                        "effect_size": cliffs_delta_size(delta),
                        "row_rank": ranks[row_key],
                        "column_rank": ranks[column_key],
                    })
    return pd.DataFrame(rows)


def treatment_series(group):
    treatments = {}
    for key, treatment in group.groupby(["method", "density_model"], dropna=False):
        values = treatment[["seed", "c_sep_xfit"]].copy()
        values["c_sep_xfit"] = pd.to_numeric(values["c_sep_xfit"], errors="coerce")
        values = values.dropna()
        if not values.empty:
            treatments[key] = values
    return treatments


def paired_by_seed(row_values, column_values):
    paired = row_values.merge(column_values, on="seed", suffixes=("_row", "_column"))
    paired = paired.dropna()
    return paired.rename(
        columns={
            "c_sep_xfit_row": "row",
            "c_sep_xfit_column": "column",
        }
    )


def recursive_ranks(treatments):
    remaining = sorted(
        treatments,
        key=lambda key: (
            float(treatments[key]["c_sep_xfit"].mean()),
            treatment_display_name(*key),
        ),
    )
    ranks = {}
    rank = 0
    while remaining:
        reference = remaining[0]
        ranks[reference] = rank
        next_remaining = []
        reference_values = treatments[reference]
        found_next_rank = False
        for candidate in remaining[1:]:
            candidate_values = treatments[candidate]
            paired = paired_by_seed(candidate_values, reference_values)
            if len(paired) < 5:
                worse_than_reference = False
            else:
                candidate_metric = paired["row"].to_numpy(dtype=float)
                reference_metric = paired["column"].to_numpy(dtype=float)
                test = paired_wilcoxon(candidate_metric, reference_metric)
                delta = cliffs_delta(candidate_metric, reference_metric)
                worse_than_reference = test["p_value"] < 0.05 and delta > 0.33
            if found_next_rank or worse_than_reference:
                found_next_rank = True
                next_remaining.append(candidate)
            else:
                ranks[candidate] = rank
        remaining = next_remaining
        rank += 1
    return ranks


def best_fairreweighing_density(group):
    fair = group[group["method"] == "fair-reweighing"]
    if fair.empty:
        return None
    means = fair.groupby("density_model", dropna=False)["c_sep_xfit"].mean(numeric_only=True)
    if means.empty:
        return None
    return means.idxmin()


def test_metrics(group):
    metrics = [("c_sep_xfit", "c_sep_xfit")]
    if "i_sep" in group.columns and pd.to_numeric(group["i_sep"], errors="coerce").notna().any():
        metrics.append(("i_sep", "i_sep"))
    if "r_sep" in group.columns and pd.to_numeric(group["r_sep"], errors="coerce").notna().any():
        metrics.append(("|r_sep-1|", "r_sep"))
    return metrics


def paired_metric_values(baseline, fair, source_column, metric):
    if source_column not in baseline.columns or source_column not in fair.columns:
        return None
    baseline_values = baseline[["seed", source_column]].copy()
    fair_values = fair[["seed", source_column]].copy()
    baseline_values[source_column] = pd.to_numeric(baseline_values[source_column], errors="coerce")
    fair_values[source_column] = pd.to_numeric(fair_values[source_column], errors="coerce")
    paired = baseline_values.merge(fair_values, on="seed", suffixes=("_baseline", "_fair"))
    paired = paired.dropna()
    if len(paired) < 5:
        return None
    baseline_metric = paired[f"{source_column}_baseline"].to_numpy(dtype=float)
    fair_metric = paired[f"{source_column}_fair"].to_numpy(dtype=float)
    if metric == "|r_sep-1|":
        baseline_metric = np.abs(baseline_metric - 1.0)
        fair_metric = np.abs(fair_metric - 1.0)
    return baseline_metric, fair_metric


def paired_wilcoxon(baseline_values, fair_values):
    differences = np.asarray(baseline_values) - np.asarray(fair_values)
    if np.allclose(differences, 0.0):
        return {"statistic": 0.0, "p_value": 1.0}
    result = wilcoxon(baseline_values, fair_values, alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def cliffs_delta(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    comparisons = np.sign(xs[:, None] - ys[None, :])
    return float(np.sum(comparisons) / comparisons.size)


def cliffs_delta_size(delta):
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def test_interpretation(baseline_values, fair_values, p_value, delta):
    fair_mean = np.mean(fair_values)
    baseline_mean = np.mean(baseline_values)
    if np.isclose(fair_mean, baseline_mean):
        direction = "same"
    else:
        direction = "lower" if fair_mean < baseline_mean else "higher"
    size = cliffs_delta_size(delta)
    if direction == "same":
        return f"no difference ({size})"
    if p_value < 0.05 and direction == "lower":
        return f"FairReweighing lower ({size})"
    if p_value < 0.05:
        return f"FairReweighing higher ({size})"
    return f"not significant; FairReweighing {direction} on average ({size})"


def treatment_display_name(method, density):
    method_key = str(method).strip().lower()
    density_key = str(density).strip()
    names = {
        "none": "None",
        "fair-reweighing": f"FR-{density_key[0].upper()}" if density_key else "FR",
        "reweighing": "Rew",
        "fairbalance": "FairBalance",
        "fairicp": "FairICP",
        "groundtruth": "GT",
        "berk-pairwise": "Berk",
        "calders-effect-control": "Calders",
        "chzhen-repair": "Chzhen",
        "agarwal-bgl": "Agarwal",
    }
    return names.get(method_key, str(method))


def read_result_csv(path):
    result = pd.read_csv(path)
    result.columns = [str(column).strip() for column in result.columns]
    for column in ("dataset", "model", "method", "density_model"):
        if column in result.columns:
            result[column] = result[column].astype(str).str.strip()
    return result


def percent_reduction(baseline, value):
    if not np.isfinite(baseline) or baseline == 0:
        return np.nan
    return 100 * (baseline - value) / baseline


def percent_change(baseline, value):
    if not np.isfinite(baseline) or baseline == 0:
        return np.nan
    return 100 * (value - baseline) / baseline


def ordered_result_columns(result, include_per_sensitive=True, include_run_details=False):
    per_sensitive = per_sensitive_metric_columns(result) if include_per_sensitive else []
    run_details = RUN_DETAIL_COLUMNS if include_run_details else []
    columns = RESULT_COLUMNS + per_sensitive + run_details
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
