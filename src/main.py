import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiment import Experiment


PAPER_DATASETS = ["Synthetic", "LSAC", "Community", "Community_Con", "Insurance", "German", "Heart"]
DEFAULT_METHODS = ["none", "fair-reweighing"]
DEFAULT_DENSITIES = ["Neighbor"]


def main():
    args = parse_args()
    datasets = PAPER_DATASETS if args.all_paper_datasets else args.datasets
    rows, examples = run_grid(args, datasets)
    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    summary_output = Path(args.summary_output) if args.summary_output else output.with_name(output.stem + "_summary.csv")
    summarize_results(result).to_csv(summary_output, index=False)

    comparison_output = Path(args.comparison_output) if args.comparison_output else output.with_name(output.stem + "_comparisons.csv")
    compare_to_baseline(result).to_csv(comparison_output, index=False)

    if args.weight_examples_output and examples:
        weight_output = Path(args.weight_examples_output)
        weight_output.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(examples, ignore_index=True).to_csv(weight_output, index=False)

    print(f"Wrote raw results to {output}")
    print(f"Wrote summary results to {summary_output}")
    print(f"Wrote paired comparisons to {comparison_output}")


def run_grid(args, datasets):
    rows = []
    examples = []
    for dataset in datasets:
        for repeat_idx in range(args.repeat):
            seed = args.seed + repeat_idx
            for model in args.models:
                for method in args.methods:
                    for density in densities_for_method(method, args.density_models):
                        try:
                            experiment = Experiment(
                                data=dataset,
                                regressor=model,
                                balance=method,
                                density_model=density,
                                seed=seed,
                                test_size=args.test_size,
                                tune_density=args.tune_density,
                                radius_grid=args.radius_grid,
                                bandwidth_grid=args.bandwidth_grid,
                                n_bins=args.discretized_bins,
                                dataset_options=scut_options(args),
                            )
                        except ValueError as error:
                            print(f"Skipping dataset={dataset} model={model}: {error}")
                            continue
                        row = experiment.run()
                        rows.append(row)
                        if not experiment.weight_examples.empty:
                            examples.append(experiment.weight_examples)
                        print(
                            f"{dataset} seed={seed} model={row['model']} method={row['method']} "
                            f"density={row['density_model']} mse={row['mse']:.4f}"
                        )
    return rows, examples


def densities_for_method(method, density_models):
    method_key = method.replace("_", "-").lower()
    if method_key in {"none", "discretized", "discretized-reweighing", "original-reweighing", "groundtruth"}:
        return ["Neighbor"]
    return density_models


def summarize_results(result):
    group_cols = ["dataset", "model", "task_type", "method", "density_model"]
    metric_cols = numeric_metric_columns(result)
    rows = []
    for keys, group in result.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(group)
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_ci95"] = 1.96 * row[f"{metric}_std"] / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compare_to_baseline(result):
    metric_cols = numeric_metric_columns(result)
    rows = []
    index_cols = ["dataset", "model", "task_type", "seed"]
    method_cols = ["dataset", "model", "task_type", "method", "density_model"]
    baseline = result[result["method"] == "none"]
    treated = result[result["method"] != "none"]
    for keys, group in treated.groupby(method_cols, dropna=False):
        meta = dict(zip(method_cols, keys))
        base_key = {key: meta[key] for key in ["dataset", "model", "task_type"]}
        base = baseline
        for key, value in base_key.items():
            base = base[base[key] == value]
        merged = group.merge(base[index_cols + metric_cols], on=index_cols, suffixes=("", "_baseline"))
        for metric in metric_cols:
            current = pd.to_numeric(merged[metric], errors="coerce")
            prior = pd.to_numeric(merged[f"{metric}_baseline"], errors="coerce")
            paired = pd.DataFrame({"current": current, "baseline": prior}).dropna()
            if len(paired) == 0:
                continue
            diff = paired["current"] - paired["baseline"]
            p_value = np.nan
            if len(paired) > 1 and not np.allclose(diff, 0):
                try:
                    p_value = wilcoxon(paired["current"], paired["baseline"]).pvalue
                except ValueError:
                    p_value = np.nan
            effect = diff.mean() / diff.std(ddof=1) if len(diff) > 1 and diff.std(ddof=1) > 0 else np.nan
            row = dict(meta)
            row.update({
                "metric": metric,
                "n_pairs": len(paired),
                "mean_delta_vs_none": diff.mean(),
                "wilcoxon_p": p_value,
                "paired_cohens_d": effect,
            })
            rows.append(row)
    return pd.DataFrame(rows)


def numeric_metric_columns(result):
    exclude = {
        "seed",
        "train_size",
        "test_size",
        "selected_radius",
        "selected_bandwidth",
    }
    return [
        col
        for col in result.columns
        if col not in exclude
        and pd.api.types.is_numeric_dtype(result[col])
        and col not in {"sample_weight_applied"}
    ]


def scut_options(args):
    options = {}
    if args.scut_data_root:
        options["data_root"] = args.scut_data_root
    if args.scut_ratings_file:
        options["ratings_file"] = args.scut_ratings_file
    if args.scut_embeddings_file:
        options["embeddings_file"] = args.scut_embeddings_file
    if args.scut_landmark_dir:
        options["landmark_dir"] = args.scut_landmark_dir
    if args.scut_target:
        options["target"] = args.scut_target
    if args.scut_max_rows:
        options["max_rows"] = args.scut_max_rows
    return options


def parse_args():
    parser = argparse.ArgumentParser(description="Run reproducible FairReweighing JAIR experiments.")
    parser.add_argument("--datasets", nargs="+", default=["Synthetic"], help="Dataset names, e.g. Synthetic LSAC SCUT.")
    parser.add_argument("--all-paper-datasets", action="store_true", help="Run the non-SCUT paper datasets.")
    parser.add_argument("--models", nargs="+", default=["auto"], help="Model names: auto, linear, ridge, svr, rf, gbr, mlp, logistic.")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, help="Methods: none, fair-reweighing, discretized-reweighing.")
    parser.add_argument("--density-models", nargs="+", default=DEFAULT_DENSITIES, help="Density estimators for fair-reweighing: Neighbor Kernel.")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeated seeded runs.")
    parser.add_argument("--seed", type=int, default=1, help="First random seed.")
    parser.add_argument("--test-size", type=float, default=0.5, help="Held-out test fraction.")
    parser.add_argument("--tune-density", action="store_true", help="Select radius/bandwidth on the training split only.")
    parser.add_argument("--radius-grid", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5, 0.75, 1.0])
    parser.add_argument("--bandwidth-grid", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--discretized-bins", type=int, default=5, help="Quantile bins for original/discretized Reweighing.")
    parser.add_argument("--output", default="result/jair_runs.csv", help="Raw per-run CSV path.")
    parser.add_argument("--summary-output", default=None, help="Summary CSV path.")
    parser.add_argument("--comparison-output", default=None, help="Paired baseline comparison CSV path.")
    parser.add_argument("--weight-examples-output", default="result/jair_weight_examples.csv", help="High/low sample weight example CSV path.")
    parser.add_argument("--scut-data-root", default=None, help="SCUT root containing ImageExp/Selected_Ratings.csv and landmark_txt/.")
    parser.add_argument("--scut-ratings-file", default=None, help="Explicit SCUT ratings CSV.")
    parser.add_argument("--scut-embeddings-file", default=None, help="Optional Filename-keyed image embedding CSV.")
    parser.add_argument("--scut-landmark-dir", default=None, help="Optional SCUT landmark_txt directory.")
    parser.add_argument("--scut-target", default="Average", help="SCUT target column: Average, P1, P2, or P3.")
    parser.add_argument("--scut-max-rows", type=int, default=None, help="Optional SCUT subsample for quick smoke runs.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
