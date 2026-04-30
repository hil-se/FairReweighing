import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiment import Experiment


PAPER_METHODS = ["none", "fair-reweighing", "discretized-reweighing"]
DEFAULT_DENSITIES = ["Neighbor"]
FIRST_SEED = 1
OUTPUT = Path("result/jair_runs.csv")


def main():
    args = parse_args()
    result, examples = run_grid(args)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    summarize_results(result).to_csv(OUTPUT.with_name(OUTPUT.stem + "_summary.csv"), index=False)
    compare_to_baseline(result).to_csv(OUTPUT.with_name(OUTPUT.stem + "_comparisons.csv"), index=False)
    if examples:
        pd.concat(examples, ignore_index=True).to_csv(OUTPUT.with_name(OUTPUT.stem + "_weights.csv"), index=False)
    print(f"Wrote results to {OUTPUT}")


def run_grid(args):
    rows, examples = [], []
    scut_options = {"target": args.scut_target}
    for dataset in args.datasets:
        for repeat_idx in range(args.repeat):
            seed = FIRST_SEED + repeat_idx
            for model in args.models:
                for method in methods_for_dataset(dataset):
                    for density in densities_for_method(method, args.density_models):
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
                        if not experiment.weight_examples.empty:
                            examples.append(experiment.weight_examples)
                        print(
                            f"{dataset} seed={seed} model={row['model']} method={row['method']} "
                            f"density={row['density_model']} mse={row['mse']:.4f}"
                        )
    return pd.DataFrame(rows), examples


def methods_for_dataset(dataset):
    return PAPER_METHODS


def densities_for_method(method, density_models):
    if method.replace("_", "-").lower() in {"none", "discretized-reweighing"}:
        return ["Neighbor"]
    return density_models


def summarize_results(result):
    group_cols = ["dataset", "model", "task_type", "method", "density_model"]
    rows = []
    for keys, group in result.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_runs"] = len(group)
        for metric in numeric_metric_columns(result):
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95"] = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compare_to_baseline(result):
    rows = []
    index_cols = ["dataset", "model", "task_type", "seed"]
    method_cols = ["dataset", "model", "task_type", "method", "density_model"]
    metric_cols = numeric_metric_columns(result)
    baseline = result[result["method"] == "none"]
    for keys, treated in result[result["method"] != "none"].groupby(method_cols, dropna=False):
        meta = dict(zip(method_cols, keys))
        base = baseline
        for col in ["dataset", "model", "task_type"]:
            base = base[base[col] == meta[col]]
        merged = treated.merge(base[index_cols + metric_cols], on=index_cols, suffixes=("", "_baseline"))
        for metric in metric_cols:
            paired = merged[[metric, f"{metric}_baseline"]].dropna()
            if paired.empty:
                continue
            diff = paired[metric] - paired[f"{metric}_baseline"]
            p_value = np.nan
            if len(diff) > 1 and not np.allclose(diff, 0):
                p_value = wilcoxon(paired[metric], paired[f"{metric}_baseline"]).pvalue
            rows.append({
                **meta,
                "metric": metric,
                "n_pairs": len(paired),
                "mean_delta_vs_none": diff.mean(),
                "wilcoxon_p": p_value,
                "paired_cohens_d": diff.mean() / diff.std(ddof=1) if len(diff) > 1 and diff.std(ddof=1) > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def numeric_metric_columns(result):
    ignored = {"seed", "train_size", "test_size", "selected_radius", "selected_bandwidth", "sample_weight_applied"}
    return [col for col in result.columns if col not in ignored and pd.api.types.is_numeric_dtype(result[col])]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FairReweighing paper experiments.")
    parser.add_argument("--datasets", nargs="+", default=["Synthetic"], help="Dataset names, e.g. Synthetic LSAC SCUT.")
    parser.add_argument("--models", nargs="+", default=["auto"], help="auto, linear, ridge, rf, mlp, vgg_face, logistic.")
    parser.add_argument("--density-models", nargs="+", default=DEFAULT_DENSITIES, help="Neighbor or Kernel.")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--scut-target", default="Average", help="Average or participant column P1 through P60.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
