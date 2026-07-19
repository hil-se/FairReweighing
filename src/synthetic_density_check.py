from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiment import Experiment
from synthetic_config import P_A1, synthetic_ground_truth_weight as analytic_synthetic_weight
from synthetic_weight_experiment import WeightComputer


P_SEX_1 = P_A1
RANK_METRICS = ("cosine_similarity", "pearson_correlation")
SYNTHETIC_SEEDS = range(1, 21)
PARAMETER_GRID = [round(value, 1) for value in np.arange(0.1, 1.01, 0.1)]
OUTPUT_DIR = Path("result") / "synthetic"
FIGURE_OUTPUT_DIR = Path("result") / "figures"
RAW_OUTPUT = OUTPUT_DIR / "synthetic_density_error_raw.csv"
WEIGHT_OUTPUT = OUTPUT_DIR / "synthetic_weight_scatter_raw.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "synthetic_density_error.csv"
PARAMETER_RAW_OUTPUT = OUTPUT_DIR / "synthetic_weight_parameter_raw.csv"
PARAMETER_SUMMARY_OUTPUT = OUTPUT_DIR / "synthetic_weight_parameter_summary.csv"
PARAMETER_PAIRWISE_OUTPUT = OUTPUT_DIR / "synthetic_weight_parameter_pairwise.csv"
SCATTER_PDF = FIGURE_OUTPUT_DIR / "synthetic_weight_scatter.pdf"
SCATTER_PNG = FIGURE_OUTPUT_DIR / "synthetic_weight_scatter.png"
PARAMETER_RANK_PDF = FIGURE_OUTPUT_DIR / "synthetic_weight_parameter_rank.pdf"
PARAMETER_RANK_PNG = FIGURE_OUTPUT_DIR / "synthetic_weight_parameter_rank.png"
WHITE_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "white_density",
    ["#ffffff", "#f8fafc", "#edf2f7", "#d8e2ec", "#9fb3c8"],
)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    parameter_raw = run_parameter_grid()
    parameter_raw.to_csv(PARAMETER_RAW_OUTPUT, index=False)
    dynamic_parameters = {
        (int(row.seed), str(row.density_model)): float(row.parameter_value)
        for row in parameter_raw[parameter_raw["parameter_mode"].eq("dynamic")].itertuples()
    }

    rows = []
    weight_rows = []
    for seed in SYNTHETIC_SEEDS:
        for density_model in ("Neighbor", "Kernel"):
            summary_row, seed_weight_rows = run_one(
                seed,
                density_model,
                dynamic_parameters[(seed, density_model)],
            )
            rows.append(summary_row)
            weight_rows.append(seed_weight_rows)

    raw = pd.DataFrame(rows)
    raw.to_csv(RAW_OUTPUT, index=False)
    weights = pd.concat(weight_rows, ignore_index=True)
    weights.to_csv(WEIGHT_OUTPUT, index=False)
    summary = summarize(raw)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    plot_weight_scatter(weights, raw)
    parameter_summary, parameter_pairwise = rank_parameter_grid(parameter_raw)
    parameter_summary.to_csv(PARAMETER_SUMMARY_OUTPUT, index=False)
    parameter_pairwise.to_csv(PARAMETER_PAIRWISE_OUTPUT, index=False)
    plot_weight_rank(parameter_summary)
    print(f"Wrote {len(raw)} seed-level rows to {RAW_OUTPUT}")
    print(f"Wrote {len(weights)} per-sample weight rows to {WEIGHT_OUTPUT}")
    print(f"Wrote {len(summary)} summary rows to {SUMMARY_OUTPUT}")
    print(f"Wrote weight scatter plots to {SCATTER_PDF} and {SCATTER_PNG}")
    print(f"Wrote {len(parameter_raw)} parameter-grid rows to {PARAMETER_RAW_OUTPUT}")
    print(f"Wrote {len(parameter_summary)} parameter summary rows to {PARAMETER_SUMMARY_OUTPUT}")
    print(f"Wrote {len(parameter_pairwise)} parameter pairwise rows to {PARAMETER_PAIRWISE_OUTPUT}")
    print(f"Wrote parameter-ablation plots to {PARAMETER_RANK_PDF} and {PARAMETER_RANK_PNG}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def run_one(seed, density_model, selected_param):
    experiment = Experiment(
        data="Synthetic",
        regressor="linear",
        balance="fair-reweighing",
        density_model=density_model,
        seed=seed,
    )
    experiment.train_test_split()
    experiment.preprocess(experiment.X_train)

    A = experiment.X_train[experiment.protected].to_numpy()
    y = np.asarray(experiment.y_train, dtype=float).reshape(-1, 1)
    true_weight = synthetic_ground_truth_weight(A, y)
    estimated_weight = WeightComputer(A, y).get(density_model, selected_param)
    summary = {
        "seed": seed,
        "density_model": density_model,
        "selected_param": selected_param,
        "cosine_similarity": cosine_similarity(estimated_weight, true_weight),
        "pearson_correlation": pearson_correlation(estimated_weight, true_weight),
        "true_weight_min": float(np.min(true_weight)),
        "true_weight_max": float(np.max(true_weight)),
        "estimated_weight_min": float(np.min(estimated_weight)),
        "estimated_weight_max": float(np.max(estimated_weight)),
    }
    sample_rows = pd.DataFrame({
        "seed": seed,
        "density_model": density_model,
        "selected_param": selected_param,
        "w_true": np.asarray(true_weight, dtype=float).reshape(-1),
        "w_estimated": np.asarray(estimated_weight, dtype=float).reshape(-1),
    })
    return summary, sample_rows


def run_parameter_grid():
    rows = []
    for seed in SYNTHETIC_SEEDS:
        for density_model in ("Neighbor", "Kernel"):
            experiment = prepare_experiment(seed, density_model)
            A = experiment.X_train[experiment.protected].to_numpy()
            y = np.asarray(experiment.y_train, dtype=float).reshape(-1, 1)
            true_weight = synthetic_ground_truth_weight(A, y)
            weight_computer = WeightComputer(A, y)
            for parameter in PARAMETER_GRID:
                rows.append(parameter_row(seed, density_model, "fixed", parameter, weight_computer, true_weight))
            dynamic_parameter = experiment.select_density_param()
            rows.append(parameter_row(seed, density_model, "dynamic", dynamic_parameter, weight_computer, true_weight))
    return pd.DataFrame(rows)


def prepare_experiment(seed, density_model):
    experiment = Experiment(
        data="Synthetic",
        regressor="linear",
        balance="fair-reweighing",
        density_model=density_model,
        seed=seed,
    )
    experiment.train_test_split()
    experiment.preprocess(experiment.X_train)
    return experiment


def parameter_row(seed, density_model, parameter_mode, parameter, weight_computer, true_weight):
    estimated_weight = weight_computer.get(density_model, parameter)
    return {
        "seed": seed,
        "density_model": density_model,
        "parameter_mode": parameter_mode,
        "parameter_label": "Dynamic" if parameter_mode == "dynamic" else f"{float(parameter):.1f}",
        "parameter_value": float(parameter),
        "cosine_similarity": cosine_similarity(estimated_weight, true_weight),
        "pearson_correlation": pearson_correlation(estimated_weight, true_weight),
        "n_samples": int(len(true_weight)),
    }

def synthetic_ground_truth_weight(A, y):
    return analytic_synthetic_weight(A, y)


def normalize_weight(weight):
    weight = np.asarray(weight, dtype=float).reshape(-1)
    total = np.sum(weight)
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(weight))
    return len(weight) * weight / total


def cosine_similarity(xs, ys):
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    denominator = np.linalg.norm(xs) * np.linalg.norm(ys)
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return float(np.dot(xs, ys) / denominator)


def pearson_correlation(xs, ys):
    xs = np.asarray(xs, dtype=float).reshape(-1)
    ys = np.asarray(ys, dtype=float).reshape(-1)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs = xs[mask]
    ys = ys[mask]
    if len(xs) < 2 or np.allclose(xs, xs[0]) or np.allclose(ys, ys[0]):
        return np.nan
    return float(np.corrcoef(xs, ys)[0, 1])


def plot_weight_scatter(weights, raw):
    point_colors = {"Neighbor": "#2563eb", "Kernel": "#dc2626"}
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for axis, density_model in zip(axes, ("Neighbor", "Kernel")):
        axis.set_facecolor("white")
        subset = weights[weights["density_model"].eq(density_model)].copy()
        x = np.asarray(subset["w_true"], dtype=float)
        y = np.asarray(subset["w_estimated"], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]
        rng = np.random.default_rng(2026)
        plot_n = min(len(x), 20000)
        idx = rng.choice(len(x), size=plot_n, replace=False)
        axis.scatter(
            x[idx],
            y[idx],
            s=5,
            alpha=0.16,
            color=point_colors[density_model],
            edgecolors="none",
            rasterized=True,
        )
        lower = max(min(x.min(), y.min()), 1e-4)
        upper = max(x.max(), y.max())
        axis.plot([lower, upper], [lower, upper], color="black", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"Dynamic {density_model}")
        axis.set_xlabel(r"Ground-truth weight $w^*$")
        stats = raw[raw["density_model"].eq(density_model)]
        mean_pearson = stats["pearson_correlation"].mean()
        mean_cosine = stats["cosine_similarity"].mean()
        axis.text(
            0.04,
            0.96,
            f"Mean Pearson={mean_pearson:.3f}\nMean cosine={mean_cosine:.3f}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    axes[0].set_ylabel(r"Estimated weight $\hat{w}$")
    fig.suptitle("Dynamic FairReweighing weights vs. synthetic ground truth", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(SCATTER_PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(SCATTER_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def summarize(raw):
    metrics = [
        "selected_param",
        "cosine_similarity",
        "pearson_correlation",
        "true_weight_min",
        "true_weight_max",
        "estimated_weight_min",
        "estimated_weight_max",
    ]
    rows = []
    for density_model, group in raw.groupby("density_model", sort=False):
        row = {"density_model": density_model, "seeds": len(group)}
        for metric in metrics:
            row[metric] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def rank_parameter_grid(raw):
    summary_rows = []
    pairwise_rows = []
    for density_model, group in raw.groupby("density_model", sort=False):
        sources = {
            key: treatment.copy()
            for key, treatment in group.groupby(["parameter_mode", "parameter_label"], sort=False)
        }
        metric_results = {}
        for metric in RANK_METRICS:
            treatments = {
                key: treatment[["seed", metric]].copy()
                for key, treatment in sources.items()
            }
            ranks = recursive_high_ranks(treatments, metric)
            reference_key = rank_zero_reference(treatments, ranks, metric)
            metric_results[metric] = (treatments, ranks, reference_key)

        for key, source in sources.items():
            row = {
                "density_model": density_model,
                "parameter_mode": key[0],
                "parameter_label": key[1],
                "parameter_value": float(source["parameter_value"].mean()),
                "plot_order": plot_order(key[0], key[1]),
                "seeds": int(source["seed"].nunique()),
                "pearson_correlation": float(source["pearson_correlation"].mean()),
                "pearson_correlation_std": float(source["pearson_correlation"].std(ddof=1)),
                "cosine_similarity": float(source["cosine_similarity"].mean()),
                "cosine_similarity_std": float(source["cosine_similarity"].std(ddof=1)),
            }
            for metric, (treatments, ranks, reference_key) in metric_results.items():
                treatment = treatments[key]
                paired = paired_by_seed(treatment, treatments[reference_key], metric)
                candidate_values = paired["row"].to_numpy(dtype=float)
                reference_values = paired["column"].to_numpy(dtype=float)
                test = paired_wilcoxon(candidate_values, reference_values)
                delta = cliffs_delta(candidate_values, reference_values)
                row[f"{metric}_rank"] = int(ranks[key])
                row[f"{metric}_top_rank"] = int(ranks[key]) == 0
                row[f"{metric}_rank0_reference"] = reference_key[1]
                row[f"{metric}_rank0_reference_mean"] = float(treatments[reference_key][metric].mean())
                pairwise_rows.append({
                    "density_model": density_model,
                    "metric": metric,
                    "parameter_mode": key[0],
                    "parameter_label": key[1],
                    "candidate_rank": int(ranks[key]),
                    "candidate_mean": float(source[metric].mean()),
                    "reference_parameter_label": reference_key[1],
                    "reference_mean": float(treatments[reference_key][metric].mean()),
                    "n": int(len(paired)),
                    "wilcoxon_stat": test["statistic"],
                    "p_value": test["p_value"],
                    "cliffs_delta": delta,
                    "conclusion": "rank 0" if int(ranks[key]) == 0 else f"rank {int(ranks[key])}",
                })
            summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["density_model", "plot_order"])
    pairwise = pd.DataFrame(pairwise_rows).sort_values(["density_model", "metric", "parameter_label"])
    return summary, pairwise


def recursive_high_ranks(treatments, metric):
    remaining = sorted(
        treatments,
        key=lambda key: (
            -float(treatments[key][metric].mean()),
            plot_order(key[0], key[1]),
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
            paired = paired_by_seed(candidate_values, reference_values, metric)
            if len(paired) < 5:
                lower_than_reference = False
            else:
                candidate_metric = paired["row"].to_numpy(dtype=float)
                reference_metric = paired["column"].to_numpy(dtype=float)
                test = paired_wilcoxon(candidate_metric, reference_metric)
                delta = cliffs_delta(candidate_metric, reference_metric)
                lower_than_reference = test["p_value"] < 0.05 and delta < -0.33
            if found_next_rank or lower_than_reference:
                found_next_rank = True
                next_remaining.append(candidate)
            else:
                ranks[candidate] = rank
        remaining = next_remaining
        rank += 1
    return ranks


def rank_zero_reference(treatments, ranks, metric):
    rank_zero = [key for key, rank in ranks.items() if rank == 0]
    return sorted(
        rank_zero,
        key=lambda key: (
            -float(treatments[key][metric].mean()),
            plot_order(key[0], key[1]),
        ),
    )[0]


def paired_by_seed(row_values, column_values, metric):
    paired = row_values.merge(column_values, on="seed", suffixes=("_row", "_column")).dropna()
    return paired.rename(
        columns={
            f"{metric}_row": "row",
            f"{metric}_column": "column",
        }
    )


def paired_wilcoxon(row_values, column_values):
    row_values = np.asarray(row_values, dtype=float)
    column_values = np.asarray(column_values, dtype=float)
    if len(row_values) == 0 or np.allclose(row_values - column_values, 0.0):
        return {"statistic": 0.0, "p_value": 1.0}
    result = wilcoxon(row_values, column_values, alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def cliffs_delta(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0 or len(ys) == 0:
        return np.nan
    comparisons = np.sign(xs[:, None] - ys[None, :])
    return float(np.sum(comparisons) / comparisons.size)


def plot_order(parameter_mode, parameter_label):
    if parameter_mode == "dynamic":
        return len(PARAMETER_GRID) + 1
    return float(parameter_label)


def plot_weight_rank(summary):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    labels = [f"{value:.1f}" for value in PARAMETER_GRID] + ["Dynamic"]
    metric = "cosine_similarity"
    for col_index, density_model in enumerate(("Neighbor", "Kernel")):
        axis = axes[col_index]
        axis.set_facecolor("white")
        subset = summary[summary["density_model"].eq(density_model)].copy()
        subset["plot_label"] = subset["parameter_label"]
        subset = subset.set_index("plot_label").reindex(labels).reset_index()
        xs = np.arange(len(labels))
        means = subset[metric].to_numpy(dtype=float)
        stds = subset[f"{metric}_std"].to_numpy(dtype=float)
        ranks = subset[f"{metric}_rank"].to_numpy(dtype=float)
        axis.errorbar(
            xs,
            means,
            yerr=stds,
            fmt="o-",
            color="#334155",
            ecolor="#94a3b8",
            capsize=3,
            linewidth=1.5,
        )
        for x, mean, rank in zip(xs, means, ranks):
            if np.isfinite(mean):
                axis.text(x, mean + 0.015, f"R{int(rank)}", ha="center", va="bottom", fontsize=8)
        axis.set_title(density_model)
        axis.set_xticks(xs)
        axis.set_xticklabels(labels, rotation=45, ha="right")
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        axis.set_ylim(0.0, 1.05)
        axis.set_xlabel("Radius" if density_model == "Neighbor" else "Bandwidth")
        if col_index == 0:
            axis.set_ylabel(r"Mean cosine($\hat{w}$, $w^*$)")
    fig.suptitle("Synthetic weight recovery across fixed parameters and Dynamic", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(PARAMETER_RANK_PDF, bbox_inches="tight", facecolor="white")
    fig.savefig(PARAMETER_RANK_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
