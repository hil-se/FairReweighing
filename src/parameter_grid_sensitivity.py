from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from experiment import DEFAULT_BANDWIDTH_GRID, DEFAULT_RADIUS_GRID, Experiment


DATASETS = [
    "Synthetic",
    "LSAC",
    "LSAC_Multi",
    "Community",
    "Community_Con",
    "Community_Multi",
    "Insurance",
    "Insurance_Multi",
]
DENSITY_MODELS = ["Neighbor", "Kernel"]
FIXED_TARGET = {"Neighbor": 0.3, "Kernel": 0.2}
PLOT_DATASETS = [
    "Synthetic",
    "LSAC",
    "Community",
    "Community_Con",
    "Insurance",
    "Insurance_Multi",
]
OUTPUT_DIR = Path("result")
PARAMETER_OUTPUT_DIR = OUTPUT_DIR / "parameter_sensitivity"
FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"
RAW_OUTPUT = PARAMETER_OUTPUT_DIR / "parameter_grid_sensitivity_raw.csv"
FIXED_CHECKPOINT_OUTPUT = PARAMETER_OUTPUT_DIR / "parameter_grid_sensitivity_fixed_checkpoint.csv"
SUMMARY_OUTPUT = PARAMETER_OUTPUT_DIR / "parameter_grid_sensitivity_summary.csv"
COMPARISON_OUTPUT = PARAMETER_OUTPUT_DIR / "parameter_grid_sensitivity_comparison.csv"
FIXED_VS_BEST_OUTPUT = PARAMETER_OUTPUT_DIR / "parameter_grid_fixed_vs_best_tests.csv"
FIXED_VS_DYNAMIC_OUTPUT = PARAMETER_OUTPUT_DIR / "parameter_grid_fixed_vs_dynamic_tests.csv"
FIXED_VS_DYNAMIC_COST_OUTPUT = PARAMETER_OUTPUT_DIR / "parameter_grid_fixed_vs_dynamic_cost.csv"
CSEP_PLOT = FIGURE_OUTPUT_DIR / "parameter_grid_csep.pdf"
MSE_PLOT = FIGURE_OUTPUT_DIR / "parameter_grid_mse.pdf"
CSEP_PNG = FIGURE_OUTPUT_DIR / "parameter_grid_csep.png"
MSE_PNG = FIGURE_OUTPUT_DIR / "parameter_grid_mse.png"
CSEP_12_PANEL_PLOT = FIGURE_OUTPUT_DIR / "parameter_grid_csep_12panel.pdf"
CSEP_12_PANEL_PNG = FIGURE_OUTPUT_DIR / "parameter_grid_csep_12panel.png"


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARAMETER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or DATASETS
    seeds = list(range(args.first_seed, args.first_seed + args.repeats))

    fixed_raw = run_fixed_grid(datasets, seeds, resume=not args.no_resume)
    dynamic_raw = read_dynamic_rows(datasets)
    raw = pd.concat([fixed_raw, dynamic_raw], ignore_index=True, sort=False)
    raw.to_csv(RAW_OUTPUT, index=False)

    summary = summarize(raw)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    comparison = compare_dynamic_to_fixed(summary)
    comparison.to_csv(COMPARISON_OUTPUT, index=False)
    fixed_vs_best = compare_fixed_target_to_best(raw, summary)
    fixed_vs_best.to_csv(FIXED_VS_BEST_OUTPUT, index=False)
    fixed_vs_dynamic = compare_fixed_target_to_dynamic(raw)
    fixed_vs_dynamic.to_csv(FIXED_VS_DYNAMIC_OUTPUT, index=False)
    fixed_vs_dynamic_cost = runtime_cost_table(fixed_vs_dynamic)
    fixed_vs_dynamic_cost.to_csv(FIXED_VS_DYNAMIC_COST_OUTPUT, index=False)
    plot_metric(summary, "c_sep_xfit", CSEP_PLOT, "Mean $\\hat{C}_{sep}^{xfit}$")
    plot_metric(summary, "c_sep_xfit", CSEP_PNG, "Mean $\\hat{C}_{sep}^{xfit}$")
    plot_csep_12_panel(summary, fixed_vs_best, CSEP_12_PANEL_PLOT)
    plot_csep_12_panel(summary, fixed_vs_best, CSEP_12_PANEL_PNG)
    plot_metric(summary, "mse", MSE_PLOT, "Mean MSE")
    plot_metric(summary, "mse", MSE_PNG, "Mean MSE")

    print(f"Wrote raw rows to {RAW_OUTPUT}")
    print(f"Wrote summary to {SUMMARY_OUTPUT}")
    print(f"Wrote comparison to {COMPARISON_OUTPUT}")
    print(f"Wrote fixed-vs-best tests to {FIXED_VS_BEST_OUTPUT}")
    print(f"Wrote fixed-vs-dynamic tests to {FIXED_VS_DYNAMIC_OUTPUT}")
    print(f"Wrote fixed-vs-dynamic cost table to {FIXED_VS_DYNAMIC_COST_OUTPUT}")
    print(f"Wrote plots to {CSEP_PLOT}, {CSEP_PNG}, {CSEP_12_PANEL_PLOT}, {CSEP_12_PANEL_PNG}, {MSE_PLOT}, and {MSE_PNG}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(fixed_vs_best.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(fixed_vs_dynamic.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(fixed_vs_dynamic_cost.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def parse_args():
    parser = ArgumentParser(description="Fixed-grid parameter sensitivity for FairReweighing.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def run_fixed_grid(datasets, seeds, resume=True):
    existing = read_existing_fixed() if resume else pd.DataFrame()
    done = set()
    if not existing.empty:
        for _, row in existing.iterrows():
            done.add((row["dataset"], row["density_model"], float(row["parameter_value"]), int(row["seed"])))

    rows = []
    total = sum(len(parameter_grid(model)) * len(seeds) for model in DENSITY_MODELS) * len(datasets)
    completed = len(done)
    for dataset in datasets:
        for density_model in DENSITY_MODELS:
            for parameter_value in parameter_grid(density_model):
                for seed in seeds:
                    key = (dataset, density_model, float(parameter_value), int(seed))
                    if key in done:
                        continue
                    row = run_one(dataset, density_model, parameter_value, seed)
                    rows.append(row)
                    completed += 1
                    if completed % 25 == 0:
                        checkpoint = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False)
                        checkpoint.to_csv(FIXED_CHECKPOINT_OUTPUT, index=False)
                        print(f"Completed {completed}/{total} fixed-grid runs", flush=True)

    new_rows = pd.DataFrame(rows)
    checkpoint = pd.concat([existing, new_rows], ignore_index=True, sort=False)
    if not checkpoint.empty:
        checkpoint.to_csv(FIXED_CHECKPOINT_OUTPUT, index=False)
    if existing.empty:
        return new_rows
    return pd.concat([existing, new_rows], ignore_index=True, sort=False)


def read_existing_fixed():
    frames = []
    for path in (FIXED_CHECKPOINT_OUTPUT, RAW_OUTPUT):
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        if "parameter_mode" not in raw.columns:
            continue
        frames.append(raw[raw["parameter_mode"].eq("fixed")].copy())
    if not frames:
        return pd.DataFrame()
    fixed = pd.concat(frames, ignore_index=True, sort=False)
    return fixed.drop_duplicates(
        subset=["dataset", "density_model", "parameter_value", "seed"],
        keep="last",
    )


def run_one(dataset, density_model, parameter_value, seed):
    experiment = Experiment(
        data=dataset,
        regressor="linear",
        balance="fair-reweighing",
        density_model=density_model,
        seed=seed,
    )
    experiment.select_density_param = lambda parameter_value=parameter_value: parameter_value
    row = experiment.run()
    row["parameter_mode"] = "fixed"
    row["parameter_value"] = float(parameter_value)
    return row


def read_dynamic_rows(datasets):
    rows = []
    for dataset in datasets:
        path = OUTPUT_DIR / "raw" / f"{dataset}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path)
        raw.columns = [str(column).strip() for column in raw.columns]
        for column in ("dataset", "method", "density_model"):
            raw[column] = raw[column].astype(str).str.strip()
        fair = raw[raw["method"].eq("fair-reweighing") & raw["density_model"].isin(DENSITY_MODELS)].copy()
        if fair.empty:
            continue
        fair["parameter_mode"] = "dynamic"
        fair["parameter_value"] = np.where(
            fair["density_model"].eq("Neighbor"),
            pd.to_numeric(fair["selected_radius"], errors="coerce"),
            pd.to_numeric(fair["selected_bandwidth"], errors="coerce"),
        )
        rows.append(fair)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def parameter_grid(density_model):
    return DEFAULT_RADIUS_GRID if density_model == "Neighbor" else DEFAULT_BANDWIDTH_GRID


def summarize(raw):
    value_cols = ["mse", "mae", "r2", "r_sep", "i_sep", "c_sep_xfit", "parameter_value"]
    rows = []
    group_cols = ["dataset", "density_model", "parameter_mode"]
    for keys, group in raw.groupby(group_cols, dropna=False):
        dataset, density_model, parameter_mode = keys
        if parameter_mode == "fixed":
            subgroups = group.groupby("parameter_value", dropna=False)
        else:
            subgroups = [(np.nan, group)]
        for parameter_value, subgroup in subgroups:
            row = {
                "dataset": dataset,
                "density_model": density_model,
                "parameter_mode": parameter_mode,
                "parameter_value": parameter_value,
                "n": int(pd.to_numeric(subgroup["seed"], errors="coerce").nunique()),
            }
            for column in value_cols:
                if column not in subgroup.columns:
                    continue
                values = pd.to_numeric(subgroup[column], errors="coerce").dropna()
                row[column] = float(values.mean()) if not values.empty else np.nan
                row[f"{column}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def compare_dynamic_to_fixed(summary):
    rows = []
    for (dataset, density_model), group in summary.groupby(["dataset", "density_model"], dropna=False):
        fixed = group[group["parameter_mode"].eq("fixed")].copy()
        dynamic = group[group["parameter_mode"].eq("dynamic")].copy()
        if fixed.empty or dynamic.empty:
            continue
        dynamic_row = dynamic.iloc[0]
        best_csep = fixed.loc[fixed["c_sep_xfit"].idxmin()]
        default_fixed = fixed[np.isclose(fixed["parameter_value"].astype(float), FIXED_TARGET[density_model])]
        default_row = default_fixed.iloc[0] if not default_fixed.empty else None
        rows.append({
            "dataset": dataset,
            "density_model": density_model,
            "dynamic_c_sep_xfit": dynamic_row["c_sep_xfit"],
            "dynamic_mse": dynamic_row["mse"],
            "mean_selected_parameter": dynamic_row["parameter_value"],
            "best_fixed_parameter": best_csep["parameter_value"],
            "best_fixed_c_sep_xfit": best_csep["c_sep_xfit"],
            "best_fixed_mse": best_csep["mse"],
            "dynamic_csep_minus_best_fixed": dynamic_row["c_sep_xfit"] - best_csep["c_sep_xfit"],
            "default_parameter": FIXED_TARGET[density_model],
            "default_c_sep_xfit": np.nan if default_row is None else default_row["c_sep_xfit"],
            "default_mse": np.nan if default_row is None else default_row["mse"],
            "dynamic_csep_minus_default": np.nan if default_row is None else dynamic_row["c_sep_xfit"] - default_row["c_sep_xfit"],
        })
    return pd.DataFrame(rows)


def compare_fixed_target_to_best(raw, summary):
    rows = []
    fixed_summary = summary[summary["parameter_mode"].eq("fixed")].copy()
    fixed_raw = raw[raw["parameter_mode"].eq("fixed")].copy()
    fixed_raw["parameter_value"] = pd.to_numeric(fixed_raw["parameter_value"], errors="coerce")
    fixed_raw["seed"] = pd.to_numeric(fixed_raw["seed"], errors="coerce").astype("Int64")
    fixed_raw["c_sep_xfit"] = pd.to_numeric(fixed_raw["c_sep_xfit"], errors="coerce")

    for (dataset, density_model), group in fixed_summary.groupby(["dataset", "density_model"], dropna=False):
        target_param = FIXED_TARGET[density_model]
        best_row = group.loc[group["c_sep_xfit"].idxmin()]
        best_param = float(best_row["parameter_value"])
        fixed_values = seed_values(fixed_raw, dataset, density_model, target_param)
        best_values = seed_values(fixed_raw, dataset, density_model, best_param)
        paired = fixed_values.merge(best_values, on="seed", suffixes=("_fixed", "_best"))
        fixed_metric = paired["c_sep_xfit_fixed"].to_numpy(dtype=float)
        best_metric = paired["c_sep_xfit_best"].to_numpy(dtype=float)
        test = paired_wilcoxon(fixed_metric, best_metric)
        delta = cliffs_delta(fixed_metric, best_metric)
        mean_difference = float(np.mean(fixed_metric - best_metric)) if len(paired) else np.nan
        oracle_best_lower = bool(test["p_value"] < 0.05 and delta > 0.33)
        rows.append({
            "dataset": dataset,
            "density_model": density_model,
            "fixed_parameter": target_param,
            "best_parameter": best_param,
            "n": int(len(paired)),
            "fixed_c_sep_xfit": float(np.mean(fixed_metric)) if len(paired) else np.nan,
            "best_c_sep_xfit": float(np.mean(best_metric)) if len(paired) else np.nan,
            "mean_difference": mean_difference,
            "wilcoxon_stat": test["statistic"],
            "p_value": test["p_value"],
            "cliffs_delta": delta,
            "oracle_best_lower": oracle_best_lower,
            "conclusion": "oracle best lower" if oracle_best_lower else "fixed accepted",
        })
    return pd.DataFrame(rows)


def compare_fixed_target_to_dynamic(raw):
    rows = []
    fixed_raw = raw[raw["parameter_mode"].eq("fixed")].copy()
    dynamic_raw = raw[raw["parameter_mode"].eq("dynamic")].copy()
    for frame in (fixed_raw, dynamic_raw):
        frame["parameter_value"] = pd.to_numeric(frame["parameter_value"], errors="coerce")
        frame["seed"] = pd.to_numeric(frame["seed"], errors="coerce").astype("Int64")
        frame["c_sep_xfit"] = pd.to_numeric(frame["c_sep_xfit"], errors="coerce")
        frame["mse"] = pd.to_numeric(frame["mse"], errors="coerce")
        frame["fit_seconds"] = pd.to_numeric(frame.get("fit_seconds"), errors="coerce")
        frame["total_seconds"] = pd.to_numeric(frame.get("total_seconds"), errors="coerce")

    pairs = sorted(set(zip(dynamic_raw["dataset"], dynamic_raw["density_model"])))
    for dataset, density_model in pairs:
        target_param = FIXED_TARGET[density_model]
        fixed_values = metric_values(fixed_raw, dataset, density_model, "fixed", "c_sep_xfit", target_param)
        dynamic_values = metric_values(dynamic_raw, dataset, density_model, "dynamic", "c_sep_xfit")
        paired_csep = fixed_values.merge(dynamic_values, on="seed", suffixes=("_fixed", "_dynamic"))
        fixed_csep = paired_csep["c_sep_xfit_fixed"].to_numpy(dtype=float)
        dynamic_csep = paired_csep["c_sep_xfit_dynamic"].to_numpy(dtype=float)

        fixed_mse_values = metric_values(fixed_raw, dataset, density_model, "fixed", "mse", target_param)
        dynamic_mse_values = metric_values(dynamic_raw, dataset, density_model, "dynamic", "mse")
        paired_mse = fixed_mse_values.merge(dynamic_mse_values, on="seed", suffixes=("_fixed", "_dynamic"))
        fixed_mse = paired_mse["mse_fixed"].to_numpy(dtype=float)
        dynamic_mse = paired_mse["mse_dynamic"].to_numpy(dtype=float)

        fixed_fit_values = metric_values(fixed_raw, dataset, density_model, "fixed", "fit_seconds", target_param)
        dynamic_fit_values = metric_values(dynamic_raw, dataset, density_model, "dynamic", "fit_seconds")
        paired_fit = fixed_fit_values.merge(dynamic_fit_values, on="seed", suffixes=("_fixed", "_dynamic"))
        fixed_fit = paired_fit["fit_seconds_fixed"].to_numpy(dtype=float)
        dynamic_fit = paired_fit["fit_seconds_dynamic"].to_numpy(dtype=float)

        fixed_total_values = metric_values(fixed_raw, dataset, density_model, "fixed", "total_seconds", target_param)
        dynamic_total_values = metric_values(dynamic_raw, dataset, density_model, "dynamic", "total_seconds")
        paired_total = fixed_total_values.merge(dynamic_total_values, on="seed", suffixes=("_fixed", "_dynamic"))
        fixed_total = paired_total["total_seconds_fixed"].to_numpy(dtype=float)
        dynamic_total = paired_total["total_seconds_dynamic"].to_numpy(dtype=float)

        csep_test = paired_wilcoxon(fixed_csep, dynamic_csep)
        csep_delta = cliffs_delta(fixed_csep, dynamic_csep)
        mse_test = paired_wilcoxon(fixed_mse, dynamic_mse)
        mse_delta = cliffs_delta(fixed_mse, dynamic_mse)
        rows.append({
            "dataset": dataset,
            "density_model": density_model,
            "fixed_parameter": target_param,
            "mean_dynamic_parameter": mean_dynamic_parameter(dynamic_raw, dataset, density_model),
            "n": int(len(paired_csep)),
            "fixed_c_sep_xfit": float(np.mean(fixed_csep)) if len(paired_csep) else np.nan,
            "dynamic_c_sep_xfit": float(np.mean(dynamic_csep)) if len(paired_csep) else np.nan,
            "csep_mean_difference": float(np.mean(fixed_csep - dynamic_csep)) if len(paired_csep) else np.nan,
            "csep_wilcoxon_stat": csep_test["statistic"],
            "csep_p_value": csep_test["p_value"],
            "csep_cliffs_delta": csep_delta,
            "csep_conclusion": comparison_conclusion(csep_test["p_value"], csep_delta),
            "fixed_mse": float(np.mean(fixed_mse)) if len(paired_mse) else np.nan,
            "dynamic_mse": float(np.mean(dynamic_mse)) if len(paired_mse) else np.nan,
            "mse_mean_difference": float(np.mean(fixed_mse - dynamic_mse)) if len(paired_mse) else np.nan,
            "mse_wilcoxon_stat": mse_test["statistic"],
            "mse_p_value": mse_test["p_value"],
            "mse_cliffs_delta": mse_delta,
            "mse_conclusion": comparison_conclusion(mse_test["p_value"], mse_delta),
            "fixed_fit_seconds": mean_or_nan(fixed_fit),
            "dynamic_fit_seconds": mean_or_nan(dynamic_fit),
            "fit_seconds_change_pct": percent_change(mean_or_nan(fixed_fit), mean_or_nan(dynamic_fit)),
            "fixed_total_seconds": mean_or_nan(fixed_total),
            "dynamic_total_seconds": mean_or_nan(dynamic_total),
            "total_seconds_change_pct": percent_change(mean_or_nan(fixed_total), mean_or_nan(dynamic_total)),
        })
    return pd.DataFrame(rows)


def runtime_cost_table(comparison):
    columns = [
        "dataset",
        "density_model",
        "fixed_parameter",
        "mean_dynamic_parameter",
        "n",
        "fixed_c_sep_xfit",
        "dynamic_c_sep_xfit",
        "fixed_mse",
        "dynamic_mse",
        "fixed_fit_seconds",
        "dynamic_fit_seconds",
        "fit_seconds_change_pct",
        "fixed_total_seconds",
        "dynamic_total_seconds",
        "total_seconds_change_pct",
    ]
    return comparison[[column for column in columns if column in comparison.columns]].copy()


def metric_values(raw, dataset, density_model, parameter_mode, metric, parameter_value=None):
    mask = (
        raw["dataset"].eq(dataset)
        & raw["density_model"].eq(density_model)
        & raw["parameter_mode"].eq(parameter_mode)
    )
    if parameter_value is not None:
        mask &= np.isclose(raw["parameter_value"].astype(float), float(parameter_value))
    return raw.loc[mask, ["seed", metric]].dropna().copy()


def mean_or_nan(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def percent_change(fixed_value, dynamic_value):
    if not np.isfinite(fixed_value) or not np.isfinite(dynamic_value) or dynamic_value == 0:
        return np.nan
    return 100.0 * (fixed_value - dynamic_value) / dynamic_value


def mean_dynamic_parameter(dynamic_raw, dataset, density_model):
    mask = dynamic_raw["dataset"].eq(dataset) & dynamic_raw["density_model"].eq(density_model)
    values = pd.to_numeric(dynamic_raw.loc[mask, "parameter_value"], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def comparison_conclusion(p_value, delta):
    if not np.isfinite(p_value) or not np.isfinite(delta):
        return "insufficient data"
    if p_value < 0.05 and delta > 0.33:
        return "dynamic lower"
    if p_value < 0.05 and delta < -0.33:
        return "fixed lower"
    return "no meaningful difference"


def seed_values(raw, dataset, density_model, parameter_value):
    mask = (
        raw["dataset"].eq(dataset)
        & raw["density_model"].eq(density_model)
        & np.isclose(raw["parameter_value"].astype(float), float(parameter_value))
    )
    return raw.loc[mask, ["seed", "c_sep_xfit"]].dropna().copy()


def paired_wilcoxon(fixed_values, best_values):
    fixed_values = np.asarray(fixed_values, dtype=float)
    best_values = np.asarray(best_values, dtype=float)
    if len(fixed_values) == 0 or np.allclose(fixed_values - best_values, 0.0):
        return {"statistic": 0.0, "p_value": 1.0}
    result = wilcoxon(fixed_values, best_values, alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def cliffs_delta(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if len(xs) == 0 or len(ys) == 0:
        return np.nan
    comparisons = np.sign(xs[:, None] - ys[None, :])
    return float(np.sum(comparisons) / comparisons.size)


def plot_csep_12_panel(summary, tests, output_path):
    fig, axes = plt.subplots(len(PLOT_DATASETS), 2, figsize=(8.2, 12.2), sharey=False)
    colors = {"Neighbor": "#2671b8", "Kernel": "#d2691e"}
    for row_index, dataset in enumerate(PLOT_DATASETS):
        for column_index, density_model in enumerate(DENSITY_MODELS):
            axis = axes[row_index, column_index]
            fixed = summary[
                summary["dataset"].eq(dataset)
                & summary["density_model"].eq(density_model)
                & summary["parameter_mode"].eq("fixed")
            ].sort_values("parameter_value")
            test_row = tests[
                tests["dataset"].eq(dataset)
                & tests["density_model"].eq(density_model)
            ]
            if fixed.empty:
                axis.axis("off")
                continue
            x = fixed["parameter_value"].to_numpy(dtype=float)
            y = fixed["c_sep_xfit"].to_numpy(dtype=float)
            axis.plot(x, y, marker="o", color=colors[density_model], linewidth=1.5, markersize=4)
            target_param = FIXED_TARGET[density_model]
            axis.axvline(target_param, color="#111827", linestyle="--", linewidth=1.0, label="fixed")
            if not test_row.empty:
                best_param = float(test_row.iloc[0]["best_parameter"])
                axis.axvline(best_param, color="#6B7280", linestyle=":", linewidth=1.0, label="best")
                conclusion = "OK" if not bool(test_row.iloc[0]["oracle_best_lower"]) else "best lower"
                p_value = float(test_row.iloc[0]["p_value"])
                delta = float(test_row.iloc[0]["cliffs_delta"])
                axis.text(
                    0.98,
                    0.95,
                    f"p={format_p_value(p_value)}\n$\\delta$={delta:.2f}\n{conclusion}",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D1D5DB", "alpha": 0.9},
                )
            axis.set_title(f"{display_dataset(dataset)}: {density_model}", fontsize=9)
            axis.set_xlabel("Radius" if density_model == "Neighbor" else "Bandwidth", fontsize=8)
            axis.set_ylabel("$\\hat{C}_{sep}^{xfit}$", fontsize=8)
            axis.grid(True, linewidth=0.35, alpha=0.35)
            axis.tick_params(axis="both", labelsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def format_p_value(p_value):
    if not np.isfinite(p_value):
        return "--"
    if p_value < 0.001:
        return "<.001"
    return f"{p_value:.3f}"


def plot_metric(summary, metric, output_path, ylabel):
    datasets = DATASETS
    fig, axes = plt.subplots(4, 2, figsize=(10, 13), sharey=False)
    axes = axes.ravel()
    colors = {"Neighbor": "#2671b8", "Kernel": "#d2691e"}
    markers = {"Neighbor": "o", "Kernel": "s"}
    for axis, dataset in zip(axes, datasets):
        subset = summary[summary["dataset"].eq(dataset)]
        for density_model in DENSITY_MODELS:
            fixed = subset[
                subset["density_model"].eq(density_model)
                & subset["parameter_mode"].eq("fixed")
            ].sort_values("parameter_value")
            dynamic = subset[
                subset["density_model"].eq(density_model)
                & subset["parameter_mode"].eq("dynamic")
            ]
            if fixed.empty:
                continue
            x = fixed["parameter_value"].to_numpy(dtype=float)
            y = fixed[metric].to_numpy(dtype=float)
            axis.plot(
                x,
                y,
                marker=markers[density_model],
                color=colors[density_model],
                label=f"{density_model} fixed",
                linewidth=1.6,
                markersize=4,
            )
            if not dynamic.empty and np.isfinite(dynamic.iloc[0][metric]):
                axis.axhline(
                    dynamic.iloc[0][metric],
                    color=colors[density_model],
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.8,
                    label=f"{density_model} dynamic",
                )
        axis.set_title(display_dataset(dataset), fontsize=10)
        axis.set_xlabel("Radius / bandwidth")
        axis.set_ylabel(ylabel)
        axis.grid(True, linewidth=0.4, alpha=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def display_dataset(dataset):
    names = {
        "Community": "Communities",
        "Community_Con": "Communities-Con",
        "Community_Multi": "Communities-Multi",
    }
    return names.get(dataset, dataset.replace("_", "-"))


if __name__ == "__main__":
    main()
