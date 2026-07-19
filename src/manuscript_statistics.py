"""Regenerate manuscript-level means, ranks, and consistency checks."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from main import (
    cliffs_delta,
    paired_by_seed,
    paired_wilcoxon,
    recursive_ranks,
    statistical_tests_pairwise,
    treatment_display_name,
    treatment_series,
)


RESULT_DIR = Path("result")
RAW_DIR = RESULT_DIR / "raw"
MAIN_DIR = RESULT_DIR / "main"
FAIRICP_DIR = RESULT_DIR / "baselines" / "fairicp"

DATASETS = [
    "Synthetic",
    "LSAC",
    "Community",
    "Community_Con",
    "Insurance",
    "SCUT_Sex",
    "LSAC_Multi",
    "Community_Multi",
    "Insurance_Multi",
    "SCUT_Multi",
    "German",
    "Heart",
]
REGRESSION_DATASETS = DATASETS[:10]
CATEGORICAL_DATASETS = ["Synthetic", "LSAC", "Community", "German", "Heart", "SCUT_Sex"]

FAIRICP_FILES = [
    "fairicp_probe.csv",
    "fairicp_probe_scut.csv",
]


def main():
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    combined = combined_seed_rows()
    combined.to_csv(MAIN_DIR / "manuscript_seed_results.csv", index=False)

    means = manuscript_means(combined)
    means.to_csv(MAIN_DIR / "manuscript_means.csv", index=False)

    raw_paths = []
    for dataset, rows in combined.groupby("dataset", sort=False):
        path = MAIN_DIR / f"_stats_{dataset}.csv"
        rows.to_csv(path, index=False)
        raw_paths.append(path)
    pairwise = statistical_tests_pairwise(raw_paths)
    pairwise.to_csv(MAIN_DIR / "stat_tests_pairwise.csv", index=False)
    for path in raw_paths:
        path.unlink()

    ranks = rank_summary(combined)
    ranks.to_csv(MAIN_DIR / "statistical_rank_summary.csv", index=False)
    estimator_ranks = density_estimator_rank_summary(ranks)
    estimator_ranks.to_csv(MAIN_DIR / "density_estimator_rank_summary.csv", index=False)

    consistency = metric_consistency(means)
    consistency.to_csv(MAIN_DIR / "metric_consistency.csv", index=False)

    print(means.to_string(index=False))
    print("\nRank summary")
    print(ranks.to_string(index=False))
    print("\nDensity-estimator ranks")
    print(estimator_ranks.to_string(index=False))
    print("\nMetric consistency")
    print(consistency.to_string(index=False))


def combined_seed_rows():
    frames = []
    for dataset in DATASETS:
        path = RAW_DIR / f"{dataset}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = read_csv(path)
        frame = frame[~frame["method"].eq("fairicp")].copy()
        frames.append(frame)

    fairicp_frames = []
    for filename in FAIRICP_FILES:
        path = FAIRICP_DIR / filename
        if path.exists():
            fairicp_frames.append(read_csv(path))
    if fairicp_frames:
        fairicp = pd.concat(fairicp_frames, ignore_index=True, sort=False)
        fairicp = fairicp[
            fairicp["dataset"].isin(DATASETS) & fairicp["method"].eq("fairicp")
        ].copy()
        fairicp["density_model"] = "none"
        fairicp = fairicp.drop_duplicates(
            subset=["dataset", "method", "density_model", "seed"],
            keep="last",
        )
        frames.append(fairicp)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["model"] = combined["dataset"].map(common_model_name)
    combined["seed"] = pd.to_numeric(combined["seed"], errors="coerce").astype("Int64")
    combined = combined.dropna(subset=["seed"])
    combined = combined.drop_duplicates(
        subset=["dataset", "method", "density_model", "seed"],
        keep="last",
    )
    verify_counts(combined)
    verify_component_scores(combined)
    return combined.sort_values(
        ["dataset", "method", "density_model", "seed"]
    ).reset_index(drop=True)


def manuscript_means(combined):
    metrics = [
        "mse",
        "mae",
        "r2",
        "accuracy",
        "f1",
        "aod",
        "eod",
        "r_sep",
        "i_sep",
        "c_sep_xfit",
    ]
    sensitive = sorted(
        column for column in combined.columns if column.startswith("c_sep_xfit_")
    )
    rows = []
    keys = ["dataset", "model", "method", "density_model"]
    for key, treatment in combined.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key))
        row["treatment"] = treatment_display_name(key[2], key[3])
        row["n"] = int(treatment["seed"].nunique())
        for metric in metrics + sensitive:
            if metric not in treatment:
                continue
            values = pd.to_numeric(treatment[metric], errors="coerce").dropna()
            if values.empty:
                row[metric] = np.nan
                continue
            row[metric] = float(values.mean())
            if metric == "c_sep_xfit":
                row["c_sep_xfit_sd"] = float(values.std(ddof=1))
                row["c_sep_xfit_ci95"] = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
        rows.append(row)
    return pd.DataFrame(rows)


def rank_summary(combined):
    rows = []
    for dataset in DATASETS:
        group = combined[combined["dataset"].eq(dataset)]
        treatments = treatment_series(group)
        ranks = recursive_ranks(treatments)
        ordered = sorted(
            treatments,
            key=lambda key: (
                ranks[key],
                float(treatments[key]["c_sep_xfit"].mean()),
                treatment_display_name(*key),
            ),
        )
        rank_zero = [treatment_display_name(*key) for key in ordered if ranks[key] == 0]
        neighbor = ("fair-reweighing", "Neighbor")
        kernel = ("fair-reweighing", "Kernel")
        paired = paired_by_seed(treatments[neighbor], treatments[kernel])
        neighbor_values = paired["row"].to_numpy(dtype=float)
        kernel_values = paired["column"].to_numpy(dtype=float)
        test = paired_wilcoxon(neighbor_values, kernel_values)
        delta = cliffs_delta(neighbor_values, kernel_values)
        rows.append({
            "dataset": dataset,
            "rank_zero_treatments": "; ".join(rank_zero),
            "neighbor_rank": int(ranks[neighbor]),
            "kernel_rank": int(ranks[kernel]),
            "neighbor_mean_c_sep_xfit": float(neighbor_values.mean()),
            "kernel_mean_c_sep_xfit": float(kernel_values.mean()),
            "neighbor_vs_kernel_p": test["p_value"],
            "neighbor_vs_kernel_delta": delta,
        })
    return pd.DataFrame(rows)


def density_estimator_rank_summary(ranks):
    rows = []
    groups = {
        "single": REGRESSION_DATASETS[:6],
        "vector": REGRESSION_DATASETS[6:],
        "overall": REGRESSION_DATASETS,
    }
    for estimator, column in (("Neighbor", "neighbor_rank"), ("Kernel", "kernel_rank")):
        row = {"estimator": estimator}
        for group_name, datasets in groups.items():
            values = ranks[ranks["dataset"].isin(datasets)][column]
            row[f"{group_name}_mean_rank"] = float(values.mean())
            row[f"{group_name}_rank0"] = int((values == 0).sum())
            row[f"{group_name}_n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows)


def metric_consistency(means):
    rows = means[means["dataset"].isin(CATEGORICAL_DATASETS)].copy()
    comparisons = {
        "|r_sep-1|": np.abs(pd.to_numeric(rows["r_sep"], errors="coerce") - 1.0),
        "i_sep": pd.to_numeric(rows["i_sep"], errors="coerce"),
    }
    csep = pd.to_numeric(rows["c_sep_xfit"], errors="coerce")
    output = []
    for name, older in comparisons.items():
        mask = older.notna() & csep.notna()
        output.append({
            "comparison": name,
            "points": int(mask.sum()),
            "pearson": float(pearsonr(older[mask], csep[mask]).statistic),
            "spearman": float(spearmanr(older[mask], csep[mask]).statistic),
        })
    return pd.DataFrame(output)


def verify_counts(combined):
    counts = combined.groupby(["dataset", "method", "density_model"])["seed"].nunique()
    incomplete = counts[counts != 20]
    if not incomplete.empty:
        raise RuntimeError(f"Expected 20 seeds per treatment:\n{incomplete}")


def verify_component_scores(combined):
    for dataset, rows in combined.groupby("dataset"):
        components = [
            column
            for column in rows.columns
            if column.startswith("c_sep_xfit_") and rows[column].notna().any()
        ]
        if not components:
            continue
        expected = rows[components].mean(axis=1, skipna=True)
        comparable = expected.notna() & rows["c_sep_xfit"].notna()
        differences = (rows.loc[comparable, "c_sep_xfit"] - expected[comparable]).abs()
        if (differences > 1e-10).any():
            raise RuntimeError(
                f"Overall and component Csep values disagree for {dataset}; "
                f"maximum absolute difference is {differences.max():.6g}."
            )


def common_model_name(dataset):
    if dataset in {"German", "Heart"}:
        return "logistic"
    if dataset in {"SCUT_Sex", "SCUT_Multi"}:
        return "vgg_face"
    return "linear"


def read_csv(path):
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip() for column in frame.columns]
    for column in ("dataset", "model", "method", "density_model"):
        if column in frame:
            frame[column] = frame[column].astype(str).str.strip()
    return frame


if __name__ == "__main__":
    main()
