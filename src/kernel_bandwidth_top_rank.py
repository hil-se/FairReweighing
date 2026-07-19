from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.stats import wilcoxon

from experiment import Experiment


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
BANDWIDTH_GRID = [round(value, 2) for value in np.arange(0.10, 0.201, 0.01)]
OUTPUT_DIR = Path("result")
PARAMETER_OUTPUT_DIR = OUTPUT_DIR / "parameter_sensitivity"
FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"
RAW_OUTPUT = PARAMETER_OUTPUT_DIR / "kernel_bandwidth_top_rank_raw.csv"
CHECKPOINT_OUTPUT = PARAMETER_OUTPUT_DIR / "kernel_bandwidth_top_rank_checkpoint.csv"
SUMMARY_OUTPUT = PARAMETER_OUTPUT_DIR / "kernel_bandwidth_top_rank_summary.csv"
PAIRWISE_OUTPUT = PARAMETER_OUTPUT_DIR / "kernel_bandwidth_top_rank_pairwise.csv"
MATRIX_OUTPUT = PARAMETER_OUTPUT_DIR / "kernel_bandwidth_top_rank_matrix.csv"
HEATMAP_PDF = FIGURE_OUTPUT_DIR / "kernel_bandwidth_top_rank_heatmap.pdf"
HEATMAP_PNG = FIGURE_OUTPUT_DIR / "kernel_bandwidth_top_rank_heatmap.png"
FAIRICP_OUTPUT = OUTPUT_DIR / "baselines" / "fairicp" / "fairicp_probe.csv"
BASELINE_RAW_DIR = OUTPUT_DIR / "raw"
CANDIDATE_METHOD = "fair-reweighing-fixed-kernel"


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARAMETER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or DATASETS
    bandwidths = args.bandwidths or BANDWIDTH_GRID
    bandwidths = [round(float(value), 2) for value in bandwidths]
    seeds = list(range(args.first_seed, args.first_seed + args.repeats))

    raw = run_bandwidth_grid(datasets, bandwidths, seeds, resume=not args.no_resume)
    verify_counts(raw, datasets, bandwidths, seeds)
    raw.to_csv(RAW_OUTPUT, index=False)

    baseline = read_baseline_rows(datasets)
    summary, pairwise = rank_bandwidths(raw, baseline, bandwidths)
    matrix = build_rank_matrix(summary)
    chosen = choose_universal_bandwidth(matrix)

    summary["chosen_universal_bandwidth"] = np.isclose(summary["bandwidth"], chosen) if np.isfinite(chosen) else False
    matrix["chosen_universal_bandwidth"] = np.isclose(matrix["bandwidth"], chosen) if np.isfinite(chosen) else False

    summary.to_csv(SUMMARY_OUTPUT, index=False)
    pairwise.to_csv(PAIRWISE_OUTPUT, index=False)
    matrix.to_csv(MATRIX_OUTPUT, index=False)
    plot_heatmap(matrix, HEATMAP_PDF)
    plot_heatmap(matrix, HEATMAP_PNG)

    print(f"Wrote raw rows to {RAW_OUTPUT}")
    print(f"Wrote summary to {SUMMARY_OUTPUT}")
    print(f"Wrote pairwise tests to {PAIRWISE_OUTPUT}")
    print(f"Wrote matrix to {MATRIX_OUTPUT}")
    print(f"Wrote heatmaps to {HEATMAP_PDF} and {HEATMAP_PNG}")
    if np.isfinite(chosen):
        print(f"Universal rank-0 bandwidth found: {chosen:.2f}")
    else:
        print("No bandwidth in [0.10, 0.20] is rank 0 for all checked datasets.")
    print(matrix.to_string(index=False))


def parse_args():
    parser = ArgumentParser(description="Dense Kernel bandwidth top-rank check.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--bandwidths", nargs="*", type=float, default=None)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def run_bandwidth_grid(datasets, bandwidths, seeds, resume=True):
    existing = read_existing_candidate_rows() if resume else pd.DataFrame()
    done = set()
    if not existing.empty:
        for _, row in existing.iterrows():
            done.add((str(row["dataset"]), round(float(row["bandwidth"]), 2), int(row["seed"])))

    rows = []
    total = len(datasets) * len(bandwidths) * len(seeds)
    completed = len(done)
    for dataset in datasets:
        for bandwidth in bandwidths:
            for seed in seeds:
                key = (dataset, round(float(bandwidth), 2), int(seed))
                if key in done:
                    continue
                rows.append(run_one(dataset, bandwidth, seed))
                completed += 1
                if completed % 20 == 0:
                    checkpoint = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False)
                    checkpoint = deduplicate_candidate_rows(checkpoint)
                    checkpoint.to_csv(CHECKPOINT_OUTPUT, index=False)
                    print(f"Completed {completed}/{total} dense Kernel runs", flush=True)

    result = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False)
    result = deduplicate_candidate_rows(result)
    if not result.empty:
        result.to_csv(CHECKPOINT_OUTPUT, index=False)
    return result


def read_existing_candidate_rows():
    frames = []
    for path in (CHECKPOINT_OUTPUT, RAW_OUTPUT):
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return deduplicate_candidate_rows(pd.concat(frames, ignore_index=True, sort=False))


def deduplicate_candidate_rows(raw):
    if raw.empty:
        return raw
    raw = raw.copy()
    raw["bandwidth"] = pd.to_numeric(raw["bandwidth"], errors="coerce").round(2)
    raw["seed"] = pd.to_numeric(raw["seed"], errors="coerce").astype("Int64")
    return raw.drop_duplicates(subset=["dataset", "bandwidth", "seed"], keep="last")


def run_one(dataset, bandwidth, seed):
    experiment = Experiment(
        data=dataset,
        regressor="linear",
        balance="fair-reweighing",
        density_model="Kernel",
        seed=seed,
    )
    experiment.select_density_param = lambda bandwidth=bandwidth: bandwidth
    row = experiment.run()
    row["method"] = CANDIDATE_METHOD
    row["density_model"] = candidate_density_name(bandwidth)
    row["bandwidth"] = round(float(bandwidth), 2)
    row["parameter_mode"] = "fixed_kernel_dense"
    row["selected_bandwidth"] = round(float(bandwidth), 2)
    return row


def verify_counts(raw, datasets, bandwidths, seeds):
    expected = len(seeds)
    counts = raw.groupby(["dataset", "bandwidth"], dropna=False)["seed"].nunique()
    missing = []
    for dataset in datasets:
        for bandwidth in bandwidths:
            count = int(counts.get((dataset, round(float(bandwidth), 2)), 0))
            if count != expected:
                missing.append(f"{dataset} h={bandwidth:.2f}: {count}/{expected}")
    if missing:
        raise RuntimeError("Incomplete dense Kernel runs: " + "; ".join(missing))


def read_baseline_rows(datasets):
    frames = []
    for dataset in datasets:
        path = BASELINE_RAW_DIR / f"{dataset}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline raw file: {path}")
        frames.append(read_result_csv(path))
    fairicp = read_fairicp_rows(datasets)
    if not fairicp.empty:
        frames.append(fairicp)
    baseline = pd.concat(frames, ignore_index=True, sort=False)
    return baseline[baseline["dataset"].isin(datasets)].copy()


def read_fairicp_rows(datasets):
    if not FAIRICP_OUTPUT.exists():
        return pd.DataFrame()
    fairicp = read_result_csv(FAIRICP_OUTPUT)
    if "method" not in fairicp.columns:
        return pd.DataFrame()
    fairicp = fairicp[fairicp["dataset"].isin(datasets) & fairicp["method"].eq("fairicp")].copy()
    if fairicp.empty:
        return fairicp
    fairicp["density_model"] = "none"
    fairicp["seed"] = pd.to_numeric(fairicp["seed"], errors="coerce").astype("Int64")
    fairicp = fairicp.dropna(subset=["seed"])
    return fairicp.drop_duplicates(subset=["dataset", "method", "density_model", "seed"], keep="last")


def read_result_csv(path):
    result = pd.read_csv(path)
    result.columns = [str(column).strip() for column in result.columns]
    for column in ("dataset", "model", "method", "density_model"):
        if column in result.columns:
            result[column] = result[column].astype(str).str.strip()
    return result


def rank_bandwidths(raw, baseline, bandwidths):
    summary_rows = []
    pairwise_rows = []
    for dataset in DATASETS:
        dataset_baseline = baseline[baseline["dataset"].eq(dataset)].copy()
        for bandwidth in bandwidths:
            candidate = raw[
                raw["dataset"].eq(dataset)
                & np.isclose(raw["bandwidth"].astype(float), float(bandwidth))
            ].copy()
            candidate["method"] = CANDIDATE_METHOD
            candidate["density_model"] = candidate_density_name(bandwidth)
            combined = pd.concat([dataset_baseline, candidate], ignore_index=True, sort=False)
            treatments = treatment_series(combined)
            candidate_key = (CANDIDATE_METHOD, candidate_density_name(bandwidth))
            if candidate_key not in treatments:
                raise RuntimeError(f"Missing candidate treatment for {dataset} h={bandwidth:.2f}")
            ranks = recursive_ranks(treatments)
            reference_key = rank_zero_reference(treatments, ranks)
            paired = paired_by_seed(treatments[candidate_key], treatments[reference_key])
            candidate_values = paired["row"].to_numpy(dtype=float)
            reference_values = paired["column"].to_numpy(dtype=float)
            test = paired_wilcoxon(candidate_values, reference_values)
            delta = cliffs_delta(candidate_values, reference_values)
            metric_summary = summarize_candidate_metrics(candidate)
            rank = int(ranks[candidate_key])
            summary_rows.append({
                "dataset": dataset,
                "display_dataset": display_dataset(dataset),
                "bandwidth": round(float(bandwidth), 2),
                "n": int(candidate["seed"].nunique()),
                **metric_summary,
                "rank": rank,
                "top_rank": rank == 0,
                "rank0_reference_method": reference_key[0],
                "rank0_reference_density": reference_key[1],
                "rank0_reference_mean_c_sep_xfit": float(treatments[reference_key]["c_sep_xfit"].mean()),
            })
            pairwise_rows.append({
                "dataset": dataset,
                "display_dataset": display_dataset(dataset),
                "bandwidth": round(float(bandwidth), 2),
                "candidate_rank": rank,
                "candidate_top_rank": rank == 0,
                "candidate_mean_c_sep_xfit": float(treatments[candidate_key]["c_sep_xfit"].mean()),
                "reference_method": reference_key[0],
                "reference_density": reference_key[1],
                "reference_mean_c_sep_xfit": float(treatments[reference_key]["c_sep_xfit"].mean()),
                "n": int(len(paired)),
                "wilcoxon_stat": test["statistic"],
                "p_value": test["p_value"],
                "cliffs_delta": delta,
                "conclusion": pairwise_conclusion(rank),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(pairwise_rows)


def summarize_candidate_metrics(candidate):
    row = {}
    for metric in ("mse", "r2", "c_sep_xfit"):
        values = pd.to_numeric(candidate[metric], errors="coerce").dropna()
        row[f"mean_{metric}"] = float(values.mean()) if not values.empty else np.nan
        row[f"std_{metric}"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
    return row


def treatment_series(group):
    treatments = {}
    for key, treatment in group.groupby(["method", "density_model"], dropna=False):
        values = treatment[["seed", "c_sep_xfit"]].copy()
        values["seed"] = pd.to_numeric(values["seed"], errors="coerce")
        values["c_sep_xfit"] = pd.to_numeric(values["c_sep_xfit"], errors="coerce")
        values = values.dropna()
        if not values.empty:
            treatments[key] = values
    return treatments


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


def rank_zero_reference(treatments, ranks):
    rank_zero = [key for key, rank in ranks.items() if rank == 0]
    return sorted(
        rank_zero,
        key=lambda key: (
            float(treatments[key]["c_sep_xfit"].mean()),
            treatment_display_name(*key),
        ),
    )[0]


def paired_by_seed(row_values, column_values):
    paired = row_values.merge(column_values, on="seed", suffixes=("_row", "_column")).dropna()
    return paired.rename(
        columns={
            "c_sep_xfit_row": "row",
            "c_sep_xfit_column": "column",
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


def build_rank_matrix(summary):
    rows = []
    for bandwidth, group in summary.groupby("bandwidth", dropna=False):
        row = {"bandwidth": float(bandwidth)}
        top_count = 0
        for dataset in DATASETS:
            match = group[group["dataset"].eq(dataset)]
            if match.empty:
                row[f"{dataset}_rank"] = np.nan
                row[f"{dataset}_top_rank"] = False
                continue
            rank = int(match.iloc[0]["rank"])
            row[f"{dataset}_rank"] = rank
            row[f"{dataset}_top_rank"] = rank == 0
            top_count += int(rank == 0)
        row["top_rank_count"] = top_count
        row["all_top_rank"] = top_count == len(DATASETS)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("bandwidth").reset_index(drop=True)


def choose_universal_bandwidth(matrix):
    candidates = matrix[matrix["all_top_rank"]].copy()
    if candidates.empty:
        return np.nan
    candidates["distance"] = (candidates["bandwidth"] - 0.15).abs()
    chosen = candidates.sort_values(["distance", "bandwidth"]).iloc[0]
    return float(chosen["bandwidth"])


def plot_heatmap(matrix, output_path):
    rank_columns = [f"{dataset}_rank" for dataset in DATASETS]
    values = matrix[rank_columns].to_numpy(dtype=float)
    max_rank = int(np.nanmax(values)) if np.isfinite(values).any() else 1
    colors = ["#2ca25f", "#fee08b", "#fdae61", "#f46d43", "#d73027", "#762a83"]
    if max_rank + 1 > len(colors):
        colors.extend(["#4d004b"] * (max_rank + 1 - len(colors)))
    cmap = ListedColormap(colors[: max_rank + 1])
    norm = BoundaryNorm(np.arange(-0.5, max_rank + 1.5, 1), cmap.N)

    fig, axis = plt.subplots(figsize=(9.2, 4.9))
    image = axis.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    axis.set_xticks(np.arange(len(DATASETS)))
    axis.set_xticklabels([display_dataset(dataset) for dataset in DATASETS], rotation=35, ha="right", fontsize=8)
    axis.set_yticks(np.arange(len(matrix)))
    axis.set_yticklabels([f"{bandwidth:.2f}" for bandwidth in matrix["bandwidth"]], fontsize=8)
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Fixed Kernel bandwidth")
    axis.set_title("Dense Kernel Bandwidth Rank-0 Check")
    for row_index in range(values.shape[0]):
        for col_index in range(values.shape[1]):
            rank = values[row_index, col_index]
            axis.text(
                col_index,
                row_index,
                f"R{int(rank)}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if rank == 0 or rank >= 4 else "#111827",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.02, ticks=np.arange(max_rank + 1))
    colorbar.set_ticklabels([f"R{rank}" for rank in range(max_rank + 1)])
    colorbar.set_label("Recursive rank")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def candidate_density_name(bandwidth):
    return f"Kernel-{float(bandwidth):.2f}"


def pairwise_conclusion(rank):
    return "rank 0" if int(rank) == 0 else f"rank {int(rank)}"


def treatment_display_name(method, density):
    method_key = str(method).strip().lower()
    density_key = str(density).strip()
    names = {
        "none": "None",
        "fair-reweighing": f"FR-{density_key[0].upper()}" if density_key else "FR",
        "fair-reweighing-fixed-kernel": f"FR-K {density_key.replace('Kernel-', '')}",
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


def display_dataset(dataset):
    names = {
        "Community": "Communities",
        "Community_Con": "Communities-Con",
        "Community_Multi": "Communities-Multi",
        "LSAC_Multi": "LSAC-Multi",
        "Insurance_Multi": "Insurance-Multi",
    }
    return names.get(dataset, dataset.replace("_", "-"))


if __name__ == "__main__":
    main()
