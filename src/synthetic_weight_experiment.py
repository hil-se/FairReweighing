"""Standalone synthetic ground-truth weight check."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, wilcoxon
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from synthetic_config import (
    N_SYNTHETIC,
    P_A1,
    synthetic_height_mean,
    synthetic_height_sd,
    synthetic_power_mean,
    synthetic_power_sd,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

EPS = 1e-12
RANK_METRIC = "cosine_similarity"
SEEDS = range(1, 21)
OUTPUT_DIR = Path("result/synthetic_weight_experiment")
FIGURE_DIR = Path("result/figures")
GRID = [round(x, 1) for x in np.arange(0.1, 1.01, 0.1)]
PARAM_GRID = {
    "Neighbor": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    "Kernel": [0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
}
BASELINE = {
    "p_a1": P_A1,
    "height_mean": {0: synthetic_height_mean(0), 1: synthetic_height_mean(1)},
    "height_sd": {0: synthetic_height_sd(0), 1: synthetic_height_sd(1)},
    "power_mean": {0: synthetic_power_mean(0), 1: synthetic_power_mean(1)},
    "power_sd": {0: synthetic_power_sd(0), 1: synthetic_power_sd(1)},
}
SCENARIOS = {
    "baseline": BASELINE,
    "balanced_groups": {**BASELINE, "p_a1": 0.5},
    "imbalanced_groups": {**BASELINE, "p_a1": 0.9},
    "weak_gap": {
        **BASELINE,
        "height_mean": {0: 1.65, 1: 1.70},
        "power_mean": {0: 0.35, 1: 0.38},
    },
    "strong_gap": {
        **BASELINE,
        "height_mean": {0: 1.55, 1: 1.85},
        "power_mean": {0: 0.25, 1: 0.65},
    },
    "low_noise": {
        **BASELINE,
        "height_sd": {0: 0.075, 1: 0.10},
        "power_sd": {0: 0.025, 1: 0.05},
    },
    "high_noise": {
        **BASELINE,
        "height_sd": {0: 0.30, 1: 0.40},
        "power_sd": {0: 0.10, 1: 0.20},
    },
    "heterogeneous_noise": {
        **BASELINE,
        "height_sd": {0: 0.08, 1: 0.30},
        "power_sd": {0: 0.03, 1: 0.15},
    },
    "rare_tail": {
        **BASELINE,
        "tail": {"group": 1, "prob": 0.10, "power_mean": 0.80, "power_sd": 0.10},
    },
}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    raw = run_all(SEEDS, GRID, SCENARIOS["baseline"])
    summary, pairwise = rank_all(raw, GRID)
    raw.to_csv(OUTPUT_DIR / "synthetic_weight_parameter_raw.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "synthetic_weight_parameter_summary.csv", index=False)
    pairwise.to_csv(OUTPUT_DIR / "synthetic_weight_parameter_pairwise.csv", index=False)

    sweep_raw = run_scenarios()
    sweep_summary, sweep_pairwise = rank_all(sweep_raw, GRID, group_cols=("scenario", "estimator"))
    rank_matrix = make_rank_matrix(sweep_summary)
    sweep_raw.to_csv(OUTPUT_DIR / "synthetic_model_parameter_raw.csv", index=False)
    sweep_summary.to_csv(OUTPUT_DIR / "synthetic_model_parameter_summary.csv", index=False)
    sweep_pairwise.to_csv(OUTPUT_DIR / "synthetic_model_parameter_pairwise.csv", index=False)
    rank_matrix.to_csv(OUTPUT_DIR / "synthetic_model_parameter_rank_matrix.csv", index=False)
    plot_rank_heatmap(rank_matrix)


def run_scenarios():
    pieces = []
    for name, scenario in SCENARIOS.items():
        pieces.append(run_all(SEEDS, GRID, scenario, scenario_name=name))
    return pd.concat(pieces, ignore_index=True)


def run_all(seeds, grid, scenario, scenario_name=None):
    rows = []
    for seed in seeds:
        X, y = make_synthetic(seed, scenario)
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.5, random_state=seed)
        X_train = X_train.reset_index(drop=True)
        y_train = np.asarray(y_train, dtype=float)

        A = X_train[["sex"]].to_numpy()
        y_col = y_train.reshape(-1, 1)
        true_w = true_synthetic_weight(A, y_col, scenario)
        weights = WeightComputer(A, y_col)

        for estimator in ("Neighbor", "Kernel"):
            for value in grid:
                rows.append(score(seed, estimator, "fixed", value, weights, true_w, scenario_name))

            # Dynamic uses the same train-only validation idea as the main experiments.
            dynamic = select_parameter(estimator, X_train, y_train, seed)
            rows.append(score(seed, estimator, "dynamic", dynamic, weights, true_w, scenario_name))

    return pd.DataFrame(rows)


def make_synthetic(seed, scenario, n=N_SYNTHETIC):
    rng = np.random.default_rng(seed)
    rows = []
    y = []
    for _ in range(n):
        sex = 1 if rng.random() < scenario["p_a1"] else 0
        height = rng.normal(scenario["height_mean"][sex], scenario["height_sd"][sex])
        power = sample_power(rng, sex, scenario)
        rows.append({"sex": sex, "height": height})
        y.append(height + power)
    return pd.DataFrame(rows, columns=["sex", "height"]), np.asarray(y)


def sample_power(rng, sex, scenario):
    tail = scenario.get("tail")
    if tail and sex == tail["group"] and rng.random() < tail["prob"]:
        return rng.normal(tail["power_mean"], tail["power_sd"])
    return rng.normal(scenario["power_mean"][sex], scenario["power_sd"][sex])


def score(seed, estimator, mode, parameter, weights, true_w, scenario_name=None):
    est_w = weights.get(estimator, parameter)
    row = {
        "seed": seed,
        "estimator": estimator,
        "parameter_mode": mode,
        "parameter_label": "Dynamic" if mode == "dynamic" else f"{parameter:.1f}",
        "parameter_value": float(parameter),
        "cosine_similarity": cosine(est_w, true_w),
        "pearson_correlation": pearson(est_w, true_w),
        "n_samples": len(true_w),
    }
    if scenario_name is not None:
        row = {"scenario": scenario_name, **row}
    return row


class WeightComputer:
    def __init__(self, A, y):
        A = np.asarray(A).reshape(len(A), -1)
        y = np.asarray(y).reshape(-1, 1)
        joint = np.column_stack([A, y])
        self.distances = {
            "a": pairwise_distances(standardize(A)).astype(np.float32),
            "y": pairwise_distances(standardize(y)).astype(np.float32),
            "joint": pairwise_distances(standardize(joint)).astype(np.float32),
        }

    def get(self, estimator, parameter):
        if estimator == "Neighbor":
            density = self.neighbor_density
        elif estimator == "Kernel":
            density = self.kernel_density
        else:
            raise ValueError(f"Unknown estimator: {estimator}")

        w = density("a", parameter) * density("y", parameter)
        w = w / np.maximum(density("joint", parameter), EPS)
        return normalize(w)

    def neighbor_density(self, name, radius):
        return np.sum(self.distances[name] < radius, axis=1).astype(float)

    def kernel_density(self, name, bandwidth):
        d2 = self.distances[name] ** 2
        return np.exp(-d2 / (2.0 * bandwidth ** 2)).sum(axis=1)


def fair_weights(estimator, parameter, A, y):
    return WeightComputer(A, y).get(estimator, parameter)


def standardize(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return StandardScaler().fit_transform(X)


def select_parameter(estimator, X_train, y_train, seed):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=seed + 17,
    )
    X_fit = X_fit.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)

    scaler = StandardScaler().fit(X_fit)
    X_fit_scaled = scaler.transform(X_fit)
    X_val_scaled = scaler.transform(X_val)

    weights = WeightComputer(X_fit[["sex"]].to_numpy(), y_fit.reshape(-1, 1))
    best_param = PARAM_GRID[estimator][0]
    best_score = float("inf")
    for param in PARAM_GRID[estimator]:
        sample_weight = weights.get(estimator, param)
        model = LinearRegression().fit(X_fit_scaled, y_fit, sample_weight=sample_weight)
        pred = model.predict(X_val_scaled)
        score = csep_xfit(y_val, pred, X_val["sex"].to_numpy(), seed) + 0.01 * mean_squared_error(y_val, pred)
        if score < best_score:
            best_score = score
            best_param = param
    return best_param


def csep_xfit(y, pred, sensitive, seed, n_splits=10):
    y = np.asarray(y)
    pred = np.asarray(pred)
    sensitive = np.asarray(sensitive)
    if len(np.unique(sensitive)) <= 1:
        return 0.0

    joint = pd.DataFrame({"y": y, "pred": pred})
    margin = y.reshape(-1, 1)
    log_joint = []
    log_margin = []
    for train, test in folds(sensitive, seed, n_splits):
        joint_model = LogisticRegression(max_iter=1000).fit(joint.iloc[train], sensitive[train])
        margin_model = LogisticRegression(max_iter=1000).fit(margin[train], sensitive[train])
        log_joint.extend(np.log(actual_class_prob(joint_model, joint.iloc[test], sensitive[test])))
        log_margin.extend(np.log(actual_class_prob(margin_model, margin[test], sensitive[test])))

    return max(0.0, float(np.mean(np.asarray(log_joint) - np.asarray(log_margin))))


def folds(values, seed, n_splits):
    values = np.asarray(values)
    k = min(n_splits, len(values))
    counts = pd.Series(values).value_counts()
    if counts.min() >= 2:
        k = min(k, int(counts.min()))
        return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(values)), values)
    return KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(values)))


def actual_class_prob(model, X, labels):
    probs = model.predict_proba(X)
    class_index = {label: idx for idx, label in enumerate(model.classes_)}
    selected = [probs[row, class_index[label]] for row, label in enumerate(labels)]
    return np.clip(np.asarray(selected), EPS, 1 - EPS)


def true_synthetic_weight(A, y, scenario):
    A = np.asarray(A, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    y_a0 = y_density_given_a(y, 0, scenario)
    y_a1 = y_density_given_a(y, 1, scenario)
    y_marginal = scenario["p_a1"] * y_a1 + (1.0 - scenario["p_a1"]) * y_a0
    y_given_a = np.where(A == 1, y_a1, y_a0)

    # FairReweighing's population target is w*(a,y) = p(y) / p(y|a).
    return normalize(y_marginal / np.maximum(y_given_a, EPS))


def y_density_given_a(y, sex, scenario):
    height_mean = scenario["height_mean"][sex]
    height_var = scenario["height_sd"][sex] ** 2
    base_mean = height_mean + scenario["power_mean"][sex]
    base_sd = np.sqrt(height_var + scenario["power_sd"][sex] ** 2)
    density_value = norm.pdf(y, loc=base_mean, scale=base_sd)

    tail = scenario.get("tail")
    if tail and sex == tail["group"]:
        tail_mean = height_mean + tail["power_mean"]
        tail_sd = np.sqrt(height_var + tail["power_sd"] ** 2)
        density_value = (
            (1.0 - tail["prob"]) * density_value
            + tail["prob"] * norm.pdf(y, loc=tail_mean, scale=tail_sd)
        )
    return density_value


def normalize(w):
    w = np.asarray(w, dtype=float).reshape(-1)
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(w))
    return len(w) * w / total


def cosine(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return np.nan if denom <= 0 or not np.isfinite(denom) else float(np.dot(x, y) / denom)


def pearson(x, y):
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = y[keep]
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_all(raw, grid, group_cols=("estimator",)):
    summary = []
    pairwise = []

    groupby_cols = group_cols[0] if len(group_cols) == 1 else list(group_cols)
    for group_key, group in raw.groupby(groupby_cols, sort=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        group_values = dict(zip(group_cols, group_key))
        treatments = {
            key: value[["seed", RANK_METRIC]].copy()
            for key, value in group.groupby(["parameter_mode", "parameter_label"], sort=False)
        }
        ranks = assign_ranks(treatments, grid)
        ref = best_rank0(treatments, ranks, grid)

        for key, values in treatments.items():
            rows = group[group["parameter_mode"].eq(key[0]) & group["parameter_label"].eq(key[1])]
            paired = paired_values(values, treatments[ref])
            candidate = paired["candidate"].to_numpy(dtype=float)
            reference = paired["reference"].to_numpy(dtype=float)
            test = wilcoxon_test(candidate, reference)
            delta = cliffs_delta(candidate, reference)

            common = {
                **group_values,
                "parameter_mode": key[0],
                "parameter_label": key[1],
                "parameter_value": float(rows["parameter_value"].mean()),
                "plot_order": order(key, grid),
                "rank": int(ranks[key]),
            }
            summary.append({
                **common,
                "seeds": int(rows["seed"].nunique()),
                "pearson_correlation": float(rows["pearson_correlation"].mean()),
                "pearson_correlation_std": float(rows["pearson_correlation"].std(ddof=1)),
                "cosine_similarity": float(rows["cosine_similarity"].mean()),
                "cosine_similarity_std": float(rows["cosine_similarity"].std(ddof=1)),
                "top_rank": int(ranks[key]) == 0,
                "rank0_reference": ref[1],
                f"rank0_reference_{RANK_METRIC}": mean_metric(treatments[ref]),
            })
            pairwise.append({
                **common,
                f"candidate_mean_{RANK_METRIC}": float(rows[RANK_METRIC].mean()),
                "reference_parameter_label": ref[1],
                f"reference_mean_{RANK_METRIC}": mean_metric(treatments[ref]),
                "n": len(paired),
                "wilcoxon_stat": test["statistic"],
                "p_value": test["p_value"],
                "cliffs_delta": delta,
            })

    return (
        pd.DataFrame(summary).sort_values(list(group_cols) + ["plot_order"]),
        pd.DataFrame(pairwise).sort_values(list(group_cols) + ["plot_order"]),
    )


def make_rank_matrix(summary):
    matrix = summary.copy()
    matrix["setting"] = matrix["estimator"] + " " + matrix["parameter_label"]
    return matrix.pivot(index="scenario", columns="setting", values="rank").reset_index()


def plot_rank_heatmap(rank_matrix):
    labels = [column for column in rank_matrix.columns if column != "scenario"]
    values = rank_matrix[labels].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    image = ax.imshow(values, cmap="YlGnBu_r", aspect="auto", vmin=0, vmax=np.nanmax(values))
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(rank_matrix)))
    ax.set_yticklabels(rank_matrix["scenario"], fontsize=9)
    ax.set_xlabel("Estimator and parameter")
    ax.set_ylabel("Synthetic scenario")
    ax.set_title("Synthetic model parameter sweep: cosine rank")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"R{int(values[row, col])}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="Rank")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "synthetic_model_parameter_sweep_heatmap.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / "synthetic_model_parameter_sweep_heatmap.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def assign_ranks(treatments, grid):
    remaining = sorted(treatments, key=lambda key: (-mean_metric(treatments[key]), order(key, grid)))
    ranks = {}
    rank = 0

    while remaining:
        ref = remaining[0]
        ranks[ref] = rank
        next_round = []
        found_worse = False

        for key in remaining[1:]:
            paired = paired_values(treatments[key], treatments[ref])
            test = wilcoxon_test(paired["candidate"], paired["reference"])
            delta = cliffs_delta(paired["candidate"], paired["reference"])
            # Higher weight similarity is better, so only a meaningfully lower candidate
            # moves to a worse rank.
            worse = test["p_value"] < 0.05 and delta < -0.33

            if found_worse or worse:
                found_worse = True
                next_round.append(key)
            else:
                ranks[key] = rank

        remaining = next_round
        rank += 1

    return ranks


def best_rank0(treatments, ranks, grid):
    keys = [key for key, rank in ranks.items() if rank == 0]
    return sorted(keys, key=lambda key: (-mean_metric(treatments[key]), order(key, grid)))[0]


def paired_values(candidate, reference):
    paired = candidate.merge(reference, on="seed", suffixes=("_candidate", "_reference")).dropna()
    return paired.rename(
        columns={
            f"{RANK_METRIC}_candidate": "candidate",
            f"{RANK_METRIC}_reference": "reference",
        }
    )


def wilcoxon_test(candidate, reference):
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if len(candidate) == 0 or np.allclose(candidate - reference, 0.0):
        return {"statistic": 0.0, "p_value": 1.0}
    result = wilcoxon(candidate, reference, alternative="two-sided")
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def cliffs_delta(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    return float(np.sign(x[:, None] - y[None, :]).sum() / (len(x) * len(y)))


def mean_metric(rows):
    return float(rows[RANK_METRIC].mean())


def order(key, grid):
    mode, label = key
    return len(grid) + 1 if mode == "dynamic" else float(label)


if __name__ == "__main__":
    main()
