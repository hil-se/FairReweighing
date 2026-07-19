"""Run FairICP on this project's tabular and SCUT feature splits.

This is intentionally separate from the main experiment grid. It uses the
authors' FairICP learner from a local clone and writes raw, summary, and
comparison CSVs for manuscript reporting.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_reader import load_dataset
from experiment import DEFAULT_TEST_SIZE, numeric_sensitive_frame, safe_metric_name
from metrics import Metrics

torch = None

DEFAULT_DATASETS = [
    "German",
    "Heart",
    "Synthetic",
    "LSAC",
    "LSAC_Multi",
    "Community",
    "Community_Con",
    "Community_Multi",
    "Insurance",
    "Insurance_Multi",
]
DEFAULT_SEEDS = list(range(1, 21))
SCUT_DATASETS = {"SCUT_Sex", "SCUT_Multi"}
SCUT_DEFAULT_SEEDS = list(range(1, 21))
_SCUT_ENCODER = None
_SCUT_PATH_FEATURES = {}
_SCUT_FEATURE_CACHE_LOADED = False
_SCUT_FEATURE_CACHE_PATH = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fairicp-root", default="/private/tmp/FairICP")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda-vec", type=float, default=0.3)
    parser.add_argument("--model-type", choices=["linear_model", "deep_model"], default="linear_model")
    parser.add_argument("--output", default="result/baselines/fairicp/fairicp_probe.csv")
    parser.add_argument("--summary-output", default="result/baselines/fairicp/fairicp_probe_summary.csv")
    parser.add_argument("--comparison-output", default="result/baselines/fairicp/fairicp_probe_comparison.csv")
    parser.add_argument("--project-summary", default="result/main/summary.csv")
    parser.add_argument("--scut-feature-cache", default="result/cache/scut_vgg_face_features.npz")
    return parser.parse_args()


def main():
    global torch
    args = parse_args()
    add_fairicp_to_path(args.fairicp_root)
    import torch as torch_module
    from FairICP import FairICP_learning

    torch = torch_module
    torch.set_num_threads(1)
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        seeds = args.seeds or default_seeds_for_dataset(dataset)
        for seed in seeds:
            rows.append(run_one(dataset, seed, args, FairICP_learning))
            pd.DataFrame(rows).to_csv(output, index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(output, index=False)
    write_summary_outputs(raw, args)
    print(f"Wrote {len(rows)} FairICP probe rows to {output}")


def add_fairicp_to_path(root):
    root_path = Path(root)
    if not (root_path / "FairICP" / "FairICP_learning.py").exists():
        raise FileNotFoundError(f"FairICP clone not found at {root_path}")
    sys.path.insert(0, str(root_path))


def run_one(dataset, seed, args, FairICP_learning):
    set_seed(seed)
    load_options = {"target": "P1"} if is_scut_dataset(dataset) else {}
    X, y, protected = load_dataset(dataset, seed=seed, **load_options)
    task_type = "classification" if len(np.unique(y)) == 2 else "regression"
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=seed,
        stratify=stratify,
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    A_train = numeric_sensitive_frame(X_train[protected])
    A_test = numeric_sensitive_frame(X_test[protected])
    X_train_model, X_test_model = preprocess_without_sensitive(X_train, X_test, protected, args)
    train_input = np.concatenate([A_train, X_train_model], axis=1).astype(np.float32)
    test_input = np.concatenate([A_test, X_test_model], axis=1).astype(np.float32)

    start = time.perf_counter()
    if task_type == "classification":
        y_pred = fit_classification(FairICP_learning, train_input, y_train, test_input, args, A_train.shape[1])
        permutation_note = "official_class_density"
    else:
        y_pred = fit_regression(
            FairICP_learning,
            train_input,
            y_train,
            test_input,
            args,
            A_train.shape[1],
            seed,
        )
        permutation_note = (
            "vgg_face_feature_quantile_bin_permutation"
            if is_scut_dataset(dataset)
            else "quantile_bin_permutation_regression_adaptation"
        )
    fit_seconds = time.perf_counter() - start

    result = evaluate(y_test, y_pred, X_test, protected, seed=seed)
    result.update({
        "dataset": dataset,
        "model": f"FairICP-{args.model_type}",
        "method": "fairicp",
        "density_model": "none",
        "seed": seed,
        "lambda_vec": args.lambda_vec,
        "epochs": args.epochs,
        "batch_size": effective_batch_size(len(y_train), args.batch_size),
        "fit_seconds": fit_seconds,
        "fairicp_permutation": permutation_note,
    })
    print(
        f"{dataset} seed={seed}: c_sep={result['c_sep_xfit']:.4f}, "
        f"mse={result['mse']:.4f}, r2={result['r2']:.4f}, seconds={fit_seconds:.1f}"
    )
    return result


def preprocess_without_sensitive(X_train, X_test, protected, args):
    if "image_path" in X_train.columns:
        return preprocess_scut_features(X_train, X_test, args)

    feature_train = X_train.drop(columns=list(protected), errors="ignore")
    feature_test = X_test.drop(columns=list(protected), errors="ignore")
    feature_train = feature_train.copy()
    feature_test = feature_test.copy()
    feature_train.columns = [str(column) for column in feature_train.columns]
    feature_test.columns = [str(column) for column in feature_test.columns]
    numerical_columns = selector(dtype_exclude=object)(feature_train)
    categorical_columns = selector(dtype_include=object)(feature_train)
    preprocessor = ColumnTransformer([
        ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
        ("StandardScaler", StandardScaler(), numerical_columns),
    ], remainder="drop")
    preprocessor.fit(feature_train)
    return preprocessor.transform(feature_train), preprocessor.transform(feature_test)


def preprocess_scut_features(X_train, X_test, args):
    train_features = scut_vgg_features(X_train["image_path"].astype(str).to_numpy(), args)
    test_features = scut_vgg_features(X_test["image_path"].astype(str).to_numpy(), args)
    scaler = StandardScaler()
    scaler.fit(train_features)
    return scaler.transform(train_features), scaler.transform(test_features)


def scut_vgg_features(paths, args):
    from vgg_face_model import _encoded_features, build_vgg_face_feature_extractor

    global _SCUT_ENCODER
    load_scut_feature_cache(args.scut_feature_cache)
    if _SCUT_ENCODER is None:
        _SCUT_ENCODER = build_vgg_face_feature_extractor()

    paths = np.asarray(paths, dtype=str)
    missing = [path for path in paths if path not in _SCUT_PATH_FEATURES]
    if missing:
        features = _encoded_features(_SCUT_ENCODER, np.asarray(missing, dtype=str), verbose=1)
        for path, feature in zip(missing, features):
            _SCUT_PATH_FEATURES[path] = feature.astype(np.float32)
        save_scut_feature_cache()
    return np.stack([_SCUT_PATH_FEATURES[path] for path in paths]).astype(np.float32)


def load_scut_feature_cache(cache_path):
    global _SCUT_FEATURE_CACHE_LOADED, _SCUT_FEATURE_CACHE_PATH
    if _SCUT_FEATURE_CACHE_LOADED:
        return
    _SCUT_FEATURE_CACHE_LOADED = True
    _SCUT_FEATURE_CACHE_PATH = Path(cache_path)
    if not _SCUT_FEATURE_CACHE_PATH.exists():
        return
    cached = np.load(_SCUT_FEATURE_CACHE_PATH, allow_pickle=False)
    paths = cached["paths"].astype(str)
    features = cached["features"].astype(np.float32)
    for path, feature in zip(paths, features):
        _SCUT_PATH_FEATURES[path] = feature
    print(f"Loaded {len(_SCUT_PATH_FEATURES)} SCUT VGG-Face features from {_SCUT_FEATURE_CACHE_PATH}")


def save_scut_feature_cache():
    if _SCUT_FEATURE_CACHE_PATH is None:
        return
    _SCUT_FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    paths = np.asarray(list(_SCUT_PATH_FEATURES.keys()), dtype=str)
    features = np.stack([_SCUT_PATH_FEATURES[path] for path in paths]).astype(np.float32)
    np.savez_compressed(_SCUT_FEATURE_CACHE_PATH, paths=paths, features=features)
    print(f"Saved {len(paths)} SCUT VGG-Face features to {_SCUT_FEATURE_CACHE_PATH}")


def fit_classification(FairICP_learning, train_input, y_train, test_input, args, a_shape):
    learner = FairICP_learning.EquiClassLearner(
        lr_loss=1e-3,
        lr_dis=1e-4,
        epochs=args.epochs,
        loss_steps=1,
        dis_steps=1 if args.lambda_vec > 0 else 0,
        cost_pred=torch.nn.CrossEntropyLoss(),
        in_shape=train_input.shape[1] - a_shape,
        batch_size=effective_batch_size(len(y_train), args.batch_size),
        model_type=args.model_type,
        lambda_vec=args.lambda_vec,
        num_classes=len(np.unique(y_train)),
        A_shape=a_shape,
    )
    learner.fit(train_input, y_train, epochs_list=[args.epochs])
    probs = learner.predict(test_input)
    return probs[:, -1]


def fit_regression(FairICP_learning, train_input, y_train, test_input, args, a_shape, seed):
    learner = FairICP_learning.EquiRegLearner(
        lr_loss=1e-3,
        lr_dis=1e-4,
        epochs=args.epochs,
        loss_steps=1,
        dis_steps=1 if args.lambda_vec > 0 else 0,
        cost_pred=torch.nn.MSELoss(),
        in_shape=train_input.shape[1] - a_shape,
        batch_size=effective_batch_size(len(y_train), args.batch_size),
        model_type=args.model_type,
        lambda_vec=args.lambda_vec,
        out_shape=1,
        A_shape=a_shape,
    )
    A_train = train_input[:, :a_shape]
    specified_at = conditional_sensitive_permutations(A_train, y_train, args.epochs, seed=seed)
    learner.fit(train_input, y_train, epochs_list=[args.epochs], specified_At=specified_at)
    return learner.predict(test_input)


def conditional_sensitive_permutations(A, y, epochs, seed, n_bins=5):
    rng = np.random.default_rng(seed)
    groups = target_groups(y, n_bins)
    copies = []
    for _ in range(epochs):
        A_tilde = A.copy()
        for group in np.unique(groups):
            idx = np.flatnonzero(groups == group)
            if len(idx) > 1:
                A_tilde[idx] = A_tilde[rng.permutation(idx)]
        copies.append(A_tilde)
    return copies


def write_summary_outputs(raw, args):
    summary = summarize_fairicp(raw)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    comparison = compare_with_project_results(summary, Path(args.project_summary))
    comparison_path = Path(args.comparison_output)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    print(f"Wrote FairICP summary to {summary_path}")
    print(f"Wrote FairICP comparison to {comparison_path}")


def summarize_fairicp(raw):
    columns = [
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
        "fit_seconds",
    ]
    rows = []
    for dataset, group in raw.groupby("dataset", sort=False):
        row = {
            "dataset": dataset,
            "n": int(group["seed"].nunique()),
            "fairicp_permutation": "; ".join(sorted(group["fairicp_permutation"].dropna().unique())),
        }
        for column in columns:
            if column in group.columns:
                row[column] = nanmean(group[column].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def compare_with_project_results(summary, project_summary_path):
    project_summary = pd.read_csv(project_summary_path)
    rows = []
    for _, fairicp in summary.iterrows():
        dataset = fairicp["dataset"]
        project = project_summary[project_summary["dataset"] == dataset]
        if project.empty:
            continue
        project_row = project.iloc[0]
        fair_rows = read_project_dataset_results(dataset)
        best_fr = best_fair_reweighing_row(fair_rows)
        rows.append({
            "dataset": dataset,
            "n": int(fairicp["n"]),
            "fairicp_csep": fairicp.get("c_sep_xfit", np.nan),
            "none_csep": project_row["baseline_c_sep_xfit"],
            "best_fr_csep": best_fr.get("c_sep_xfit", np.nan),
            "fairicp_mse": fairicp.get("mse", np.nan),
            "none_mse": project_row["baseline_mse"],
            "best_fr_mse": best_fr.get("mse", np.nan),
            "fairicp_r2": fairicp.get("r2", np.nan),
            "none_r2": project_row["baseline_r2"],
            "best_fr_r2": best_fr.get("r2", np.nan),
            "fairicp_acc": fairicp.get("accuracy", np.nan),
            "best_fr_acc": best_fr.get("accuracy", np.nan),
            "permutation": fairicp["fairicp_permutation"],
        })
    return pd.DataFrame(rows)


def read_project_dataset_results(dataset):
    path = Path("result") / f"{dataset}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def best_fair_reweighing_row(rows):
    if rows.empty or "method" not in rows.columns:
        return {}
    fair_rows = rows[rows["method"] == "fair-reweighing"].copy()
    if fair_rows.empty:
        return {}
    fair_rows = fair_rows.sort_values("c_sep_xfit", na_position="last")
    return fair_rows.iloc[0].to_dict()


def target_groups(y, n_bins):
    y = np.asarray(y)
    if len(np.unique(y)) <= 20:
        return y
    try:
        return np.asarray(pd.qcut(y, q=n_bins, labels=False, duplicates="drop"))
    except ValueError:
        return np.zeros(len(y), dtype=int)


def effective_batch_size(n, requested):
    return max(4, min(int(requested), max(4, n // 2)))


def default_seeds_for_dataset(dataset):
    return SCUT_DEFAULT_SEEDS if is_scut_dataset(dataset) else DEFAULT_SEEDS


def is_scut_dataset(dataset):
    return dataset in SCUT_DATASETS


def evaluate(y_test, y_pred, X_test, protected, seed):
    metrics = Metrics(y_test, y_pred)
    result = {
        "mse": metrics.mse(),
        "mae": metrics.mae(),
        "r2": metrics.r2(),
        "accuracy": metrics.accuracy(),
        "f1": metrics.f1(),
    }
    r_sep = []
    i_sep = []
    c_sep_xfit = []
    aod = []
    eod = []
    for column in protected:
        name = safe_metric_name(column)
        sensitive = X_test[column].to_numpy()
        categorical_sep = metrics.categorical_separation(sensitive, seed=seed)
        equalized_odds = metrics.equalized_odds_metrics(sensitive)
        c_sep = metrics.continuous_mi_crossfit(sensitive, seed=seed)
        result[f"r_sep_{name}"] = categorical_sep["r_sep"]
        result[f"i_sep_{name}"] = categorical_sep["i_sep"]
        result[f"c_sep_xfit_{name}"] = c_sep
        result[f"aod_{name}"] = equalized_odds["aod"]
        result[f"eod_{name}"] = equalized_odds["eod"]
        r_sep.append(categorical_sep["r_sep"])
        i_sep.append(categorical_sep["i_sep"])
        c_sep_xfit.append(c_sep)
        aod.append(equalized_odds["aod"])
        eod.append(equalized_odds["eod"])
    result["aod"] = nanmean(aod)
    result["eod"] = nanmean(eod)
    result["r_sep"] = nanmean(r_sep)
    result["i_sep"] = nanmean(i_sep)
    result["c_sep_xfit"] = nanmean(c_sep_xfit)
    return result


def nanmean(values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
