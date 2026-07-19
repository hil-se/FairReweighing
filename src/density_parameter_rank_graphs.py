from argparse import ArgumentParser
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from data_reader import load_dataset
from density_balance import DensityBalance
from experiment import DEFAULT_TEST_SIZE, Experiment, nanmean_or_nan, safe_metric_name
from metrics import Metrics
from vgg_face_model import HEAD_BATCH_SIZE, LEARNING_RATE, build_regression_head


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
    "SCUT_Sex",
    "SCUT_Multi",
]
ESTIMATORS = ["Neighbor", "Kernel"]
PARAMETER_GRID = [round(value, 1) for value in np.arange(0.1, 1.01, 0.1)]
CLASSIFICATION_DATASETS = {"German", "Heart"}
SCUT_DATASETS = {"SCUT_Sex", "SCUT_Multi"}
TIMING_DATASETS = [
    "Synthetic",
    "LSAC",
    "LSAC_Multi",
    "Community",
    "Community_Con",
    "Community_Multi",
    "Insurance",
    "Insurance_Multi",
]
FIXED_DEFAULTS = {"Neighbor": 0.3, "Kernel": 0.2}
SCUT_OPTIONS = {"target": "P1"}
OUTPUT_DIR = Path("result")
SCUT_FEATURE_CACHE = OUTPUT_DIR / "cache" / "scut_vgg_face_features.npz"
PARAMETER_OUTPUT_DIR = OUTPUT_DIR / "parameter_sensitivity"
FIGURE_OUTPUT_DIR = OUTPUT_DIR / "figures"
BASELINE_RAW_DIR = OUTPUT_DIR / "raw"
FAIRICP_OUTPUTS = [
    OUTPUT_DIR / "baselines" / "fairicp" / "fairicp_probe.csv",
    OUTPUT_DIR / "baselines" / "fairicp" / "fairicp_probe_scut.csv",
]
REUSABLE_GRID_OUTPUT = PARAMETER_OUTPUT_DIR / "parameter_grid_sensitivity_raw.csv"
RAW_OUTPUT = PARAMETER_OUTPUT_DIR / "density_parameter_rank_raw.csv"
CHECKPOINT_OUTPUT = PARAMETER_OUTPUT_DIR / "density_parameter_rank_checkpoint.csv"
SUMMARY_OUTPUT = PARAMETER_OUTPUT_DIR / "density_parameter_rank_summary.csv"
PAIRWISE_OUTPUT = PARAMETER_OUTPUT_DIR / "density_parameter_rank_pairwise.csv"
RUNTIME_OUTPUT = PARAMETER_OUTPUT_DIR / "parameter_grid_fixed_vs_dynamic_cost.csv"
RADIUS_PDF = FIGURE_OUTPUT_DIR / "radius_rank_by_project.pdf"
RADIUS_PNG = FIGURE_OUTPUT_DIR / "radius_rank_by_project.png"
BANDWIDTH_PDF = FIGURE_OUTPUT_DIR / "bandwidth_rank_by_project.pdf"
BANDWIDTH_PNG = FIGURE_OUTPUT_DIR / "bandwidth_rank_by_project.png"
FIXED_METHOD = "fair-reweighing-fixed-parameter"
DYNAMIC_METHOD = "fair-reweighing-dynamic-parameter"
SCUT_DIAGNOSTIC_EPOCHS = 15
_SCUT_FEATURES = None
_SCUT_FEATURE_INDEX = None


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARAMETER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = args.datasets or DATASETS
    seeds_by_dataset = {
        dataset: list(range(args.first_seed, args.first_seed + repeats_for_dataset(dataset, args.repeats, args.scut_repeats)))
        for dataset in datasets
    }

    if args.rank_only:
        if not RAW_OUTPUT.exists():
            raise FileNotFoundError(f"Missing parameter-rank rows: {RAW_OUTPUT}")
        raw = deduplicate_parameter_rows(pd.read_csv(RAW_OUTPUT))
        raw = raw[raw["dataset"].isin(datasets)].copy()
    else:
        fixed_raw = run_fixed_grid(datasets, PARAMETER_GRID, seeds_by_dataset, resume=not args.no_resume)
        dynamic_raw = read_dynamic_rows(datasets, seeds_by_dataset)
        raw = pd.concat([fixed_raw, dynamic_raw], ignore_index=True, sort=False)
        raw = deduplicate_parameter_rows(raw)
        verify_counts(raw, datasets, PARAMETER_GRID, seeds_by_dataset)
        raw.to_csv(RAW_OUTPUT, index=False)

    summary, pairwise = rank_parameter_rows(raw, datasets)
    summary.to_csv(SUMMARY_OUTPUT, index=False)
    pairwise.to_csv(PAIRWISE_OUTPUT, index=False)
    runtime = fixed_dynamic_runtime(raw, datasets)
    runtime.to_csv(RUNTIME_OUTPUT, index=False)
    plot_rank_graph(summary, "Neighbor", RADIUS_PDF)
    plot_rank_graph(summary, "Neighbor", RADIUS_PNG)
    plot_rank_graph(summary, "Kernel", BANDWIDTH_PDF)
    plot_rank_graph(summary, "Kernel", BANDWIDTH_PNG)

    print(f"Wrote raw rows to {RAW_OUTPUT}")
    print(f"Wrote summary to {SUMMARY_OUTPUT}")
    print(f"Wrote pairwise tests to {PAIRWISE_OUTPUT}")
    print(f"Wrote fixed-vs-Dynamic timing comparison to {RUNTIME_OUTPUT}")
    print(f"Wrote graphs to {RADIUS_PDF}, {RADIUS_PNG}, {BANDWIDTH_PDF}, and {BANDWIDTH_PNG}")
    print(summary[["dataset", "estimator", "parameter_label", "mean_c_sep_xfit", "rank"]].to_string(index=False))


def fixed_dynamic_runtime(raw, datasets):
    rows = []
    included = [dataset for dataset in TIMING_DATASETS if dataset in datasets]
    for dataset in included:
        for estimator, fixed_parameter in FIXED_DEFAULTS.items():
            fixed = raw[
                raw["dataset"].eq(dataset)
                & raw["estimator"].eq(estimator)
                & raw["parameter_mode"].eq("fixed")
                & np.isclose(
                    pd.to_numeric(raw["fixed_parameter"], errors="coerce"),
                    fixed_parameter,
                )
            ]
            dynamic = raw[
                raw["dataset"].eq(dataset)
                & raw["estimator"].eq(estimator)
                & raw["parameter_mode"].eq("dynamic")
            ]
            if fixed.empty or dynamic.empty:
                continue
            row = {
                "dataset": dataset,
                "density_model": estimator,
                "fixed_parameter": fixed_parameter,
                "mean_dynamic_parameter": pd.to_numeric(
                    dynamic["parameter_value"], errors="coerce"
                ).mean(),
                "n": int(min(fixed["seed"].nunique(), dynamic["seed"].nunique())),
            }
            for metric in ("c_sep_xfit", "mse", "fit_seconds", "total_seconds"):
                fixed_mean = pd.to_numeric(fixed[metric], errors="coerce").mean()
                dynamic_mean = pd.to_numeric(dynamic[metric], errors="coerce").mean()
                row[f"fixed_{metric}"] = fixed_mean
                row[f"dynamic_{metric}"] = dynamic_mean
                if metric in {"fit_seconds", "total_seconds"}:
                    row[f"{metric}_change_pct"] = 100.0 * (
                        fixed_mean / dynamic_mean - 1.0
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    parser = ArgumentParser(description="Statistical rank graphs for fixed and dynamic FairReweighing density parameters.")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--scut-repeats", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--rank-only", action="store_true", help="Regenerate summaries and figures from saved raw rows.")
    return parser.parse_args()


def run_fixed_grid(datasets, parameters, seeds_by_dataset, resume=True):
    existing = read_existing_fixed_rows() if resume else pd.DataFrame()
    done = set()
    if not existing.empty:
        for _, row in existing.iterrows():
            done.add((
                str(row["dataset"]),
                str(row["estimator"]),
                round(float(row["fixed_parameter"]), 1),
                int(row["seed"]),
            ))

    rows = []
    total = sum(len(seeds_by_dataset[dataset]) for dataset in datasets) * len(ESTIMATORS) * len(parameters)
    completed = len(done)
    for dataset in datasets:
        for estimator in ESTIMATORS:
            for parameter in parameters:
                for seed in seeds_by_dataset[dataset]:
                    key = (dataset, estimator, round(float(parameter), 1), int(seed))
                    if key in done:
                        continue
                    rows.append(run_one(dataset, estimator, parameter, seed))
                    completed += 1
                    if completed % 20 == 0:
                        checkpoint = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False)
                        checkpoint = deduplicate_parameter_rows(checkpoint)
                        checkpoint.to_csv(CHECKPOINT_OUTPUT, index=False)
                        print(f"Completed {completed}/{total} fixed-parameter runs", flush=True)

    result = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False)
    result = deduplicate_parameter_rows(result)
    if not result.empty:
        result.to_csv(CHECKPOINT_OUTPUT, index=False)
    return result


def read_existing_fixed_rows():
    frames = []
    # Load reusable/older artifacts first so newer raw and checkpoint rows win
    # when the same dataset, estimator, parameter, and seed are present.
    if REUSABLE_GRID_OUTPUT.exists():
        reusable = pd.read_csv(REUSABLE_GRID_OUTPUT)
        reusable.columns = [str(column).strip() for column in reusable.columns]
        reusable = reusable[
            reusable["parameter_mode"].eq("fixed")
            & reusable["density_model"].isin(ESTIMATORS)
        ].copy()
        reusable["parameter_value"] = pd.to_numeric(reusable["parameter_value"], errors="coerce").round(1)
        reusable = reusable[reusable["parameter_value"].isin(PARAMETER_GRID)]
        if not reusable.empty:
            frames.append(normalize_fixed_rows(reusable))
    for path in (RAW_OUTPUT, CHECKPOINT_OUTPUT):
        if path.exists():
            raw = pd.read_csv(path)
            if "parameter_mode" in raw.columns:
                frames.append(raw[raw["parameter_mode"].eq("fixed")].copy())
    if not frames:
        return pd.DataFrame()
    return deduplicate_parameter_rows(pd.concat(frames, ignore_index=True, sort=False))


def run_one(dataset, estimator, parameter, seed):
    if is_scut_dataset(dataset):
        return run_one_scut_cached(dataset, estimator, parameter, seed)
    experiment = Experiment(
        data=dataset,
        regressor=model_for_dataset(dataset),
        balance="fair-reweighing",
        density_model=estimator,
        seed=seed,
        dataset_options=dataset_options_for_dataset(dataset),
    )
    experiment.select_density_param = lambda parameter=parameter: parameter
    row = experiment.run()
    return normalize_fixed_rows(pd.DataFrame([row]), estimator=estimator, parameter=parameter).iloc[0].to_dict()


def run_one_scut_cached(dataset, estimator, parameter, seed):
    started = time.perf_counter()
    X, y, protected = load_dataset(dataset, seed=seed, **SCUT_OPTIONS)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=DEFAULT_TEST_SIZE,
        random_state=seed,
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    train_features = scut_features_for_paths(X_train["image_path"])
    test_features = scut_features_for_paths(X_test["image_path"])

    fit_started = time.perf_counter()
    balance = DensityBalance(
        model=estimator,
        radius=parameter if estimator == "Neighbor" else 0.5,
        bandwidth=parameter if estimator == "Kernel" else 0.2,
    )
    weights = balance.weight(
        X_train[protected].to_numpy(),
        np.transpose([y_train]),
        treatment="fair-reweighing",
    )
    y_pred = fit_cached_scut_head(train_features, y_train, test_features, weights, seed)
    fit_seconds = time.perf_counter() - fit_started

    result = evaluate_predictions(y_test, y_pred, X_test, protected, seed)
    result.update({
        "dataset": dataset,
        "model": "vgg_face_cached_linear",
        "method": "fair-reweighing",
        "density_model": estimator,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "selected_radius": parameter if estimator == "Neighbor" else np.nan,
        "selected_bandwidth": parameter if estimator == "Kernel" else np.nan,
    })
    return normalize_fixed_rows(pd.DataFrame([result]), estimator=estimator, parameter=parameter).iloc[0].to_dict()


def fit_cached_scut_head(train_features, y_train, test_features, weights, seed):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    train_index, validation_index = train_test_split(
        np.arange(len(train_features)),
        test_size=0.2,
        random_state=seed,
    )
    model = build_regression_head(train_features.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        steps_per_execution=20,
    )
    model.fit(
        train_features[train_index],
        y_train[train_index],
        sample_weight=weights[train_index],
        validation_data=(train_features[validation_index], y_train[validation_index], weights[validation_index]),
        batch_size=HEAD_BATCH_SIZE,
        epochs=SCUT_DIAGNOSTIC_EPOCHS,
        verbose=0,
    )
    return model.predict(test_features, batch_size=HEAD_BATCH_SIZE, verbose=0).reshape(-1)


def evaluate_predictions(y_true, y_pred, X_test, protected, seed):
    metrics = Metrics(y_true, y_pred)
    r_sep = []
    i_sep = []
    c_sep_xfit = []
    aod = []
    eod = []
    result = {}
    for protected_column in protected:
        protected_name = safe_metric_name(protected_column)
        sensitive = X_test[protected_column].to_numpy()
        categorical_sep = metrics.categorical_separation(sensitive, seed=seed)
        equalized_odds = metrics.equalized_odds_metrics(sensitive)
        protected_r_sep = categorical_sep["r_sep"]
        protected_i_sep = categorical_sep["i_sep"]
        protected_c_sep_xfit = metrics.continuous_mi_crossfit(sensitive, seed=seed)
        protected_aod = equalized_odds["aod"]
        protected_eod = equalized_odds["eod"]
        r_sep.append(protected_r_sep)
        i_sep.append(protected_i_sep)
        c_sep_xfit.append(protected_c_sep_xfit)
        aod.append(protected_aod)
        eod.append(protected_eod)
        result.update({
            f"r_sep_{protected_name}": protected_r_sep,
            f"i_sep_{protected_name}": protected_i_sep,
            f"c_sep_xfit_{protected_name}": protected_c_sep_xfit,
            f"aod_{protected_name}": protected_aod,
            f"eod_{protected_name}": protected_eod,
        })
    result.update({
        "mse": metrics.mse(),
        "mae": metrics.mae(),
        "r2": metrics.r2(),
        "accuracy": metrics.accuracy(),
        "f1": metrics.f1(),
        "aod": nanmean_or_nan(aod),
        "eod": nanmean_or_nan(eod),
        "r_sep": nanmean_or_nan(r_sep),
        "i_sep": nanmean_or_nan(i_sep),
        "c_sep_xfit": nanmean_or_nan(c_sep_xfit),
    })
    return result


def scut_features_for_paths(paths):
    global _SCUT_FEATURES, _SCUT_FEATURE_INDEX
    if _SCUT_FEATURES is None or _SCUT_FEATURE_INDEX is None:
        if not SCUT_FEATURE_CACHE.exists():
            raise FileNotFoundError(f"Missing SCUT feature cache: {SCUT_FEATURE_CACHE}")
        cache = np.load(SCUT_FEATURE_CACHE, allow_pickle=True)
        _SCUT_FEATURES = cache["features"]
        _SCUT_FEATURE_INDEX = {str(path): index for index, path in enumerate(cache["paths"].astype(str))}
    indices = []
    for path in paths.astype(str):
        if path not in _SCUT_FEATURE_INDEX:
            raise KeyError(f"SCUT feature cache is missing image path: {path}")
        indices.append(_SCUT_FEATURE_INDEX[path])
    return _SCUT_FEATURES[np.asarray(indices, dtype=int)]


def normalize_fixed_rows(rows, estimator=None, parameter=None):
    rows = rows.copy()
    if estimator is None:
        rows["estimator"] = rows["density_model"].astype(str).str.strip()
    else:
        rows["estimator"] = estimator
    if parameter is None:
        rows["fixed_parameter"] = pd.to_numeric(rows["parameter_value"], errors="coerce").round(1)
    else:
        rows["fixed_parameter"] = round(float(parameter), 1)
    rows["parameter_mode"] = "fixed"
    rows["parameter_value"] = rows["fixed_parameter"]
    rows["parameter_label"] = rows["fixed_parameter"].map(lambda value: f"{float(value):.1f}")
    rows["plot_order"] = rows["fixed_parameter"]
    rows["method"] = FIXED_METHOD
    rows["density_model"] = [
        treatment_density(estimator_value, parameter_value)
        for estimator_value, parameter_value in zip(rows["estimator"], rows["fixed_parameter"])
    ]
    if "selected_radius" in rows.columns:
        rows["selected_radius"] = pd.to_numeric(rows["selected_radius"], errors="coerce")
    else:
        rows["selected_radius"] = np.nan
    if "selected_bandwidth" in rows.columns:
        rows["selected_bandwidth"] = pd.to_numeric(rows["selected_bandwidth"], errors="coerce")
    else:
        rows["selected_bandwidth"] = np.nan
    rows.loc[rows["estimator"].eq("Neighbor"), "selected_radius"] = rows.loc[rows["estimator"].eq("Neighbor"), "fixed_parameter"]
    rows.loc[rows["estimator"].eq("Kernel"), "selected_bandwidth"] = rows.loc[rows["estimator"].eq("Kernel"), "fixed_parameter"]
    return rows


def read_dynamic_rows(datasets, seeds_by_dataset):
    rows = []
    for dataset in datasets:
        if is_scut_dataset(dataset):
            path = BASELINE_RAW_DIR / f"{dataset}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing raw result file: {path}")
            main_raw = read_result_csv(path)
            for estimator in ESTIMATORS:
                parameter_column = "selected_radius" if estimator == "Neighbor" else "selected_bandwidth"
                selected = main_raw[
                    main_raw["method"].eq("fair-reweighing")
                    & main_raw["density_model"].eq(estimator)
                ][["seed", parameter_column]].copy()
                selected["seed"] = pd.to_numeric(selected["seed"], errors="coerce").astype("Int64")
                selected[parameter_column] = pd.to_numeric(selected[parameter_column], errors="coerce")
                parameter_by_seed = selected.dropna().set_index("seed")[parameter_column].to_dict()
                missing = [seed for seed in seeds_by_dataset[dataset] if seed not in parameter_by_seed]
                if missing:
                    raise RuntimeError(
                        f"Missing Dynamic {estimator} parameters for {dataset} seeds: {missing}"
                    )
                dataset_rows = []
                for seed in seeds_by_dataset[dataset]:
                    parameter = float(parameter_by_seed[seed])
                    row = run_one_scut_cached(dataset, estimator, parameter, seed)
                    row["parameter_value"] = parameter
                    dataset_rows.append(row)
                dynamic = normalize_dynamic_rows(pd.DataFrame(dataset_rows), estimator=estimator)
                rows.append(dynamic)
            continue
        path = BASELINE_RAW_DIR / f"{dataset}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw result file: {path}")
        raw = read_result_csv(path)
        fair = raw[raw["method"].eq("fair-reweighing") & raw["density_model"].isin(ESTIMATORS)].copy()
        if fair.empty:
            continue
        fair["estimator"] = fair["density_model"]
        fair["parameter_mode"] = "dynamic"
        fair["fixed_parameter"] = np.nan
        fair["parameter_value"] = np.where(
            fair["estimator"].eq("Neighbor"),
            pd.to_numeric(fair["selected_radius"], errors="coerce"),
            pd.to_numeric(fair["selected_bandwidth"], errors="coerce"),
        )
        fair["parameter_label"] = "Dynamic"
        fair["plot_order"] = len(PARAMETER_GRID) + 1
        fair["method"] = DYNAMIC_METHOD
        fair["density_model"] = fair["estimator"].map(lambda estimator: f"{estimator}-Dynamic")
        rows.append(fair)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def normalize_dynamic_rows(rows, estimator, parameter=None):
    rows = rows.copy()
    rows["estimator"] = estimator
    rows["parameter_mode"] = "dynamic"
    rows["fixed_parameter"] = np.nan
    if parameter is not None:
        rows["parameter_value"] = parameter
    elif "parameter_value" not in rows.columns:
        parameter_column = "selected_radius" if estimator == "Neighbor" else "selected_bandwidth"
        rows["parameter_value"] = pd.to_numeric(rows[parameter_column], errors="coerce")
    rows["parameter_label"] = "Dynamic"
    rows["plot_order"] = len(PARAMETER_GRID) + 1
    rows["method"] = DYNAMIC_METHOD
    rows["density_model"] = f"{estimator}-Dynamic"
    if parameter is not None:
        rows.loc[rows["estimator"].eq("Neighbor"), "selected_radius"] = parameter
        rows.loc[rows["estimator"].eq("Kernel"), "selected_bandwidth"] = parameter
    return rows


def verify_counts(raw, datasets, parameters, seeds_by_dataset):
    problems = []
    for dataset in datasets:
        expected = len(seeds_by_dataset[dataset])
        for estimator in ESTIMATORS:
            for parameter in parameters:
                count = raw[
                    raw["dataset"].eq(dataset)
                    & raw["estimator"].eq(estimator)
                    & raw["parameter_mode"].eq("fixed")
                    & np.isclose(raw["fixed_parameter"].astype(float), float(parameter))
                ]["seed"].nunique()
                if int(count) != expected:
                    problems.append(f"{dataset} {estimator} {parameter:.1f}: {count}/{expected}")
            dynamic_count = raw[
                raw["dataset"].eq(dataset)
                & raw["estimator"].eq(estimator)
                & raw["parameter_mode"].eq("dynamic")
            ]["seed"].nunique()
            if int(dynamic_count) != expected:
                problems.append(f"{dataset} {estimator} Dynamic: {dynamic_count}/{expected}")
    if problems:
        raise RuntimeError("Incomplete parameter-rank rows: " + "; ".join(problems))


def read_baseline_rows(datasets):
    frames = []
    for dataset in datasets:
        path = BASELINE_RAW_DIR / f"{dataset}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline raw file: {path}")
        raw = read_result_csv(path)
        frames.append(raw[~raw["method"].eq("fair-reweighing")].copy())
    fairicp = read_fairicp_rows(datasets)
    if not fairicp.empty:
        frames.append(fairicp)
    return pd.concat(frames, ignore_index=True, sort=False)


def read_fairicp_rows(datasets):
    frames = []
    for path in FAIRICP_OUTPUTS:
        if not path.exists():
            continue
        fairicp = read_result_csv(path)
        if "method" in fairicp.columns:
            frames.append(fairicp)
    if not frames:
        return pd.DataFrame()
    fairicp = pd.concat(frames, ignore_index=True, sort=False)
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


def repeats_for_dataset(dataset, repeat_count, scut_repeat_count):
    return scut_repeat_count if is_scut_dataset(dataset) else repeat_count


def model_for_dataset(dataset):
    if is_scut_dataset(dataset):
        return "vgg_face"
    if dataset in CLASSIFICATION_DATASETS:
        return "logistic"
    return "linear"


def dataset_options_for_dataset(dataset):
    return SCUT_OPTIONS if is_scut_dataset(dataset) else None


def is_scut_dataset(dataset):
    return dataset in SCUT_DATASETS


def rank_parameter_rows(raw, datasets):
    summary_rows = []
    pairwise_rows = []
    for dataset in datasets:
        for estimator in ESTIMATORS:
            candidates = raw[raw["dataset"].eq(dataset) & raw["estimator"].eq(estimator)].copy()
            treatments = treatment_series(candidates)
            ranks = recursive_ranks(treatments)
            reference_key = rank_zero_reference(treatments, ranks)
            for key, treatment in candidate_treatments(candidates):
                paired = paired_by_seed(treatments[key], treatments[reference_key])
                candidate_values = paired["row"].to_numpy(dtype=float)
                reference_values = paired["column"].to_numpy(dtype=float)
                test = paired_wilcoxon(candidate_values, reference_values)
                delta = cliffs_delta(candidate_values, reference_values)
                metric_summary = summarize_metrics(treatment)
                rank = int(ranks[key])
                summary_rows.append({
                    "dataset": dataset,
                    "display_dataset": display_dataset(dataset),
                    "estimator": estimator,
                    "parameter_mode": treatment["parameter_mode"].iloc[0],
                    "parameter_label": treatment["parameter_label"].iloc[0],
                    "plot_order": float(treatment["plot_order"].iloc[0]),
                    "mean_selected_parameter": float(pd.to_numeric(treatment["parameter_value"], errors="coerce").mean()),
                    "n": int(treatment["seed"].nunique()),
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
                    "estimator": estimator,
                    "parameter_label": treatment["parameter_label"].iloc[0],
                    "candidate_rank": rank,
                    "candidate_top_rank": rank == 0,
                    "candidate_mean_c_sep_xfit": float(treatments[key]["c_sep_xfit"].mean()),
                    "reference_method": reference_key[0],
                    "reference_density": reference_key[1],
                    "reference_mean_c_sep_xfit": float(treatments[reference_key]["c_sep_xfit"].mean()),
                    "n": int(len(paired)),
                    "wilcoxon_stat": test["statistic"],
                    "p_value": test["p_value"],
                    "cliffs_delta": delta,
                    "conclusion": "rank 0" if rank == 0 else f"rank {rank}",
                })
    summary = pd.DataFrame(summary_rows).sort_values(["estimator", "dataset", "plot_order"])
    pairwise = pd.DataFrame(pairwise_rows).sort_values(["estimator", "dataset", "parameter_label"])
    return summary, pairwise


def candidate_treatments(candidates):
    for key, treatment in candidates.groupby(["method", "density_model"], dropna=False):
        yield key, treatment


def summarize_metrics(treatment):
    row = {}
    for metric in ("mse", "r2", "c_sep_xfit"):
        values = pd.to_numeric(treatment[metric], errors="coerce").dropna()
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


def plot_rank_graph(summary, estimator, output_path):
    subset = summary[summary["estimator"].eq(estimator)].copy()
    x_labels = [f"{value:.1f}" for value in PARAMETER_GRID] + ["Dynamic"]
    subset["plot_label"] = subset["parameter_label"].map(normalize_plot_label)
    heatmap = (
        subset.pivot_table(index="dataset", columns="plot_label", values="rank", aggfunc="first")
        .reindex(index=DATASETS, columns=x_labels)
        .T
    )
    matrix = heatmap.to_numpy(dtype=float)
    max_rank = int(np.nanmax(matrix)) if np.isfinite(matrix).any() else 1

    fig, axis = plt.subplots(figsize=(10.5, 6.8))
    image = axis.imshow(matrix, cmap="RdYlGn_r", vmin=-0.5, vmax=max_rank + 0.5, aspect="auto")

    axis.set_xticks(np.arange(len(DATASETS)))
    axis.set_xticklabels([display_dataset(dataset) for dataset in DATASETS], rotation=35, ha="right")
    axis.set_yticks(np.arange(len(x_labels)))
    axis.set_yticklabels(x_labels)
    axis.set_xlabel("Project")
    axis.set_ylabel("Radius" if estimator == "Neighbor" else "Bandwidth")

    for row_index, label in enumerate(x_labels):
        for col_index, dataset in enumerate(DATASETS):
            value = heatmap.loc[label, dataset]
            if pd.notna(value):
                axis.text(
                    col_index,
                    row_index,
                    f"R{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    axis.set_xticks(np.arange(-0.5, len(DATASETS), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    fig.suptitle(
        f"{'Radius' if estimator == 'Neighbor' else 'Bandwidth'} Statistical Rank Heatmap",
        y=0.98,
        fontsize=13,
    )
    colorbar = fig.colorbar(image, ax=axis, ticks=np.arange(0, max_rank + 1), pad=0.015)
    colorbar.set_label("Recursive rank (0 is best)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def deduplicate_parameter_rows(raw):
    if raw.empty:
        return raw
    raw = raw.copy()
    raw["seed"] = pd.to_numeric(raw["seed"], errors="coerce").astype("Int64")
    raw["fixed_parameter"] = pd.to_numeric(raw["fixed_parameter"], errors="coerce").round(1)
    raw["parameter_mode"] = raw["parameter_mode"].astype(str).str.strip()
    fixed_mask = raw["parameter_mode"].eq("fixed")
    dynamic_mask = raw["parameter_mode"].eq("dynamic")
    raw.loc[fixed_mask, "parameter_label"] = raw.loc[fixed_mask, "fixed_parameter"].map(lambda value: f"{float(value):.1f}")
    raw.loc[dynamic_mask, "parameter_label"] = "Dynamic"
    return raw.drop_duplicates(
        subset=["dataset", "estimator", "parameter_mode", "parameter_label", "seed"],
        keep="last",
    )


def treatment_density(estimator, parameter):
    return f"{estimator}-{float(parameter):.1f}"


def treatment_display_name(method, density):
    method_key = str(method).strip().lower()
    density_key = str(density).strip()
    names = {
        "none": "None",
        "fair-reweighing-fixed-parameter": f"FR-{density_key}",
        "fair-reweighing-dynamic-parameter": f"FR-{density_key}",
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
        "SCUT_Sex": "SCUT-Sex",
        "SCUT_Multi": "SCUT-Multi",
    }
    return names.get(dataset, dataset.replace("_", "-"))


def normalize_plot_label(value):
    if str(value) == "Dynamic":
        return "Dynamic"
    return f"{float(value):.1f}"


if __name__ == "__main__":
    main()
