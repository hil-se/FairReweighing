"""Extend the main SCUT experiments using the cached VGG-Face features."""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_reader import load_dataset
from density_balance import DensityBalance
from density_parameter_rank_graphs import evaluate_predictions, scut_features_for_paths
from experiment import (
    DEFAULT_BANDWIDTH_GRID,
    DEFAULT_RADIUS_GRID,
    DEFAULT_TEST_SIZE,
    quantile_repair,
    sensitive_groups,
)
from main import average_trials, has_multiple_sensitive_attributes, ordered_result_columns
from metrics import Metrics
from vgg_face_model import (
    EPOCHS,
    HEAD_BATCH_SIZE,
    LEARNING_RATE,
    PATIENCE,
    VALIDATION_SPLIT,
    build_regression_head,
)


DATASETS = ("SCUT_Sex", "SCUT_Multi")
METHODS = (
    ("none", "none"),
    ("fair-reweighing", "Neighbor"),
    ("fair-reweighing", "Kernel"),
    ("chzhen-repair", "none"),
)
SEEDS = range(1, 21)
RAW_DIR = Path("result/raw")
MAIN_DIR = Path("result/main")
SCUT_OPTIONS = {"target": "P1"}


def main():
    for dataset in DATASETS:
        path = RAW_DIR / f"{dataset}.csv"
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        done = treatment_keys(existing)
        rows = []
        for seed in SEEDS:
            missing = [treatment for treatment in METHODS if (*treatment, seed) not in done]
            if not missing:
                continue
            rows.extend(run_seed(dataset, seed, missing))
            updated = merge_rows(existing, rows)
            updated.to_csv(path, index=False)
            print(f"{dataset}: completed seed {seed}; {len(updated)} total rows", flush=True)

        raw = merge_rows(existing, rows)
        raw.to_csv(path, index=False)
        averaged = average_trials(raw)
        averaged = averaged[ordered_result_columns(
            averaged,
            include_per_sensitive=has_multiple_sensitive_attributes(averaged),
        )]
        averaged.to_csv(MAIN_DIR / f"{dataset}.csv", index=False)
        verify_complete(raw, dataset)


def run_seed(dataset, seed, missing):
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
    rows = []

    unweighted = [item for item in missing if item[0] in {"none", "chzhen-repair"}]
    if unweighted:
        started = time.perf_counter()
        train_pred, test_pred, fit_seconds = fit_head(
            train_features,
            y_train,
            test_features,
            None,
            seed,
        )
        total_seconds = time.perf_counter() - started
        for method, density_model in unweighted:
            predictions = test_pred
            if method == "chzhen-repair":
                train_groups, test_groups = sensitive_groups(
                    X_train[protected],
                    X_test[protected],
                )
                predictions = quantile_repair(train_pred, test_pred, train_groups, test_groups)
            rows.append(result_row(
                dataset,
                method,
                density_model,
                seed,
                y_test,
                predictions,
                X_test,
                protected,
                fit_seconds,
                total_seconds,
            ))

    for method, density_model in missing:
        if method != "fair-reweighing":
            continue
        started = time.perf_counter()
        parameter = select_density_parameter(
            density_model,
            X_train,
            y_train,
            protected,
            train_features,
            seed,
        )
        balance = DensityBalance(
            model=density_model,
            radius=parameter if density_model == "Neighbor" else 0.5,
            bandwidth=parameter if density_model == "Kernel" else 0.2,
        )
        weights = balance.weight(
            X_train[protected].to_numpy(),
            np.transpose([y_train]),
            treatment="fair-reweighing",
        )
        _, predictions, fit_seconds = fit_head(
            train_features,
            y_train,
            test_features,
            weights,
            seed,
        )
        rows.append(result_row(
            dataset,
            method,
            density_model,
            seed,
            y_test,
            predictions,
            X_test,
            protected,
            fit_seconds,
            time.perf_counter() - started,
            parameter,
        ))
    return rows


def select_density_parameter(density_model, X_train, y_train, protected, train_features, seed):
    grid = DEFAULT_RADIUS_GRID if density_model == "Neighbor" else DEFAULT_BANDWIDTH_GRID
    fit_index, validation_index = train_test_split(
        np.arange(len(y_train)),
        test_size=0.25,
        random_state=seed + 17,
    )
    y_train = np.asarray(y_train)
    best_parameter = grid[0]
    best_score = float("inf")

    for parameter in grid:
        balance = DensityBalance(
            model=density_model,
            radius=parameter if density_model == "Neighbor" else 0.5,
            bandwidth=parameter if density_model == "Kernel" else 0.2,
        )
        try:
            weights = balance.weight(
                X_train.iloc[fit_index][protected].to_numpy(),
                np.transpose([y_train[fit_index]]),
                treatment="fair-reweighing",
            )
            _, predictions, _ = fit_head(
                train_features[fit_index],
                y_train[fit_index],
                train_features[validation_index],
                weights,
                seed,
            )
            metrics = Metrics(y_train[validation_index], predictions)
            separation = np.asarray([
                metrics.continuous_mi_crossfit(
                    X_train.iloc[validation_index][column].to_numpy(),
                    seed=seed,
                )
                for column in protected
            ], dtype=float)
            if not np.isfinite(separation).any():
                continue
            score = float(np.nanmean(separation)) + 0.01 * metrics.mse()
        except Exception as exc:
            print(
                f"SCUT parameter selection skipped {density_model}={parameter}: {exc}",
                flush=True,
            )
            continue
        if score < best_score:
            best_score = score
            best_parameter = parameter
    return best_parameter


def fit_head(train_features, y_train, test_features, weights, seed):
    import tensorflow as tf

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    train_index, validation_index = train_test_split(
        np.arange(len(train_features)),
        test_size=VALIDATION_SPLIT,
        random_state=seed,
    )
    model = build_regression_head(train_features.shape[1])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="mse",
        steps_per_execution=20,
    )
    train_weights = None if weights is None else weights[train_index]
    validation_data = (
        train_features[validation_index],
        y_train[validation_index],
    )
    if weights is not None:
        validation_data = (*validation_data, weights[validation_index])
    started = time.perf_counter()
    model.fit(
        train_features[train_index],
        y_train[train_index],
        sample_weight=train_weights,
        validation_data=validation_data,
        batch_size=HEAD_BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            min_delta=1e-4,
            restore_best_weights=True,
        )],
    )
    fit_seconds = time.perf_counter() - started
    train_pred = model.predict(train_features, batch_size=HEAD_BATCH_SIZE, verbose=0).reshape(-1)
    test_pred = model.predict(test_features, batch_size=HEAD_BATCH_SIZE, verbose=0).reshape(-1)
    return train_pred, test_pred, fit_seconds


def result_row(
    dataset,
    method,
    density_model,
    seed,
    y_test,
    predictions,
    X_test,
    protected,
    fit_seconds,
    total_seconds,
    parameter=np.nan,
):
    row = evaluate_predictions(y_test, predictions, X_test, protected, seed)
    row.update({
        "dataset": dataset,
        "model": "vgg_face",
        "method": method,
        "density_model": density_model,
        "seed": seed,
        "fit_seconds": fit_seconds,
        "total_seconds": total_seconds,
        "selected_radius": parameter if density_model == "Neighbor" else np.nan,
        "selected_bandwidth": parameter if density_model == "Kernel" else np.nan,
    })
    return row


def treatment_keys(rows):
    if rows.empty:
        return set()
    return {
        (str(row.method), str(row.density_model), int(row.seed))
        for row in rows.itertuples()
    }


def merge_rows(existing, new_rows):
    frames = [frame for frame in (existing, pd.DataFrame(new_rows)) if not frame.empty]
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    return result.drop_duplicates(
        subset=["dataset", "method", "density_model", "seed"],
        keep="last",
    ).sort_values(["seed", "method", "density_model"]).reset_index(drop=True)


def verify_complete(rows, dataset):
    counts = rows.groupby(["method", "density_model"])["seed"].nunique()
    missing = [
        f"{method}/{density}: {int(counts.get((method, density), 0))}/20"
        for method, density in METHODS
        if int(counts.get((method, density), 0)) != 20
    ]
    if missing:
        raise RuntimeError(f"Incomplete {dataset} rows: {'; '.join(missing)}")


if __name__ == "__main__":
    main()
