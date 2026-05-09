import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import has_fit_parameter

from data_reader import load_dataset
from density_balance import DensityBalance
from metrics import Metrics
from vgg_face_model import ScutVGGFaceRegressor


DEFAULT_RADIUS_GRID = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
DEFAULT_BANDWIDTH_GRID = [0.05, 0.1, 0.2, 0.5, 1.0]
DEFAULT_TEST_SIZE = 0.5
DEFAULT_BINS = 5
BERK_PAIRWISE_LAMBDA = 1.0
BERK_PAIRWISE_NEIGHBORS = 5
BERK_RIDGE = 1e-6

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class Experiment:
    def __init__(
        self,
        data="Community",
        regressor="auto",
        balance="none",
        density_model="Neighbor",
        seed=0,
        dataset_options=None,
    ):
        self.data = data
        self.seed = seed
        self.balance = normalize_method(balance)
        self.density_model = normalize_density(density_model)
        self.test_size = DEFAULT_TEST_SIZE
        self.n_bins = DEFAULT_BINS
        self.dataset_options = dataset_options or {}
        self.X, self.y, self.protected = load_dataset(data, seed=seed, **self.dataset_options)
        self.task_type = "classification" if len(np.unique(self.y)) == 2 else "regression"
        if self.balance == "fair-reweighing" and self.task_type == "classification":
            self.density_model = "Reweighing"
        self.regressor_name = self.default_model(regressor)
        self.regressor = build_estimator(self.regressor_name, self.task_type, seed)
        if has_image_features(self.X) and not is_raw_image_model(self.regressor):
            raise ValueError("SCUT image data uses model=vgg_face.")
        if is_raw_image_model(self.regressor) and not has_image_features(self.X):
            raise ValueError("The vgg_face model requires SCUT image_path features.")
        self.preprocessor = None
        self.selected_density_param = None

    def run(self):
        started = time.perf_counter()
        self.train_test_split()
        if is_raw_image_model(self.regressor):
            X_train = self.X_train[["image_path"]]
            X_test = self.X_test[["image_path"]]
        else:
            self.preprocess(self.X_train)
            X_train = self.preprocessor.transform(self.X_train)
            X_test = self.preprocessor.transform(self.X_test)

        fit_started = time.perf_counter()
        if self.balance == "berk-pairwise":
            y_pred_for_metrics = self.berk_pairwise_predict(X_train, X_test)
        else:
            estimator = clone(self.regressor)
            y_fit = self.training_target()
            sample_weight = self.make_sample_weight_for_estimator(X_train, y_fit)
            fit_estimator(estimator, X_train, y_fit, sample_weight)

            y_pred = estimator.predict(X_test)
            if self.task_type == "classification" and hasattr(estimator, "predict_proba"):
                y_pred_for_metrics = estimator.predict_proba(X_test)[:, 1]
            else:
                y_pred_for_metrics = y_pred
            if self.balance == "chzhen-repair":
                train_pred = estimator.predict(X_train)
                y_pred_for_metrics = self.chzhen_repair(train_pred, y_pred_for_metrics)
        fit_seconds = time.perf_counter() - fit_started

        result = self.evaluate(y_pred_for_metrics)
        result.update({
            "dataset": self.data,
            "model": self.regressor_name,
            "method": self.balance,
            "density_model": self.density_model if self.balance == "fair-reweighing" else "none",
            "seed": self.seed,
            "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - started,
            "selected_radius": self.selected_density_param if self.density_model == "Neighbor" else np.nan,
            "selected_bandwidth": self.selected_density_param if self.density_model == "Kernel" else np.nan,
        })
        return result

    def train_test_split(self):
        stratify = self.y if self.task_type == "classification" else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=stratify,
        )
        self.X_train = self.X_train.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)

    def default_model(self, regressor):
        if regressor not in (None, "auto"):
            return regressor
        if has_image_features(self.X):
            return "vgg_face"
        return default_model_for_task(self.task_type)

    def preprocess(self, X):
        numerical_columns = selector(dtype_exclude=object)(X)
        categorical_columns = selector(dtype_include=object)(X)
        self.preprocessor = ColumnTransformer([
            ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns),
            ("StandardScaler", StandardScaler(), numerical_columns),
        ], remainder="drop")
        self.preprocessor.fit(X)

    def training_target(self):
        if self.balance == "calders-effect-control":
            return self.calders_adjusted_target()
        return self.y_train

    def make_sample_weight_for_estimator(self, X_train, y_fit):
        if self.balance == "agarwal-bgl":
            self.selected_density_param = np.nan
            return self.agarwal_bgl_weight(X_train, y_fit)
        if self.balance in {"calders-effect-control", "chzhen-repair"}:
            self.selected_density_param = np.nan
            return None
        return self.make_sample_weight()

    def make_sample_weight(self):
        if self.balance == "none":
            self.selected_density_param = np.nan
            return None

        if self.balance == "fair-reweighing" and self.density_model in {"Neighbor", "Kernel"}:
            self.selected_density_param = self.select_density_param()
        elif self.density_model == "Neighbor":
            self.selected_density_param = 0.5
        elif self.density_model == "Kernel":
            self.selected_density_param = 0.2
        else:
            self.selected_density_param = np.nan

        balance = DensityBalance(
            model=self.density_model,
            radius=self.selected_density_param if self.density_model == "Neighbor" else 0.5,
            bandwidth=self.selected_density_param if self.density_model == "Kernel" else 0.2,
            n_bins=self.n_bins,
        )
        weights = balance.weight(
            self.X_train[self.protected].to_numpy(),
            np.transpose([self.y_train]),
            treatment=self.balance,
        )
        return weights

    def calders_adjusted_target(self):
        protected = numeric_sensitive_frame(self.X_train[self.protected])
        effect_model = LinearRegression().fit(protected, self.y_train)
        effect = effect_model.predict(protected)
        return self.y_train - (effect - np.mean(effect))

    def chzhen_repair(self, train_pred, test_pred):
        train_groups, test_groups = sensitive_groups(
            self.X_train[self.protected],
            self.X_test[self.protected],
            n_bins=self.n_bins,
        )
        return quantile_repair(train_pred, test_pred, train_groups, test_groups)

    def agarwal_bgl_weight(self, X_train, y_fit):
        base = clone(self.regressor)
        fit_estimator(base, X_train, y_fit)
        train_pred = base.predict(X_train)
        groups, _ = sensitive_groups(self.X_train[self.protected], n_bins=self.n_bins)
        errors = (np.asarray(y_fit) - np.asarray(train_pred)) ** 2
        global_loss = max(float(np.mean(errors)), 1e-12)
        weights = np.ones(len(errors), dtype=float)
        for group in pd.Series(groups).dropna().unique():
            mask = groups == group
            if mask.any():
                weights[mask] = np.mean(errors[mask]) / global_loss
        return normalize_weights(np.clip(weights, 0.25, 4.0))

    def berk_pairwise_predict(self, X_train, X_test):
        self.selected_density_param = np.nan
        groups, _ = sensitive_groups(self.X_train[self.protected], n_bins=self.n_bins)
        similarity = similarity_matrix_without_sensitive(self.X_train, self.protected)
        coef = fit_berk_pairwise_linear(
            np.asarray(X_train, dtype=float),
            np.asarray(self.y_train, dtype=float),
            groups,
            similarity,
        )
        X_test_aug = add_intercept(np.asarray(X_test, dtype=float))
        return X_test_aug @ coef

    def select_density_param(self):
        if is_raw_image_model(self.regressor):
            return 0.5 if self.density_model == "Neighbor" else 0.2
        if self.regressor_name in {"rf", "mlp"}:
            return 0.5 if self.density_model == "Neighbor" else 0.2

        grid = DEFAULT_RADIUS_GRID if self.density_model == "Neighbor" else DEFAULT_BANDWIDTH_GRID
        X_fit, X_val, y_fit, y_val = train_test_split(
            self.X_train,
            self.y_train,
            test_size=0.25,
            random_state=self.seed + 17,
            stratify=self.y_train if self.task_type == "classification" else None,
        )
        X_fit = X_fit.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        preprocessor = clone(self.preprocessor).fit(X_fit)
        X_fit_t = preprocessor.transform(X_fit)
        X_val_t = preprocessor.transform(X_val)

        best_param = grid[0]
        best_score = float("inf")
        for param in grid:
            balance = DensityBalance(
                model=self.density_model,
                radius=param if self.density_model == "Neighbor" else 0.5,
                bandwidth=param if self.density_model == "Kernel" else 0.2,
                n_bins=self.n_bins,
            )
            try:
                weights = balance.weight(X_fit[self.protected].to_numpy(), np.transpose([y_fit]), treatment=self.balance)
                estimator = clone(self.regressor)
                fit_estimator(estimator, X_fit_t, y_fit, weights)
                pred = estimator.predict(X_val_t)
                if self.task_type == "classification" and hasattr(estimator, "predict_proba"):
                    pred = estimator.predict_proba(X_val_t)[:, 1]
                metrics = Metrics(y_val, pred)
                fairness = [
                    metrics.continuous_mi_crossfit(X_val[protected].to_numpy(), seed=self.seed)
                    for protected in self.protected
                ]
                fairness_score = nanmean_or_nan(fairness)
                if not np.isfinite(fairness_score):
                    continue
                score = fairness_score + 0.01 * metrics.mse()
            except Exception:
                continue
            if score < best_score:
                best_score = score
                best_param = param
        return best_param

    def evaluate(self, y_pred):
        metrics = Metrics(self.y_test, y_pred)
        r_sep = []
        i_sep = []
        c_sep_xfit = []
        extra_metrics = {
            "pred_pearson_abs": [],
            "pred_spearman_abs": [],
            "pred_mean_gap": [],
            "residual_mean_gap": [],
            "group_mse_gap": [],
            "wasserstein_pred_gap": [],
        }
        result = {}
        for protected in self.protected:
            protected_name = safe_metric_name(protected)
            sensitive = self.X_test[protected].to_numpy()
            categorical_sep = metrics.categorical_separation(sensitive, seed=self.seed)
            protected_r_sep = categorical_sep["r_sep"]
            protected_i_sep = categorical_sep["i_sep"]
            protected_c_sep_xfit = metrics.continuous_mi_crossfit(sensitive, seed=self.seed)
            r_sep.append(protected_r_sep)
            i_sep.append(protected_i_sep)
            c_sep_xfit.append(protected_c_sep_xfit)
            fairness_summary = metrics.regression_fairness_summary(sensitive, n_bins=self.n_bins)
            for metric_name, metric_value in fairness_summary.items():
                extra_metrics[metric_name].append(metric_value)
                result[f"{metric_name}_{protected_name}"] = metric_value
            result.update({
                f"r_sep_{protected_name}": protected_r_sep,
                f"i_sep_{protected_name}": protected_i_sep,
                f"c_sep_xfit_{protected_name}": protected_c_sep_xfit,
            })
        result.update({
            "mse": metrics.mse(),
            "mae": metrics.mae(),
            "r2": metrics.r2(),
            "r_sep": nanmean_or_nan(r_sep),
            "i_sep": nanmean_or_nan(i_sep),
            "c_sep_xfit": nanmean_or_nan(c_sep_xfit),
        })
        result.update({
            metric_name: nanmean_or_nan(values)
            for metric_name, values in extra_metrics.items()
        })
        return result


def fit_estimator(estimator, X, y, sample_weight=None):
    if sample_weight is not None and has_fit_parameter(estimator, "sample_weight"):
        estimator.fit(X, y, sample_weight=sample_weight)
        return
    estimator.fit(X, y)


def nanmean_or_nan(values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return np.nan
    return float(np.nanmean(values))


def safe_metric_name(name):
    return str(name).replace(" ", "_").replace("%", "pct")


def numeric_sensitive_frame(values):
    frame = pd.DataFrame(values).reset_index(drop=True)
    result = pd.DataFrame(index=frame.index)
    for column in frame:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            result[str(column)] = series.astype(float)
        else:
            dummies = pd.get_dummies(series.astype(str), prefix=str(column), dtype=float)
            result = pd.concat([result, dummies], axis=1)
    return result.to_numpy(dtype=float)


def similarity_matrix_without_sensitive(X, protected):
    frame = pd.DataFrame(X).reset_index(drop=True)
    feature_frame = frame.drop(columns=list(protected), errors="ignore")
    if feature_frame.shape[1] == 0:
        feature_frame = frame
    feature_frame = pd.get_dummies(feature_frame, dummy_na=True, dtype=float)
    return StandardScaler().fit_transform(feature_frame.to_numpy(dtype=float))


def fit_berk_pairwise_linear(X, y, groups, similarity):
    X_aug = add_intercept(X)
    objective = X_aug.T @ X_aug / len(X_aug)
    penalty = berk_pairwise_penalty(X_aug, groups, similarity)
    ridge = BERK_RIDGE * np.eye(X_aug.shape[1])
    ridge[0, 0] = 0.0
    target = X_aug.T @ y / len(X_aug)
    return np.linalg.pinv(objective + BERK_PAIRWISE_LAMBDA * penalty + ridge) @ target


def berk_pairwise_penalty(X_aug, groups, similarity):
    groups = np.asarray(groups)
    penalty = np.zeros((X_aug.shape[1], X_aug.shape[1]), dtype=float)
    pair_count = 0
    for group in pd.Series(groups).dropna().unique():
        source_idx = np.flatnonzero(groups == group)
        target_idx = np.flatnonzero(groups != group)
        if len(source_idx) == 0 or len(target_idx) == 0:
            continue
        k = min(BERK_PAIRWISE_NEIGHBORS, len(target_idx))
        neighbors = NearestNeighbors(n_neighbors=k).fit(similarity[target_idx])
        nearest = neighbors.kneighbors(similarity[source_idx], return_distance=False)
        for row, neighbor_rows in enumerate(nearest):
            i = source_idx[row]
            for neighbor_row in neighbor_rows:
                j = target_idx[neighbor_row]
                diff = X_aug[i] - X_aug[j]
                penalty += np.outer(diff, diff)
                pair_count += 1
    if pair_count == 0:
        return penalty
    return penalty / pair_count


def add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def sensitive_groups(train_values, test_values=None, n_bins=5):
    train = pd.DataFrame(train_values).reset_index(drop=True)
    test = None if test_values is None else pd.DataFrame(test_values).reset_index(drop=True)
    train_parts = []
    test_parts = []
    for column in train:
        train_col = train[column]
        test_col = None if test is None else test[column]
        if is_categorical_sensitive(train_col):
            train_part = train_col.astype(str).to_numpy()
            test_part = None if test_col is None else test_col.astype(str).to_numpy()
        else:
            train_part, test_part = quantile_groups(train_col, test_col, n_bins)
        train_parts.append(train_part.astype(str))
        if test_part is not None:
            test_parts.append(test_part.astype(str))
    train_groups = combine_group_parts(train_parts)
    test_groups = None if test is None else combine_group_parts(test_parts)
    return train_groups, test_groups


def is_categorical_sensitive(values):
    return pd.Series(values).nunique(dropna=False) <= 20


def quantile_groups(train_col, test_col, n_bins):
    train_numeric = pd.to_numeric(train_col, errors="coerce").fillna(0).to_numpy(dtype=float)
    try:
        _, bins = pd.qcut(train_numeric, q=n_bins, retbins=True, duplicates="drop")
    except ValueError:
        bins = np.array([np.min(train_numeric), np.max(train_numeric)])
    bins = np.unique(bins)
    if len(bins) <= 2 or np.isclose(bins[0], bins[-1]):
        train_part = np.zeros(len(train_numeric), dtype=int)
        test_part = None if test_col is None else np.zeros(len(test_col), dtype=int)
        return train_part, test_part
    train_part = np.searchsorted(bins[1:-1], train_numeric, side="right")
    if test_col is None:
        return train_part, None
    test_numeric = pd.to_numeric(test_col, errors="coerce").fillna(0).to_numpy(dtype=float)
    test_part = np.searchsorted(bins[1:-1], test_numeric, side="right")
    return train_part, test_part


def combine_group_parts(parts):
    if not parts:
        return np.array([])
    combined = parts[0].copy()
    for part in parts[1:]:
        combined = np.char.add(np.char.add(combined, "|"), part)
    return combined


def quantile_repair(train_pred, test_pred, train_groups, test_groups):
    train_pred = np.asarray(train_pred, dtype=float)
    test_pred = np.asarray(test_pred, dtype=float)
    repaired = test_pred.copy()
    target = np.sort(train_pred)
    if len(target) == 0:
        return repaired
    for group in pd.Series(test_groups).dropna().unique():
        source = np.sort(train_pred[train_groups == group])
        mask = test_groups == group
        if len(source) < 2 or not mask.any():
            continue
        ranks = np.searchsorted(source, test_pred[mask], side="right")
        quantiles = np.clip((ranks - 0.5) / len(source), 0.0, 1.0)
        repaired[mask] = np.quantile(target, quantiles)
    return repaired


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(weights))
    return len(weights) * weights / total


def build_estimator(name, task_type, seed):
    key = str(name).lower()
    if task_type == "classification":
        models = {
            "logistic": LogisticRegression(max_iter=1000, random_state=seed),
        }
    else:
        models = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=1.0, random_state=seed),
            "rf": RandomForestRegressor(n_estimators=50, random_state=seed, n_jobs=-1),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(16,),
                max_iter=50,
                early_stopping=True,
                n_iter_no_change=5,
                tol=1e-3,
                random_state=seed,
            ),
            "vgg_face": ScutVGGFaceRegressor(seed=seed),
        }
    if key not in models:
        raise ValueError(f"Unknown {task_type} model: {name}")
    return models[key]


def default_model_for_task(task_type):
    return "logistic" if task_type == "classification" else "linear"


def has_image_features(X):
    return hasattr(X, "columns") and "image_path" in X.columns


def is_raw_image_model(estimator):
    return bool(getattr(estimator, "uses_raw_images", False))


def normalize_method(method):
    aliases = {
        "none": "none",
        "fairreweighing": "fair-reweighing",
        "fair-reweighing": "fair-reweighing",
        "reweighing": "fair-reweighing",
        "berk": "berk-pairwise",
        "berk-pairwise": "berk-pairwise",
        "calders": "calders-effect-control",
        "calders-effect-control": "calders-effect-control",
        "chzhen": "chzhen-repair",
        "chzhen-repair": "chzhen-repair",
        "agarwal": "agarwal-bgl",
        "agarwal-bgl": "agarwal-bgl",
        "groundtruth": "groundtruth",
    }
    key = aliases.get(str(method).replace("_", "-").lower())
    if key is None:
        raise ValueError(f"Unknown method: {method}")
    return key


def normalize_density(density_model):
    aliases = {
        None: "Neighbor",
        "none": "Neighbor",
        "neighbor": "Neighbor",
        "neighbors": "Neighbor",
        "kernel": "Kernel",
        "kde": "Kernel",
        "reweighing": "Reweighing",
        "discrete": "Reweighing",
    }
    key = aliases.get(str(density_model).lower() if density_model is not None else None)
    if key is None:
        raise ValueError(f"Unknown density model: {density_model}")
    return key
