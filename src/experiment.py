import time

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import has_fit_parameter

from data_reader import load_dataset
from density_balance import DensityBalance
from metrics import Metrics
from vgg_face_model import ScutVGGFaceRegressor


DEFAULT_RADIUS_GRID = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
DEFAULT_BANDWIDTH_GRID = [0.05, 0.1, 0.2, 0.5, 1.0]
DEFAULT_TEST_SIZE = 0.5
DEFAULT_BINS = 5


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
        sample_weight = self.make_sample_weight()
        estimator = clone(self.regressor)
        fit_estimator(estimator, X_train, self.y_train, sample_weight)
        fit_seconds = time.perf_counter() - fit_started

        y_pred = estimator.predict(X_test)
        if self.task_type == "classification" and hasattr(estimator, "predict_proba"):
            y_pred_for_metrics = estimator.predict_proba(X_test)[:, 1]
        else:
            y_pred_for_metrics = y_pred

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

    def select_density_param(self):
        if is_raw_image_model(self.regressor):
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
        c_sep = []
        c_sep_xfit = []
        result = {}
        for protected in self.protected:
            protected_name = safe_metric_name(protected)
            sensitive = self.X_test[protected].to_numpy()
            categorical_sep = metrics.categorical_separation(sensitive, seed=self.seed)
            protected_r_sep = categorical_sep["r_sep"]
            protected_i_sep = categorical_sep["i_sep"]
            protected_c_sep = metrics.continuous_mi(sensitive)
            protected_c_sep_xfit = metrics.continuous_mi_crossfit(sensitive, seed=self.seed)
            r_sep.append(protected_r_sep)
            i_sep.append(protected_i_sep)
            c_sep.append(protected_c_sep)
            c_sep_xfit.append(protected_c_sep_xfit)
            result.update({
                f"r_sep_{protected_name}": protected_r_sep,
                f"i_sep_{protected_name}": protected_i_sep,
                f"c_sep_{protected_name}": protected_c_sep,
                f"c_sep_xfit_{protected_name}": protected_c_sep_xfit,
            })
        result.update({
            "mse": metrics.mse(),
            "mae": metrics.mae(),
            "r2": metrics.r2(),
            "r_sep": nanmean_or_nan(r_sep),
            "i_sep": nanmean_or_nan(i_sep),
            "c_sep": nanmean_or_nan(c_sep),
            "c_sep_xfit": nanmean_or_nan(c_sep_xfit),
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
            "rf": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
            "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
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
        "discretized": "discretized-reweighing",
        "discretized-reweighing": "discretized-reweighing",
        "original-reweighing": "discretized-reweighing",
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
