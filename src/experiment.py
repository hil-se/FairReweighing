import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import has_fit_parameter

from data_reader import load_dataset
from density_balance import DensityBalance
from metrics import Metrics


DEFAULT_RADIUS_GRID = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
DEFAULT_BANDWIDTH_GRID = [0.05, 0.1, 0.2, 0.5, 1.0]


class Experiment:
    def __init__(
        self,
        data="Community",
        regressor="auto",
        balance="none",
        density_model="Neighbor",
        seed=0,
        test_size=0.5,
        tune_density=False,
        radius_grid=None,
        bandwidth_grid=None,
        n_bins=5,
        dataset_options=None,
    ):
        self.data = data
        self.seed = seed
        self.balance = normalize_method(balance)
        self.density_model = normalize_density(density_model)
        self.test_size = test_size
        self.tune_density = tune_density
        self.radius_grid = radius_grid or DEFAULT_RADIUS_GRID
        self.bandwidth_grid = bandwidth_grid or DEFAULT_BANDWIDTH_GRID
        self.n_bins = n_bins
        self.dataset_options = dataset_options or {}
        self.X, self.y, self.protected = load_dataset(data, seed=seed, **self.dataset_options)
        self.task_type = "classification" if len(np.unique(self.y)) == 2 else "regression"
        self.regressor_name = default_model_for_task(self.task_type) if regressor in (None, "auto") else regressor
        self.regressor = build_estimator(self.regressor_name, self.task_type, seed)
        self.preprocessor = None
        self.selected_density_param = None
        self.weight_examples = pd.DataFrame()
        self.sample_weight_applied = False

    def run(self):
        started = time.perf_counter()
        self.train_test_split()
        self.preprocess(self.X_train)
        X_train = self.preprocessor.transform(self.X_train)
        X_test = self.preprocessor.transform(self.X_test)

        fit_started = time.perf_counter()
        sample_weight = self.make_sample_weight()
        estimator = clone(self.regressor)
        self.sample_weight_applied = fit_estimator(estimator, X_train, self.y_train, sample_weight)
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
            "task_type": self.task_type,
            "method": self.balance,
            "density_model": self.density_model if self.balance == "fair-reweighing" else "none",
            "seed": self.seed,
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "fit_seconds": fit_seconds,
            "total_seconds": time.perf_counter() - started,
            "selected_radius": self.selected_density_param if self.density_model == "Neighbor" else np.nan,
            "selected_bandwidth": self.selected_density_param if self.density_model == "Kernel" else np.nan,
            "sample_weight_applied": self.sample_weight_applied,
        })
        result.update(weight_summary(sample_weight))
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
            self.weight_examples = pd.DataFrame()
            return None

        if self.balance == "fair-reweighing" and self.tune_density:
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
        self.weight_examples = self._weight_examples(weights)
        return weights

    def select_density_param(self):
        grid = self.radius_grid if self.density_model == "Neighbor" else self.bandwidth_grid
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
                fairness = []
                for protected in self.protected:
                    fairness.append(abs(metrics.continuous_mi(X_val[protected].to_numpy())))
                score = float(np.nanmean(fairness)) + 0.01 * metrics.mse()
            except Exception:
                continue
            if score < best_score:
                best_score = score
                best_param = param
        return best_param

    def evaluate(self, y_pred):
        metrics = Metrics(self.y_test, y_pred)
        result = {
            "mse": metrics.mse(),
            "mae": metrics.mae(),
            "rmse": metrics.rmse(),
            "r2": metrics.r2(),
            "pearson": metrics.pearsonr_coefficient(),
            "spearman": metrics.spearmanr_coefficient(),
        }
        if self.task_type == "classification":
            result.update({
                "accuracy": metrics.accuracy(),
                "precision": metrics.precision(),
                "recall": metrics.recall(),
                "f1": metrics.f1(),
            })
        continuous_mi_values = []
        bgl_values = []
        for protected in self.protected:
            protected_name = str(protected).replace(" ", "_")
            summary = metrics.fairness_summary(self.X_test[protected].to_numpy(), protected_name)
            result.update(summary)
            continuous_mi_values.append(abs(summary[f"continuous_mi_{protected_name}"]))
            bgl_values.append(summary[f"bgl_{protected_name}"])
        result["fairness_mean_abs_continuous_mi"] = float(np.nanmean(continuous_mi_values))
        result["fairness_max_bgl"] = float(np.nanmax(bgl_values))
        return result

    def _weight_examples(self, weights, n=5):
        if weights is None:
            return pd.DataFrame()
        frame = self.X_train[self.protected].copy()
        frame["y"] = self.y_train
        frame["sample_weight"] = weights
        low = frame.nsmallest(n, "sample_weight").assign(weight_rank="low")
        high = frame.nlargest(n, "sample_weight").assign(weight_rank="high")
        examples = pd.concat([low, high], ignore_index=True)
        examples.insert(0, "seed", self.seed)
        examples.insert(0, "density_model", self.density_model)
        examples.insert(0, "method", self.balance)
        examples.insert(0, "model", self.regressor_name)
        examples.insert(0, "dataset", self.data)
        return examples


def fit_estimator(estimator, X, y, sample_weight=None):
    if sample_weight is not None and has_fit_parameter(estimator, "sample_weight"):
        estimator.fit(X, y, sample_weight=sample_weight)
        return True
    estimator.fit(X, y)
    return False


def weight_summary(weights):
    if weights is None:
        return {
            "weight_mean": np.nan,
            "weight_std": np.nan,
            "weight_min": np.nan,
            "weight_max": np.nan,
            "weight_p01": np.nan,
            "weight_p99": np.nan,
            "weight_effective_n": np.nan,
        }
    weights = np.asarray(weights, dtype=float)
    return {
        "weight_mean": float(np.mean(weights)),
        "weight_std": float(np.std(weights)),
        "weight_min": float(np.min(weights)),
        "weight_max": float(np.max(weights)),
        "weight_p01": float(np.quantile(weights, 0.01)),
        "weight_p99": float(np.quantile(weights, 0.99)),
        "weight_effective_n": float((weights.sum() ** 2) / np.sum(weights ** 2)),
    }


def build_estimator(name, task_type, seed):
    key = str(name).lower()
    if task_type == "classification":
        models = {
            "logistic": LogisticRegression(max_iter=1000, random_state=seed),
            "rf": RandomForestClassifier(n_estimators=200, random_state=seed),
            "randomforest": RandomForestClassifier(n_estimators=200, random_state=seed),
        }
    else:
        models = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=1.0, random_state=seed),
            "svr": SVR(kernel="rbf"),
            "dt": DecisionTreeRegressor(max_depth=8, random_state=seed),
            "rf": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
            "randomforest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1),
            "gbr": GradientBoostingRegressor(random_state=seed),
            "gradientboosting": GradientBoostingRegressor(random_state=seed),
            "mlp": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
        }
    if key not in models:
        raise ValueError(f"Unknown {task_type} model: {name}")
    return models[key]


def default_model_for_task(task_type):
    return "logistic" if task_type == "classification" else "linear"


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
