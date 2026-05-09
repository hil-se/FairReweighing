import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.stats import norm, pearsonr, spearmanr, wasserstein_distance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold


EPS = 1e-12
MAX_CATEGORICAL_GROUPS = 20


class Metrics:
    def __init__(self, y, y_pred):
        self.y = np.asarray(y)
        self.y_pred = np.asarray(y_pred)

    def mse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def r2(self):
        if len(np.unique(self.y)) <= 1:
            return np.nan
        return sklearn.metrics.r2_score(self.y, self.y_pred)

    def categorical_separation(self, s, seed=0, n_splits=10):
        s = np.asarray(s)
        result = {"r_sep": np.nan, "i_sep": np.nan}
        if not _is_categorical(s):
            return result

        joint_actual, margin_actual, joint_positive, margin_positive = self._crossfit_classification_probabilities(
            s,
            seed,
            n_splits,
        )

        mi_mask = np.isfinite(joint_actual) & np.isfinite(margin_actual)
        if mi_mask.any():
            cmi = np.mean(np.log(joint_actual[mi_mask] / margin_actual[mi_mask]))
            conditional_entropy = -np.mean(np.log(margin_actual[mi_mask]))
            if conditional_entropy > EPS:
                result["i_sep"] = float(cmi / conditional_entropy)

        if len(np.unique(s)) == 2:
            ratio_mask = np.isfinite(joint_positive) & np.isfinite(margin_positive)
            if ratio_mask.any():
                odds_joint = joint_positive[ratio_mask] / (1 - joint_positive[ratio_mask])
                odds_margin = (1 - margin_positive[ratio_mask]) / margin_positive[ratio_mask]
                result["r_sep"] = float(np.mean(odds_joint * odds_margin))
        return result

    def r_sep(self, s, seed=0, n_splits=10):
        return self.categorical_separation(s, seed, n_splits)["r_sep"]

    def i_sep(self, s, seed=0, n_splits=10):
        return self.categorical_separation(s, seed, n_splits)["i_sep"]

    def continuous_mi_crossfit(self, s, seed=0, n_splits=10):
        s = np.asarray(s)
        if len(np.unique(s)) <= 1:
            return 0.0

        if _is_categorical(s):
            value = self._categorical_mi_crossfit(s, seed, n_splits)
        else:
            value = self._gaussian_mi_crossfit(s.astype(float), seed, n_splits)
        if np.isnan(value):
            return np.nan
        return max(0.0, value)

    def regression_fairness_summary(self, s, n_bins=5):
        s = np.asarray(s)
        groups = _groups_for_sensitive(s, n_bins)
        residual = self.y_pred - self.y
        return {
            "pred_pearson_abs": _abs_corr(s, self.y_pred, pearsonr),
            "pred_spearman_abs": _abs_corr(s, self.y_pred, spearmanr),
            "pred_mean_gap": _max_group_gap(self.y_pred, groups),
            "residual_mean_gap": _max_group_gap(residual, groups),
            "group_mse_gap": _max_group_mse_gap(self.y, self.y_pred, groups),
            "wasserstein_pred_gap": _max_pairwise_wasserstein(self.y_pred, groups),
        }

    def _joint_frame(self):
        return pd.DataFrame({"y": self.y, "y_pred": self.y_pred})

    def _margin_frame(self):
        return self.y.reshape(-1, 1)

    def _crossfit_classification_probabilities(self, s, seed, n_splits):
        joint_x = self._joint_frame()
        margin_x = self._margin_frame()
        n = len(s)
        positive = np.sort(np.unique(s))[-1]
        joint_actual = np.full(n, np.nan)
        margin_actual = np.full(n, np.nan)
        joint_positive = np.full(n, np.nan)
        margin_positive = np.full(n, np.nan)

        for train, test in _folds(s, seed, n_splits):
            if len(np.unique(s[train])) < 2:
                continue
            try:
                joint_model = LogisticRegression(max_iter=1000).fit(joint_x.iloc[train], s[train])
                margin_model = LogisticRegression(max_iter=1000).fit(margin_x[train], s[train])
            except ValueError:
                continue
            joint_actual[test] = _actual_class_prob(joint_model, joint_x.iloc[test], s[test])
            margin_actual[test] = _actual_class_prob(margin_model, margin_x[test], s[test])
            joint_positive[test] = _class_prob(joint_model, joint_x.iloc[test], positive)
            margin_positive[test] = _class_prob(margin_model, margin_x[test], positive)

        return joint_actual, margin_actual, joint_positive, margin_positive

    def _categorical_mi_crossfit(self, s, seed, n_splits):
        joint_x = self._joint_frame()
        margin_x = self._margin_frame()
        log_joint = []
        log_margin = []
        for train, test in _folds(s, seed, n_splits):
            try:
                joint_model = LogisticRegression(max_iter=1000).fit(joint_x.iloc[train], s[train])
                margin_model = LogisticRegression(max_iter=1000).fit(margin_x[train], s[train])
            except ValueError:
                continue
            log_joint.extend(np.log(_actual_class_prob(joint_model, joint_x.iloc[test], s[test])))
            log_margin.extend(np.log(_actual_class_prob(margin_model, margin_x[test], s[test])))
        if not log_joint:
            return np.nan
        return float(np.mean(np.asarray(log_joint) - np.asarray(log_margin)))

    def _gaussian_mi_crossfit(self, s, seed, n_splits):
        joint_x = self._joint_frame()
        margin_x = self._margin_frame()
        log_joint = []
        log_margin = []
        for train, test in _folds(s, seed, n_splits):
            joint_model = LinearRegression().fit(joint_x.iloc[train], s[train])
            margin_model = LinearRegression().fit(margin_x[train], s[train])

            joint_train_pred = joint_model.predict(joint_x.iloc[train])
            margin_train_pred = margin_model.predict(margin_x[train])
            joint_scale = max(np.std(s[train] - joint_train_pred), EPS)
            margin_scale = max(np.std(s[train] - margin_train_pred), EPS)

            joint_pred = joint_model.predict(joint_x.iloc[test])
            margin_pred = margin_model.predict(margin_x[test])
            log_joint.extend(norm.logpdf(s[test], joint_pred, joint_scale))
            log_margin.extend(norm.logpdf(s[test], margin_pred, margin_scale))
        if not log_joint:
            return np.nan
        return float(np.mean(np.asarray(log_joint) - np.asarray(log_margin)))


def _is_categorical(s):
    return 1 < len(np.unique(s)) <= MAX_CATEGORICAL_GROUPS


def _groups_for_sensitive(s, n_bins):
    series = pd.Series(np.asarray(s))
    if _is_categorical(series.to_numpy()):
        return series.to_numpy()
    try:
        return pd.qcut(series, q=n_bins, labels=False, duplicates="drop").to_numpy()
    except ValueError:
        return np.zeros(len(series), dtype=int)


def _abs_corr(x, y, corr_fn):
    x = _numeric_array(x)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2 or len(np.unique(x[mask])) <= 1 or len(np.unique(y[mask])) <= 1:
        return np.nan
    return float(abs(corr_fn(x[mask], y[mask]).statistic))


def _numeric_array(values):
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        codes, _ = pd.factorize(values)
        return codes.astype(float)


def _max_group_gap(values, groups):
    summaries = _group_values(values, groups)
    means = [np.mean(group) for group in summaries if len(group) > 0]
    if len(means) < 2:
        return np.nan
    return float(np.max(means) - np.min(means))


def _max_group_mse_gap(y, y_pred, groups):
    y_groups = _group_values(y, groups)
    pred_groups = _group_values(y_pred, groups)
    mses = [
        sklearn.metrics.mean_squared_error(y_group, pred_group)
        for y_group, pred_group in zip(y_groups, pred_groups)
        if len(y_group) > 0
    ]
    if len(mses) < 2:
        return np.nan
    return float(np.max(mses) - np.min(mses))


def _max_pairwise_wasserstein(values, groups):
    summaries = [group for group in _group_values(values, groups) if len(group) > 0]
    if len(summaries) < 2:
        return np.nan
    distances = []
    for left in range(len(summaries)):
        for right in range(left + 1, len(summaries)):
            distances.append(wasserstein_distance(summaries[left], summaries[right]))
    return float(np.max(distances))


def _group_values(values, groups):
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    result = []
    for group in pd.Series(groups).dropna().unique():
        mask = groups == group
        group_values = values[mask]
        group_values = group_values[np.isfinite(group_values)]
        result.append(group_values)
    return result


def _folds(values, seed, n_splits):
    values = np.asarray(values)
    n = len(values)
    k = min(n_splits, n)
    if k < 2:
        return []
    if _is_categorical(values):
        counts = pd.Series(values).value_counts()
        if counts.min() >= 2:
            k = min(k, int(counts.min()))
            return StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n), values)
    return KFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(n))


def _actual_class_prob(model, X, values):
    probs = _probability_matrix(model, X)
    selected = []
    for row, value in enumerate(values):
        index = _class_index(model, value)
        selected.append(EPS if index is None else probs[row, index])
    return np.clip(np.asarray(selected), EPS, 1 - EPS)


def _class_prob(model, X, value):
    probs = _probability_matrix(model, X)
    index = _class_index(model, value)
    if index is None:
        return np.full(len(probs), EPS)
    return np.clip(probs[:, index], EPS, 1 - EPS)


def _probability_matrix(model, X):
    return model.predict_proba(X)


def _class_index(model, value):
    class_index = {label: index for index, label in enumerate(model.classes_)}
    return class_index.get(value)
