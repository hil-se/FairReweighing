import math

import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.stats import norm, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression

try:
    from fairlearn.metrics import demographic_parity_difference
except ImportError:
    demographic_parity_difference = None

from DP_helper import extract_group_pred, pmf2disp, weighted_pmf


EPS = 1e-12


class Metrics:
    def __init__(self, y, y_pred):
        self.y = np.asarray(y)
        self.y_pred = np.asarray(y_pred)

    def mse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def rmse(self):
        return math.sqrt(self.mse())

    def r2(self):
        if len(np.unique(self.y)) <= 1:
            return np.nan
        return sklearn.metrics.r2_score(self.y, self.y_pred)

    def accuracy(self):
        return sklearn.metrics.accuracy_score(self.y, self._class_predictions())

    def f1(self):
        return sklearn.metrics.f1_score(self.y, self._class_predictions(), zero_division=0)

    def precision(self):
        return sklearn.metrics.precision_score(self.y, self._class_predictions(), zero_division=0)

    def recall(self):
        return sklearn.metrics.recall_score(self.y, self._class_predictions(), zero_division=0)

    def pearsonr_coefficient(self):
        if len(np.unique(self.y)) <= 1 or len(np.unique(self.y_pred)) <= 1:
            return np.nan
        return pearsonr(self.y, self.y_pred)[0]

    def spearmanr_coefficient(self):
        if len(np.unique(self.y)) <= 1 or len(np.unique(self.y_pred)) <= 1:
            return np.nan
        return spearmanr(self.y, self.y_pred)[0]

    def DP(self, s):
        if demographic_parity_difference is None:
            return np.nan
        return demographic_parity_difference(self.y, self._class_predictions(), sensitive_features=s)

    def EOD(self, s):
        groups = _binary_groups(s)
        if groups is None:
            return np.nan
        y0, y1, y0_pred, y1_pred = self._grouped(groups, s)
        tp, fp, tn, fn = self.confusion(y0, y0_pred)
        denom0 = tp + fn
        tp, fp, tn, fn = self.confusion(y1, y1_pred)
        denom1 = tp + fn
        if denom0 == 0 or denom1 == 0:
            return np.nan
        return float(tp) / denom1 - float(self.confusion(y0, y0_pred)[0]) / denom0

    def AOD(self, s):
        groups = _binary_groups(s)
        if groups is None:
            return np.nan
        y0, y1, y0_pred, y1_pred = self._grouped(groups, s)
        tp, fp, tn, fn = self.confusion(y0, y0_pred)
        if (tp + fn) == 0 or (fp + tn) == 0:
            return np.nan
        od0 = float(tp) / (tp + fn) + float(fp) / (fp + tn)
        tp, fp, tn, fn = self.confusion(y1, y1_pred)
        if (tp + fn) == 0 or (fp + tn) == 0:
            return np.nan
        od1 = float(tp) / (tp + fn) + float(fp) / (fp + tn)
        return (od1 - od0) / 2

    def confusion(self, y, y_pred):
        tp = fp = tn = fn = 0
        for true, pred in zip(y, y_pred):
            if true > 0:
                if pred > 0:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred > 0:
                    fp += 1
                else:
                    tn += 1
        return tp, fp, tn, fn

    def bgl(self, s, n_bins=2):
        groups = _groups_for_sensitive(s, n_bins=n_bins)
        errors = []
        for group in groups:
            mask = group["mask"]
            if mask.sum() > 0:
                errors.append(sklearn.metrics.mean_squared_error(self.y[mask], self.y_pred[mask]))
        return max(errors) if errors else np.nan

    def continuous_mi(self, s):
        s = np.asarray(s, dtype=float)
        if len(np.unique(s)) <= 1:
            return 0.0
        joint = pd.DataFrame({"y": self.y, "y_pred": self.y_pred}, columns=["y", "y_pred"])
        margin = self.y.reshape(-1, 1)

        model_joint = LinearRegression().fit(joint, s)
        model_margin = LinearRegression().fit(margin, s)
        pred_joint = model_joint.predict(joint)
        pred_margin = model_margin.predict(margin)

        rse_joint = max(np.std(pred_joint - s), EPS)
        rse_margin = max(np.std(pred_margin - s), EPS)
        pdf_joint = np.maximum(norm.pdf(s, pred_joint, rse_joint), EPS)
        pdf_margin = np.maximum(norm.pdf(s, pred_margin, rse_margin), EPS)
        return float(np.mean(np.log(pdf_joint / pdf_margin)))

    def continuous_mi_normalized(self, s):
        s = np.asarray(s, dtype=float)
        if len(np.unique(s)) <= 1:
            return 0.0
        margin = self.y.reshape(-1, 1)
        model_margin = LinearRegression().fit(margin, s)
        pred_margin = model_margin.predict(margin)
        rse_margin = max(np.std(pred_margin - s), EPS)
        pdf_margin = np.maximum(norm.pdf(s, pred_margin, rse_margin), EPS)
        entropy_proxy = -float(np.mean(np.log(pdf_margin)))
        if abs(entropy_proxy) < EPS:
            return np.nan
        return self.continuous_mi(s) / entropy_proxy

    def r_sep(self, s):
        groups = _binary_groups(s)
        if groups is None:
            return np.nan
        s = np.asarray(s)
        joint = pd.DataFrame({"y": self.y, "y_pred": self.y_pred}, columns=["y", "y_pred"])
        margin = self.y.reshape(-1, 1)
        try:
            model_joint = LogisticRegression(max_iter=1000).fit(joint, s)
            model_margin = LogisticRegression(max_iter=1000).fit(margin, s)
        except ValueError:
            return np.nan

        prob_joint = np.clip(model_joint.predict_proba(joint)[:, 1], EPS, 1 - EPS)
        prob_margin = np.clip(model_margin.predict_proba(margin)[:, 1], EPS, 1 - EPS)
        ratio = (prob_joint / (1 - prob_joint)) * ((1 - prob_margin) / prob_margin)
        return float(np.mean(ratio))

    def r_sep_a(self, s):
        groups = _binary_groups(s)
        if groups is None:
            return np.nan
        joint = pd.DataFrame({"y": self.y, "y_pred": self.y_pred}, columns=["y", "y_pred"])
        try:
            prob_joint = LogisticRegression(max_iter=1000).fit(joint, s).predict_proba(joint)[:, 1]
        except ValueError:
            return np.nan
        prob_joint = np.clip(prob_joint, EPS, 1 - EPS)
        return float(np.mean(prob_joint / (1 - prob_joint)))

    def DP_disp(self, s, Theta):
        pred_group = extract_group_pred(self.y_pred, s)
        pmf_all = weighted_pmf(self.y_pred, Theta)
        pmf_group = [weighted_pmf(pred_group[g], Theta) for g in pred_group]
        return max([pmf2disp(pmf_g, pmf_all) for pmf_g in pmf_group])

    def fairness_summary(self, s, prefix):
        s = np.asarray(s)
        summary = {
            f"bgl_{prefix}": self.bgl(s),
            f"continuous_mi_{prefix}": self.continuous_mi(s),
            f"continuous_mi_norm_{prefix}": self.continuous_mi_normalized(s),
            f"r_sep_{prefix}": self.r_sep(s),
        }
        if _is_binary_target(self.y):
            summary.update({
                f"dp_{prefix}": self.DP(s),
                f"aod_{prefix}": self.AOD(s),
                f"eod_{prefix}": self.EOD(s),
            })
        return summary

    def _class_predictions(self):
        if _is_binary_target(self.y):
            labels = np.sort(np.unique(self.y))
            return np.where(self.y_pred >= 0.5, labels[-1], labels[0])
        return np.rint(self.y_pred)

    def _grouped(self, groups, s):
        s = np.asarray(s)
        mask0 = s == groups[0]
        mask1 = s == groups[1]
        return self.y[mask0], self.y[mask1], self._class_predictions()[mask0], self._class_predictions()[mask1]


def _is_binary_target(y):
    return len(np.unique(y)) == 2


def _binary_groups(s):
    unique = np.unique(np.asarray(s))
    return unique if len(unique) == 2 else None


def _groups_for_sensitive(s, n_bins=2):
    s = np.asarray(s)
    unique = np.unique(s)
    if len(unique) <= 10:
        return [{"label": value, "mask": s == value} for value in unique]
    bins = pd.qcut(pd.Series(s), q=n_bins, labels=False, duplicates="drop")
    return [{"label": value, "mask": bins.to_numpy() == value} for value in np.unique(bins.dropna())]
