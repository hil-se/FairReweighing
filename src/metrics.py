import numpy as np
import pandas as pd
import sklearn.metrics
import torch
from fairlearn.metrics import MetricFrame
from kde import kde_fair

from src.DP_helper import weighted_pmf, extract_group_pred, pmf2disp
from fairlearn.reductions import BoundedGroupLoss, ZeroOneLoss

class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    def bgl_mse(self, s):
        s.name = "Sen"
        mse_frame = MetricFrame(metrics=sklearn.metrics.mean_squared_error,
                            y_true=self.y,
                            y_pred=self.y_pred,
                            sensitive_features=s)

        return max(mse_frame.by_group)

    def bgl_mae(self, s):
        s.name = "Sen"
        mae_frame = MetricFrame(metrics=sklearn.metrics.mean_absolute_error,
                                y_true=self.y,
                                y_pred=self.y_pred,
                                sensitive_features=s)

        return max(mae_frame.by_group)

    def mse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

    def mae(self):
        return sklearn.metrics.mean_absolute_error(self.y, self.y_pred)

    def rmse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred, squared=False)

    def accuracy(self):
        return sklearn.metrics.accuracy_score(self.y, self.y_pred)

    def f1(self):
        return sklearn.metrics.f1_score(self.y, self.y_pred)

    def r2(self):
        return sklearn.metrics.r2_score(self.y, self.y_pred)

    def AOD(self, s):
        # s is an array of numerical values of a sensitive attribute
        t = n = tp = fp = tn = fn = 0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if s[i] - s[j] > 0:
                    if self.y[i] - self.y[j] > 0:
                        t += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            tp += 1
                        if self.y_pred[i] < self.y_pred[j]:
                            fn += 1
                    elif self.y[j] - self.y[i] > 0:
                        n += 1
                        if self.y_pred[i] > self.y_pred[j]:
                            fp += 1
                        elif self.y_pred[i] < self.y_pred[j]:
                            tn += 1

        tpr = tp / t
        tnr = tn / n
        fpr = fp / n
        fnr = fn / t
        aod = (tpr + fpr - tnr - fnr) / 2
        return aod

    def AODc(self, s):
        # s is an array of numerical values of a sensitive attribute
        t = n = tp = fp = 0.0
        for i in range(len(self.y)):
            for j in range(len(self.y)):
                if s[i] - s[j] > 0:
                    y_diff = self.y[i] - self.y[j]
                    y_pred_diff = self.y_pred[i] - self.y_pred[j]
                    if y_diff > 0:
                        t += y_diff
                        tp += y_pred_diff
                    elif y_diff < 0:
                        n += y_diff
                        fp += y_pred_diff

        aod = (tp / t - fp / n) / 2
        return aod

    def DP_disp(self, s, Theta):
        pred_group = extract_group_pred(self.y_pred, s)
        PMF_all = weighted_pmf(self.y_pred, Theta)
        PMF_group = [weighted_pmf(pred_group[g], Theta) for g in pred_group]
        DP_disp = max([pmf2disp(PMF_g, PMF_all) for PMF_g in PMF_group])
        return DP_disp

    def GDP(self, s):
        test_sol = 1e-3
        device_gpu = torch.device("mps")
        x_appro = torch.arange(test_sol, 1 - test_sol, test_sol).to(device_gpu)
        KDE_FAIR = kde_fair(x_appro)
        penalty = KDE_FAIR.forward
        y_pred = torch.tensor(self.y_pred.astype(np.float32)).to(device_gpu)
        s = torch.tensor(s.astype(np.float32)).to(device_gpu)
        DP_test = penalty(y_pred, s, device_gpu).item()

        return DP_test

    def convex_individual(self, s):
        y0 = self.y[s == 0]
        y1 = self.y[s == 1]
        y0_pred = self.y_pred[s == 0]
        y1_pred = self.y_pred[s == 1]

        def convex_ind(y0, y1, y0_pred, y1_pred):
            if isinstance(y0, float):
                d = np.exp(-(y0 - y1) ** 2)
            else:
                d = 1 if y0 == y1 else 0
            return (y0_pred - y1_pred) ** 2 * d

        error = 0
        for i in range(len(y0)):
            for j in range(len(y1)):
                error += convex_ind(y0[i], y1[j], y0_pred[i], y1_pred[j])
        error = float(error) / len(y0) / len(y1)
        return error

    def convex_group(self, s):
        y0 = self.y[s == 0]
        y1 = self.y[s == 1]
        y0_pred = self.y_pred[s == 0]
        y1_pred = self.y_pred[s == 1]
        def convex_grp(y0, y1, y0_pred, y1_pred):
            if isinstance(y0, float):
                d = np.exp(-(y0 - y1) ** 2)
            else:
                d = 1 if y0 == y1 else 0
            return (y0_pred - y1_pred) * d

        error = 0
        for i in range(len(y0)):
            for j in range(len(y1)):
                error += convex_grp(y0[i], y1[j], y0_pred[i], y1_pred[j])
        error = float(error) / len(y0) / len(y1)
        return error ** 2

