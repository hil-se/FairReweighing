import numpy as np
import pandas as pd


def get_histogram(pred, theta_indices):
    theta_counts = pd.Series(np.zeros(len(theta_indices)))
    for theta in theta_indices:
        theta_counts[theta_indices == theta] = len(pred[pred >= theta])
    return theta_counts


def weighted_pmf(pred, Theta):
    theta_indices = pd.Series(Theta)

    weighted_histograms = get_histogram(pd.Series(pred), theta_indices)
    pmf = weighted_histograms / len(pred)
    return pmf


def pmf2disp(pmf1, pmf2):
    diff = pmf1 - pmf2
    diff = abs(diff)
    return max(diff)


def extract_group_pred(total_pred, a):
    groups = list(pd.Series.unique(a))
    pred_per_group = {}
    for g in groups:
        pred_per_group[g] = total_pred[a == g]
    return pred_per_group
