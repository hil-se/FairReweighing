import numpy as np
import pandas as pd


def get_histogram(pred, theta_indices):
    """
    Given a list of discrete predictions and Theta, compute a histogram
    pred: discrete prediction Series vector
    Theta: the discrete range of predictions as a Series vector
    """
    theta_counts = pd.Series(np.zeros(len(theta_indices)))
    for theta in theta_indices:
        theta_counts[theta_indices == theta] = len(pred[pred >= theta])
    return theta_counts


def weighted_pmf(pred, Theta):
    """
    Given a list of predictions and a set of weights, compute pmf.
    pl: a list of prediction vectors
    result_weights: a vector of weights over the classifiers
    """
    width = Theta[1] - Theta[0]
    theta_indices = pd.Series(Theta)
    # weighted_histograms = [(get_histogram(pred.iloc[:, i],
    #                                       theta_indices))
    #                        for i in range(pred.shape[1])]
    weighted_histograms = get_histogram(pd.Series(pred), theta_indices)
    # theta_counts = sum(weighted_histograms)
    pmf = weighted_histograms / len(pred)
    return pmf


def pmf2disp(pmf1, pmf2):
    """
    Take two empirical PMF vectors with the same support and calculate
    the K-S stats
    """
    # cdf_1 = pmf1.cumsum()
    # cdf_2 = pmf2.cumsum()
    diff = pmf1 - pmf2
    diff = abs(diff)
    return max(diff)


def extract_group_pred(total_pred, a):
    """
    total_pred: predictions over the data
    a: protected group attributes
    extract the relevant predictions for each protected group
    """
    groups = list(pd.Series.unique(a))
    pred_per_group = {}
    for g in groups:
        pred_per_group[g] = total_pred[a == g]
    return pred_per_group
