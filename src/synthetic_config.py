import numpy as np
from scipy.stats import norm


P_A1 = 0.7
N_SYNTHETIC = 10000

MA = 1.65
MB = 0.10
MC = 0.15
MD = 0.05
ME = 0.35
MF = 0.10
MG = 0.05
MH = 0.05


def synthetic_height_mean(sex):
    return MA + MB * sex


def synthetic_height_sd(sex):
    return MC + MD * sex


def synthetic_power_mean(sex):
    return ME + MF * sex


def synthetic_power_sd(sex):
    return MG + MH * sex


def synthetic_y_mean(sex):
    return synthetic_height_mean(sex) + synthetic_power_mean(sex)


def synthetic_y_sd(sex):
    return np.sqrt(synthetic_height_sd(sex) ** 2 + synthetic_power_sd(sex) ** 2)


def synthetic_ground_truth_weight(A, y, eps=1e-12):
    A = np.asarray(A, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    y_a0 = norm.pdf(y, loc=synthetic_y_mean(0), scale=synthetic_y_sd(0))
    y_a1 = norm.pdf(y, loc=synthetic_y_mean(1), scale=synthetic_y_sd(1))
    y_marginal = (1.0 - P_A1) * y_a0 + P_A1 * y_a1
    y_given_a = np.where(A == 1, y_a1, y_a0)
    return normalize_weight(y_marginal / np.maximum(y_given_a, eps))


def normalize_weight(weight):
    weight = np.asarray(weight, dtype=float).reshape(-1)
    total = weight.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(weight))
    return len(weight) * weight / total
