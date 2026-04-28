import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler


class DensityKernel:
    def __init__(self, kernel="gaussian", bandwidth=0.2):
        self.kernel = kernel
        self.bandwidth = bandwidth

    def density(self, X):
        X_z = _standardize(X)
        kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        kde.fit(X_z)
        return np.exp(kde.score_samples(X_z))


class DensityNeighbor:
    def __init__(self, radius=0.5, distance="euclidean"):
        self.radius = radius
        self.distance = distance

    def density(self, X):
        X_z = _standardize(X)
        dists = pairwise_distances(X_z, metric=self.distance)
        return np.sum(dists < self.radius, axis=1).astype(float)


class Reweighing:
    def density(self, X):
        rows = pd.DataFrame(np.asarray(X)).astype(str).agg("|".join, axis=1)
        counts = rows.map(rows.value_counts())
        return counts.to_numpy(dtype=float) / len(rows)


def _standardize(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return StandardScaler().fit_transform(X)
