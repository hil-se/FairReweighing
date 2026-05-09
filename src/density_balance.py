import numpy as np
from scipy.stats import bernoulli, norm

from density_est import DensityKernel, DensityNeighbor, Reweighing


EPS = 1e-12


class DensityBalance:
    def __init__(
        self,
        model="Neighbor",
        radius=0.5,
        bandwidth=0.2,
        kernel="gaussian",
        distance="euclidean",
        n_bins=5,
    ):
        model_key = _normalize_model(model)
        models = {
            "Neighbor": DensityNeighbor(radius=radius, distance=distance),
            "Kernel": DensityKernel(kernel=kernel, bandwidth=bandwidth),
            "Reweighing": Reweighing(),
        }
        self.model_name = model_key
        self.model = models[model_key]
        self.n_bins = n_bins

    def weight(self, A, y, treatment="FairReweighing"):
        treatment_key = _normalize_treatment(treatment)
        A = _as_2d(A)
        y = _as_2d(y)

        if treatment_key == "none":
            return None
        if treatment_key == "groundtruth":
            return self._synthetic_ground_truth_weight(A, y)
        if treatment_key == "fairbalancevariant":
            return _normalize_weight(1.0 / np.maximum(self.model.density(np.concatenate((A, y), axis=1)), EPS))
        if treatment_key == "fairbalance":
            joint = np.concatenate((A, y), axis=1)
            return _normalize_weight(self.model.density(A) / np.maximum(self.model.density(joint), EPS))
        if treatment_key == "groupbalance":
            joint = np.concatenate((A, y), axis=1)
            return _normalize_weight(self.model.density(y) / np.maximum(self.model.density(joint), EPS))

        return _ratio_weight(self.model, A, y)

    def _synthetic_ground_truth_weight(self, A, y):
        A = A.astype(float)
        y = y.astype(float)
        wA_true = bernoulli.pmf(A, 0.7).flatten()
        wy_true = np.zeros(len(y))
        w_true = np.zeros(len(y))

        for i in range(len(y)):
            yi = y[i][0]
            wy_true[i] = norm.pdf(yi, 2.35, 0.21) * 0.7 + norm.pdf(yi, 2.15, 0.14) * 0.3
            if A[i][0] == 1:
                w_true[i] = norm.pdf(yi, 2.35, 0.21) * 0.7
            else:
                w_true[i] = norm.pdf(yi, 2.15, 0.14) * 0.3

        return _normalize_weight(wA_true * wy_true / np.maximum(w_true, EPS))


def _ratio_weight(model, A, y):
    joint = np.concatenate((A, y), axis=1)
    w_joint = np.maximum(model.density(joint), EPS)
    w_a = model.density(A)
    w_y = model.density(y)
    return _normalize_weight(w_a * w_y / w_joint)


def _normalize_weight(weight):
    weight = np.asarray(weight, dtype=float).flatten()
    weight = np.nan_to_num(weight, nan=1.0, posinf=np.nanmax(weight[np.isfinite(weight)]) if np.isfinite(weight).any() else 1.0)
    total = weight.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(len(weight))
    return len(weight) * weight / total


def _as_2d(values):
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _normalize_model(model):
    aliases = {
        None: "Neighbor",
        "neighbor": "Neighbor",
        "neighbors": "Neighbor",
        "kernel": "Kernel",
        "kde": "Kernel",
        "reweighing": "Reweighing",
        "discrete": "Reweighing",
    }
    key = aliases.get(str(model).lower() if model is not None else None)
    if key is None:
        raise ValueError(f"Unknown density model: {model}")
    return key


def _normalize_treatment(treatment):
    aliases = {
        "none": "none",
        "fairreweighing": "fair-reweighing",
        "fair-reweighing": "fair-reweighing",
        "reweighing": "fair-reweighing",
        "groundtruth": "groundtruth",
        "fairbalancevariant": "fairbalancevariant",
        "fairbalance": "fairbalance",
        "groupbalance": "groupbalance",
    }
    key = aliases.get(str(treatment).replace("_", "-").lower())
    if key is None:
        raise ValueError(f"Unknown treatment: {treatment}")
    return key
