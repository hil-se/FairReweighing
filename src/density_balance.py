import numpy as np
from scipy.stats import bernoulli, norm

from density_est import DensityKernel, DensityNeighbor, Reweighing


class DensityBalance():
    def __init__(self, model='Neighbor'):
        models = {'Neighbor': DensityNeighbor(), "Kernel": DensityKernel(), 'Reweighing': Reweighing()}
        self.model = models[model]

    def weight(self, A, y, treatment="FairBalance"):
        X = np.concatenate((A, y), axis=1)

        w = self.model.density(X)
        wA = self.model.density(A)
        wy = self.model.density(y)

        wA_true = bernoulli.pmf(A, 0.7).flatten()
        wy_true = y.copy().flatten()
        w_true = y.copy().flatten()

        for i in range(A.size):
            wy_true[i] = norm.pdf(y[i][0], 2.35, 0.21) * 0.7 + norm.pdf(y[i][0], 2.15, 0.14) * 0.3

        for i in range(A.size):
            if A[i] == 1:
                w_true[i] = norm.pdf(y[i][0], 2.35, 0.21) * 0.7
            else:
                w_true[i] = norm.pdf(y[i][0], 2.15, 0.14) * 0.3

        if treatment == "FairBalanceVariant":
            weight = 1 / w
        elif treatment == "FairBalance":
            weight = wA / w
        elif treatment == "GroupBalance":
            weight = wy / w
        elif treatment == "Reweighing":
            weight = wA * wy / w
        else:
            weight = wA_true * wy_true / w_true

        weight = (len(weight) * weight / sum(weight)).flatten()
        return weight
