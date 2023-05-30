import numpy as np

from density_est import DensityKernel, DensityNeighbor


class DensityBalance():
    def __init__(self, model='Neighbor'):
        models = {'Neighbor': DensityNeighbor(), "Kernel": DensityKernel()}
        self.model = models[model]

    def weight(self, A, y, treatment="FairBalance"):
        X = np.concatenate((A, y), axis=1)
        w = self.model.density(X)

        if treatment == "FairBalanceVariant":
            weight = 1 / w
        else:
            wA = self.model.density(A)
            if treatment == "FairBalance":
                weight = wA / w
            else:
                wy = self.model.density(y)
                if treatment == "GroupBalance":
                    weight = wy / w
                else:
                    weight = wA * wy / w

        weight = len(weight) * weight / sum(weight)
        return weight
