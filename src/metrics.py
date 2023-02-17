import sklearn.metrics


class Metrics:
    def __init__(self, y, y_pred):
        # y and y_pred are 1-d arrays of true values and predicted values
        self.y = y
        self.y_pred = y_pred

    def mse(self):
        return sklearn.metrics.mean_squared_error(self.y, self.y_pred)

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
