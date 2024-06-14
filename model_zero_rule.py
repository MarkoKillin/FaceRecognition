import numpy as np


class ZeroRuleClassifier:
    def __init__(self, X, y):
        self.most_frequent = np.bincount(y).argmax()

    def predict(self, X):
        return np.full(X.shape[0], self.most_frequent)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
