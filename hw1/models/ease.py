import numpy as np
from scipy.linalg.lapack import spotrf, spotri


class EASE:
    def __init__(self, regularization=100.0):
        self.regularization = regularization

    def fit(self, data):
        self.interactions = data["positives"]
        precision = (self.interactions.T @ self.interactions).toarray(order="F").astype(np.float32, copy=False)
        diagonal = np.diag_indices_from(precision)
        precision[diagonal] += self.regularization
        precision, info = spotrf(precision, lower=0, overwrite_a=True, clean=False)
        if info != 0:
            raise np.linalg.LinAlgError(f"Cholesky factorization failed: {info}")
        precision, info = spotri(precision, lower=0, overwrite_c=True)
        if info != 0:
            raise np.linalg.LinAlgError(f"Precision matrix inversion failed: {info}")
        for row in range(precision.shape[0]):
            precision[row, :row] = precision[:row, row]
        precision /= -precision[diagonal].copy()[None, :]
        precision[diagonal] = 0
        self.weights = np.ascontiguousarray(precision)
        return self

    def score(self, users):
        return self.interactions[users] @ self.weights
