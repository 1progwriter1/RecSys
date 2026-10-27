import numpy as np


class IALS:
    def __init__(self, factors=32, regularization=1.0, alpha=20.0, iterations=10, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def update_factors(self, interactions, fixed):
        base = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        result = np.zeros((interactions.shape[0], self.factors), dtype=np.float32)
        for row in range(interactions.shape[0]):
            start, end = interactions.indptr[row:row + 2]
            if start == end:
                continue
            vectors = fixed[interactions.indices[start:end]]
            system = base + self.alpha * (vectors.T @ vectors)
            target = (1 + self.alpha) * vectors.sum(axis=0)
            result[row] = np.linalg.solve(system, target)
        return result

    def fit(self, data):
        interactions = data["positives"].tocsr()
        transposed = interactions.T.tocsr()
        rng = np.random.default_rng(self.seed)
        self.items = rng.normal(0, 0.05, (interactions.shape[1], self.factors)).astype(np.float32)
        for _ in range(self.iterations):
            self.users = self.update_factors(interactions, self.items)
            self.items = self.update_factors(transposed, self.users)
        return self

    def score(self, users):
        return self.users[users] @ self.items.T
