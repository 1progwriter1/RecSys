import numpy as np
from scipy import sparse


class ItemKNN:
    def __init__(self, neighbors=50, shrinkage=10.0):
        self.neighbors = neighbors
        self.shrinkage = shrinkage

    def fit(self, data):
        self.interactions = data["positives"]
        cooccurrence = (self.interactions.T @ self.interactions).tocsr()
        norms = np.sqrt(np.asarray(self.interactions.sum(axis=0)).ravel())
        rows, columns, values = [], [], []
        for item in range(cooccurrence.shape[0]):
            start, end = cooccurrence.indptr[item:item + 2]
            indices = cooccurrence.indices[start:end]
            counts = cooccurrence.data[start:end]
            valid = indices != item
            indices, counts = indices[valid], counts[valid]
            similarity = counts / (norms[item] * norms[indices] + self.shrinkage)
            order = np.lexsort((indices, -similarity))[:self.neighbors]
            rows.extend([item] * len(order))
            columns.extend(indices[order])
            values.extend(similarity[order])
        self.similarity = sparse.csr_matrix(
            (np.asarray(values, dtype=np.float32), (rows, columns)), shape=cooccurrence.shape,
        )
        return self

    def score(self, users):
        return (self.interactions[users] @ self.similarity).toarray()
