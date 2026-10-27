import numpy as np
from scipy import sparse
from scipy.special import expit

from .common import Adam, grouped_sum


class FM:
    def __init__(self, factors=16, regularization=1e-5, learning_rate=0.01, epochs=20,
                 batch_size=8192, use_metadata=True, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_metadata = use_metadata
        self.seed = seed

    def initialize(self, data):
        self.seen = data["seen"].astype(bool).tocsr()
        self.seen.eliminate_zeros()
        n_users, n_items = self.seen.shape
        users, items = data["positives"].nonzero()
        saturated = np.diff(self.seen.indptr) == n_items
        eligible = ~saturated[users]
        self.user_ids = users[eligible].astype(np.int32)
        self.item_ids = items[eligible].astype(np.int32)
        if not len(self.user_ids):
            raise ValueError("FM needs positive pairs with at least one unseen candidate")
        self.metadata = data["item_features"].astype(np.float32).copy()
        if not self.use_metadata:
            self.metadata = sparse.csr_matrix((n_items, 0), dtype=np.float32)
        norm = np.sqrt(np.asarray(self.metadata.power(2).sum(axis=1)).ravel())
        self.metadata = (sparse.diags(1 / np.maximum(norm, 1)) @ self.metadata).tocsr()
        self.metadata_squared = self.metadata.power(2)
        rng = np.random.default_rng(self.seed)
        self.parameters = {
            "user": rng.normal(0, 0.05, (n_users, self.factors)).astype(np.float32),
            "item": rng.normal(0, 0.05, (n_items, self.factors)).astype(np.float32),
            "meta": rng.normal(0, 0.05, (self.metadata.shape[1], self.factors)).astype(np.float32),
            "user_bias": np.zeros(n_users, dtype=np.float32),
            "item_bias": np.zeros(n_items, dtype=np.float32),
            "meta_bias": np.zeros(self.metadata.shape[1], dtype=np.float32),
        }

    def sample_negatives(self, users, rng):
        n_items = self.seen.shape[1]
        if np.any(np.diff(self.seen.indptr)[users] == n_items):
            raise ValueError("A sampled user has seen every candidate")
        negatives = rng.integers(n_items, size=len(users), dtype=np.int32)
        collisions = np.asarray(self.seen[users, negatives]).ravel()
        while collisions.any():
            negatives[collisions] = rng.integers(n_items, size=int(collisions.sum()), dtype=np.int32)
            collisions = np.asarray(self.seen[users, negatives]).ravel()
        return negatives

    def item_terms(self):
        p = self.parameters
        vectors = p["item"] + self.metadata @ p["meta"]
        squares = p["item"] ** 2 + self.metadata_squared @ (p["meta"] ** 2)
        internal = 0.5 * np.sum(vectors ** 2 - squares, axis=1)
        biases = p["item_bias"] + self.metadata @ p["meta_bias"] + internal
        return vectors, biases

    def loss_and_gradients(self, triples):
        p = self.parameters
        u, i, j = triples
        n_users, n_items = len(p["user"]), len(p["item"])
        vectors, biases = self.item_terms()
        delta = biases[i] - biases[j] + np.sum(p["user"][u] * (vectors[i] - vectors[j]), axis=1)
        error = -expit(-delta)
        item_error = grouped_sum(i, error, n_items) - grouped_sum(j, error, n_items)
        total = grouped_sum(i, error[:, None] * (p["user"][u] + vectors[i]), n_items)
        total -= grouped_sum(j, error[:, None] * (p["user"][u] + vectors[j]), n_items)
        gradients = {
            "user": grouped_sum(u, error[:, None] * (vectors[i] - vectors[j]), n_users),
            "item": total - item_error[:, None] * p["item"],
            "meta": self.metadata.T @ total - (self.metadata_squared.T @ item_error)[:, None] * p["meta"],
            "user_bias": np.zeros_like(p["user_bias"]),
            "item_bias": item_error,
            "meta_bias": self.metadata.T @ item_error,
        }
        loss = float(np.logaddexp(0, -delta).mean(dtype=np.float64))
        for name, parameter in p.items():
            loss += 0.5 * self.regularization * float(np.square(parameter).sum(dtype=np.float64))
            gradients[name] = gradients[name] / len(u) + self.regularization * parameter
        return loss, gradients

    def fit(self, data):
        self.initialize(data)
        optimizer = Adam(self.parameters, self.learning_rate)
        rng = np.random.default_rng(self.seed + 1)
        for _ in range(self.epochs):
            order = rng.permutation(len(self.user_ids))
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                users, items = self.user_ids[indices], self.item_ids[indices]
                negatives = self.sample_negatives(users, rng)
                loss, gradients = self.loss_and_gradients((users, items, negatives))
                if not np.isfinite(loss) or any(not np.isfinite(value).all() for value in gradients.values()):
                    raise FloatingPointError("Non-finite BPR objective or gradient")
                optimizer.step(gradients)
        self.item_vectors, self.item_biases = self.item_terms()
        return self

    def score(self, users):
        return (self.parameters["user_bias"][users, None] + self.item_biases[None, :]
                + self.parameters["user"][users] @ self.item_vectors.T)
