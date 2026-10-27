import numpy as np

from .common import add_regularization, grouped_sum, optimize


class TimeSVDPlusPlus:
    def __init__(self, factors=16, regularization=20.0, learning_rate=0.01, epochs=50,
                 bin_days=90, beta=0.4, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.bin_days = bin_days
        self.beta = beta
        self.seed = seed

    def initialize(self, data):
        history = data["history"]
        self.user_ids = history["user_idx"].to_numpy(dtype=np.int32)
        self.item_ids = history["item_idx"].to_numpy(dtype=np.int32)
        self.targets = history["rating"].to_numpy(dtype=np.float32)
        self.mean = float(self.targets.mean())
        self.seen = data["seen"]
        n_users, n_items = self.seen.shape
        days = history["timestamp"].dt.as_unit("ns").astype("int64").to_numpy() // 86_400_000_000_000
        self.first_day = int(days.min())
        self.mean_days = np.bincount(self.user_ids, weights=days, minlength=n_users) / np.bincount(self.user_ids, minlength=n_users)
        self.deviations = self.deviation(days, self.user_ids)
        self.bins = ((days - self.first_day) // self.bin_days).astype(np.int32)
        self.n_bins = int(self.bins.max()) + 1
        day_keys = np.column_stack([self.user_ids, days])
        _, self.day_ids = np.unique(day_keys, axis=0, return_inverse=True)
        n_days = int(self.day_ids.max()) + 1
        self.norm = 1 / np.sqrt(np.diff(self.seen.indptr))
        self.query_day = int(data["evaluation"]["timestamp"].min().value // 86_400_000_000_000)
        rng = np.random.default_rng(self.seed)
        self.parameters = {
            "user": rng.normal(0, 0.05, (n_users, self.factors)).astype(np.float32),
            "item": rng.normal(0, 0.05, (n_items, self.factors)).astype(np.float32),
            "implicit": rng.normal(0, 0.02, (n_items, self.factors)).astype(np.float32),
            "drift": np.zeros((n_users, self.factors), dtype=np.float32),
            "day_factor": np.zeros((n_days, self.factors), dtype=np.float32),
            "user_bias": np.zeros(n_users, dtype=np.float32),
            "item_bias": np.zeros(n_items, dtype=np.float32),
            "user_drift": np.zeros(n_users, dtype=np.float32),
            "day_bias": np.zeros(n_days, dtype=np.float32),
            "item_bin": np.zeros((n_items, self.n_bins), dtype=np.float32),
        }

    def deviation(self, days, users):
        delta = (days - self.mean_days[users]) / 365.25
        return (np.sign(delta) * np.abs(delta) ** self.beta).astype(np.float32)

    def implicit_factors(self):
        return (self.seen @ self.parameters["implicit"]) * self.norm[:, None]

    def loss_and_gradients(self):
        p = self.parameters
        u, i, d, t = self.user_ids, self.item_ids, self.day_ids, self.bins
        n_users, n_items = self.seen.shape
        vectors = p["user"][u] + p["drift"][u] * self.deviations[:, None]
        vectors += p["day_factor"][d] + self.implicit_factors()[u]
        predictions = (self.mean + p["user_bias"][u] + p["user_drift"][u] * self.deviations
                       + p["day_bias"][d] + p["item_bias"][i] + p["item_bin"][i, t]
                       + np.sum(p["item"][i] * vectors, axis=1))
        error = predictions - self.targets
        derivative = error[:, None] * p["item"][i]
        total = grouped_sum(u, derivative, n_users)
        gradients = {
            "user": total,
            "item": grouped_sum(i, error[:, None] * vectors, n_items),
            "implicit": self.seen.T @ (total * self.norm[:, None]),
            "drift": grouped_sum(u, derivative * self.deviations[:, None], n_users),
            "day_factor": grouped_sum(d, derivative, len(p["day_factor"])),
            "user_bias": grouped_sum(u, error, n_users),
            "item_bias": grouped_sum(i, error, n_items),
            "user_drift": grouped_sum(u, error * self.deviations, n_users),
            "day_bias": grouped_sum(d, error, len(p["day_bias"])),
            "item_bin": grouped_sum(i * self.n_bins + t, error, n_items * self.n_bins).reshape(n_items, self.n_bins),
        }
        return add_regularization(p, gradients, 0.5 * float(error @ error), self.regularization, len(u))

    def fit(self, data):
        self.initialize(data)
        optimize(self, self.epochs, self.learning_rate)
        self.implicit_vectors = self.implicit_factors()
        return self

    def score(self, users):
        p = self.parameters
        dev = self.deviation(self.query_day, users)
        vectors = p["user"][users] + p["drift"][users] * dev[:, None] + self.implicit_vectors[users]
        time_bin = min((self.query_day - self.first_day) // self.bin_days, self.n_bins - 1)
        user_bias = p["user_bias"][users] + p["user_drift"][users] * dev
        return (self.mean + user_bias[:, None] + p["item_bias"][None, :]
                + p["item_bin"][:, time_bin][None, :] + vectors @ p["item"].T)
