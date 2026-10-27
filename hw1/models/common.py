import numpy as np


def grouped_sum(indices, values, size):
    if values.ndim == 1:
        return np.bincount(indices, weights=values, minlength=size).astype(np.float32)
    return np.column_stack([
        np.bincount(indices, weights=values[:, column], minlength=size)
        for column in range(values.shape[1])
    ]).astype(np.float32)


class Adam:
    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.steps = 0

    def step(self, gradients):
        self.steps += 1
        for name, gradient in gradients.items():
            first, second = self.first[name], self.second[name]
            first *= 0.9
            first += 0.1 * gradient
            second *= 0.999
            second += 0.001 * gradient ** 2
            estimate = first / (1 - 0.9 ** self.steps)
            variance = second / (1 - 0.999 ** self.steps)
            self.parameters[name] -= self.learning_rate * estimate / (np.sqrt(variance) + 1e-8)


def optimize(model, epochs, learning_rate):
    optimizer = Adam(model.parameters, learning_rate)
    for _ in range(epochs):
        loss, gradients = model.loss_and_gradients()
        if not np.isfinite(loss) or any(not np.isfinite(value).all() for value in gradients.values()):
            raise FloatingPointError("Non-finite training objective or gradient")
        optimizer.step(gradients)


def add_regularization(parameters, gradients, loss, regularization, count):
    penalty = 0.0
    for name, parameter in parameters.items():
        penalty += float(np.square(parameter).sum())
        gradients[name] = (gradients[name] + regularization * parameter) / count
    return (loss + 0.5 * regularization * penalty) / count, gradients
