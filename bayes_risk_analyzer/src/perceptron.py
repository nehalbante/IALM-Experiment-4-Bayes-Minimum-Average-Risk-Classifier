"""Rosenblatt perceptron (error-correction rule) for Part 10."""
from __future__ import annotations

import numpy as np


def perceptron_train(X, y, epochs=50, eta=1.0, seed=0):
    """Rosenblatt's error-correction rule on augmented inputs ``[1, x]``.

    Returns the learned weight vector (bias first) and the number of mistakes
    made in each epoch. The desired output is ``d = +1`` for malignant and
    ``-1`` for benign.
    """
    rng = np.random.default_rng(seed)
    Xa = np.c_[np.ones(len(X)), np.asarray(X, float)]
    d = np.where(np.asarray(y) == 1, 1, -1)
    w = np.zeros(Xa.shape[1])
    mistakes = []
    for _ in range(epochs):
        m = 0
        for i in rng.permutation(len(Xa)):
            if d[i] * (Xa[i] @ w) <= 0:
                w += eta * d[i] * Xa[i]
                m += 1
        mistakes.append(m)
    return w, mistakes


def perceptron_predict(X, w):
    """Predict labels from an augmented perceptron weight vector (bias first)."""
    Xa = np.c_[np.ones(len(X)), np.asarray(X, float)]
    return (Xa @ w > 0).astype(int)


def perceptron_weights(w):
    """Split an augmented vector into ``(bias, feature_weights)``."""
    return float(w[0]), np.asarray(w[1:], dtype=float)
