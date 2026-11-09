"""Bayes minimum-risk classifier and reusable mathematical functions.

This is the single source of truth for the classifier math. It ports the
original ``bayes_min_risk.py`` (Haykin, Sec. 1.4) and exposes both a class
and a set of pure functions so the UI never duplicates a calculation.

Conventions
-----------
* ``C1`` = malignant (label 1), ``C2`` = benign (label 0).
* ``c[i][j]`` = cost of deciding class ``i`` when class ``j`` is true.
* Decision rule: ``g(x) = x @ w + b``;  ``g(x) > 0  -> malignant``.

    w = C^-1 (mu1 - mu2)
    b = 1/2 (mu2' C^-1 mu2 - mu1' C^-1 mu1) - ln xi
    xi = p2 (c12 - c22) / (p1 (c21 - c11))
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from . import config


def validate_priors(p1: float, p2: float) -> None:
    """Raise ``ValueError`` if the priors are not valid probabilities."""
    if not (0 < p1 < 1) or not (0 < p2 < 1):
        raise ValueError("Priors must be strictly between 0 and 1.")
    if not np.isclose(p1 + p2, 1.0, atol=1e-9):
        raise ValueError("Priors must sum to 1.")


def validate_costs(c11: float, c12: float, c21: float, c22: float) -> None:
    """Raise ``ValueError`` if the cost matrix makes xi undefined or non-positive."""
    if (c21 - c11) <= 0:
        raise ValueError(
            "The regret of a miss (c21 - c11) must be positive, otherwise the "
            "threshold xi is undefined or non-positive."
        )
    if (c12 - c22) <= 0:
        raise ValueError(
            "The regret of a false alarm (c12 - c22) must be positive, otherwise "
            "the threshold xi is undefined or non-positive."
        )


def compute_threshold(p1: float, p2: float, c11: float, c12: float, c21: float, c22: float) -> float:
    """Minimum-risk likelihood-ratio threshold ``xi = p2 (c12 - c22) / (p1 (c21 - c11))``."""
    validate_priors(p1, p2)
    validate_costs(c11, c12, c21, c22)
    return p2 * (c12 - c22) / (p1 * (c21 - c11))


def compute_weights(mu1, mu2, Cinv):
    """``w = C^-1 (mu1 - mu2)``."""
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    Cinv = np.asarray(Cinv, dtype=float)
    return Cinv @ (mu1 - mu2)


def compute_b_geom(mu1, mu2, Cinv) -> float:
    """Geometric bias ``b_geom = 1/2 (mu2' C^-1 mu2 - mu1' C^-1 mu1)``."""
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    Cinv = np.asarray(Cinv, dtype=float)
    return float(0.5 * (mu2 @ Cinv @ mu2 - mu1 @ Cinv @ mu1))


def compute_bias(b_geom: float, xi: float) -> float:
    """Minimum-risk bias ``b = b_geom - ln xi``."""
    return b_geom - np.log(xi)


def decision_function(X, w, b):
    """Perceptron output ``g(x) = x @ w + b``."""
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b


def predict(X, w, b):
    """``g(x) > 0 -> malignant (1)``, otherwise benign (0)."""
    return (decision_function(X, w, b) > 0).astype(int)


def log_likelihood_ratio(X, w, b_geom):
    """``ln Lambda(x) = x @ w + b_geom`` (no priors, no costs)."""
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b_geom


def posterior(X, w, b_geom, p1: float, p2: float):
    """``P(C1 | x) = sigmoid(ln Lambda + ln(p1 / p2))``."""
    z = log_likelihood_ratio(X, w, b_geom) + np.log(p1 / p2)
    return 1.0 / (1.0 + np.exp(-z))


class BayesMinRiskClassifier:
    """Minimum-risk Bayes classifier for two Gaussian classes sharing a covariance.

    Port of the original lab class. ``fit`` estimates priors/means/covariance by
    maximum likelihood; ``predict`` then applies the minimum-risk rule.
    """

    def __init__(self, c11=8.0, c12=3.0, c21=40.0, c22=1.0, priors=None):
        self.c11, self.c12, self.c21, self.c22 = float(c11), float(c12), float(c21), float(c22)
        self.priors = priors

    # ---------------------------------------------------------------- learning
    def fit(self, X, y):
        """Maximum-likelihood estimates of priors, means and shared covariance."""
        X, y = np.asarray(X, float), np.asarray(y)
        X1, X2 = X[y == 1], X[y == 0]
        n = len(X)
        self.p1_, self.p2_ = (len(X1) / n, len(X2) / n) if self.priors is None else self.priors
        self.mu1_, self.mu2_ = X1.mean(axis=0), X2.mean(axis=0)
        D = np.vstack([X1 - self.mu1_, X2 - self.mu2_])
        self.C_ = D.T @ D / n
        self.Cinv_ = np.linalg.inv(self.C_)
        self._weights()
        return self

    def _weights(self):
        self.w_ = self.Cinv_ @ (self.mu1_ - self.mu2_)
        self.b_geom_ = float(0.5 * (self.mu2_ @ self.Cinv_ @ self.mu2_ - self.mu1_ @ self.Cinv_ @ self.mu1_))
        self.xi_ = self.threshold()
        self.b_ = self.b_geom_ - np.log(self.xi_)

    def threshold(self, rule: str = "min_risk") -> float:
        """Likelihood-ratio threshold ``xi`` for the three slide rules."""
        if rule == "ml":
            return 1.0
        if rule == "map":
            return self.p2_ / self.p1_
        return self.p2_ * (self.c12 - self.c22) / (self.p1_ * (self.c21 - self.c11))

    # ---------------------------------------------------------------- deciding
    def log_likelihood_ratio(self, X):
        return np.asarray(X, float) @ self.w_ + self.b_geom_

    def decision_function(self, X, rule: str = "min_risk"):
        return self.log_likelihood_ratio(X) - np.log(self.threshold(rule))

    def predict(self, X, rule: str = "min_risk"):
        return (self.decision_function(X, rule) > 0).astype(int)

    def posterior(self, X):
        z = self.log_likelihood_ratio(X) + np.log(self.p1_ / self.p2_)
        return 1.0 / (1.0 + np.exp(-z))

    def conditional_risk(self, X):
        p1x = self.posterior(X)
        p2x = 1.0 - p1x
        return self.c11 * p1x + self.c12 * p2x, self.c21 * p1x + self.c22 * p2x

    # ---------------------------------------------------------------- evaluating
    def cost_matrix(self):
        return np.array([[self.c11, self.c12], [self.c21, self.c22]])

    def evaluate(self, X, y, rule: str = "min_risk", y_pred=None):
        """Counts, rates, error rate and average cost per patient."""
        y = np.asarray(y)
        yp = self.predict(X, rule) if y_pred is None else np.asarray(y_pred)
        caught = int(np.sum((yp == 1) & (y == 1)))
        missed = int(np.sum((yp == 0) & (y == 1)))
        false_alarm = int(np.sum((yp == 1) & (y == 0)))
        cleared = int(np.sum((yp == 0) & (y == 0)))
        n = len(y)
        cost = (self.c11 * caught + self.c21 * missed + self.c12 * false_alarm + self.c22 * cleared) / n
        return {
            "caught": caught, "missed": missed, "false_alarms": false_alarm, "cleared": cleared,
            "miss_rate": missed / max(caught + missed, 1),
            "false_alarm_rate": false_alarm / max(false_alarm + cleared, 1),
            "error_rate": (missed + false_alarm) / n,
            "avg_cost": cost,
        }

    def theoretical(self, rule: str = "min_risk"):
        """Exact rates and risk for the fitted Gaussian model (normal-table formulas)."""
        dmu = self.mu1_ - self.mu2_
        d2 = float(dmu @ self.Cinv_ @ dmu)
        d = np.sqrt(d2)
        lx = np.log(self.threshold(rule))
        miss = norm.cdf((lx - d2 / 2) / d)
        fa = 1 - norm.cdf((lx + d2 / 2) / d)
        p1, p2 = self.p1_, self.p2_
        risk = self.c11 * p1 * (1 - miss) + self.c21 * p1 * miss + self.c12 * p2 * fa + self.c22 * p2 * (1 - fa)
        return {
            "d2": d2, "miss_rate": miss, "false_alarm_rate": fa,
            "error_rate": p1 * miss + p2 * fa, "avg_cost": risk,
        }


def fit_default_classifier(train_X, train_y):
    """Fit the classifier with the lab's default costs (used across the app)."""
    return BayesMinRiskClassifier(
        c11=config.DEFAULT_COSTS["c11"],
        c12=config.DEFAULT_COSTS["c12"],
        c21=config.DEFAULT_COSTS["c21"],
        c22=config.DEFAULT_COSTS["c22"],
    ).fit(train_X, train_y)


def conditional_risks(X, w, b_geom, p1: float, p2: float, c11: float, c12: float, c21: float, c22: float):
    """Expected cost of each decision: ``(R(say malignant|x), R(say benign|x))``."""
    p1x = posterior(X, w, b_geom, p1, p2)
    p2x = 1.0 - p1x
    return c11 * p1x + c12 * p2x, c21 * p1x + c22 * p2x
