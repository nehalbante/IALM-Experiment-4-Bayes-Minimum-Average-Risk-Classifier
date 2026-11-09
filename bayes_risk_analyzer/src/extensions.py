"""The five extension experiments (Exercises 1-5) as reusable functions.

Each function computes everything from the fitted model and the test set —
no hardcoded experimental results.
"""
from __future__ import annotations

import numpy as np

from . import config
from .bayes import compute_bias, compute_threshold, compute_weights, predict
from .metrics import scorecard, theoretical_performance


def run_configuration(clf, Xt, yt, costs, p1, p2) -> dict:
    """Evaluate the minimum-risk rule for a given cost matrix and prior.

    Uses the classifier's learned geometry (weights and geometric bias) and
    only recomputes ``xi``, ``b`` and the resulting predictions/metrics.
    """
    c11, c12, c21, c22 = costs
    xi = compute_threshold(p1, p2, c11, c12, c21, c22)
    b = compute_bias(clf.b_geom_, xi)
    y_pred = predict(Xt, clf.w_, b)
    empirical = scorecard(yt, y_pred, costs)
    theoretical = theoretical_performance(clf.mu1_, clf.mu2_, clf.Cinv_, p1, p2, costs, xi)
    return {
        "costs": costs, "p1": p1, "p2": p2, "xi": xi, "b": b,
        "y_pred": y_pred, "empirical": empirical, "theoretical": theoretical,
    }


def extension_cheaper_misses(clf, Xt, yt, new_c21: float = 11.0) -> dict:
    """Exercise 1 — reduce the cost of missing a malignant tumor."""
    costs = (clf.c11, clf.c12, new_c21, clf.c22)
    return run_configuration(clf, Xt, yt, costs, clf.p1_, clf.p2_)


def extension_screening(clf, Xt, yt, p1: float = 0.05) -> dict:
    """Exercise 2 — screening population with a lower malignant prior."""
    p2 = 1.0 - p1
    costs = (clf.c11, clf.c12, clf.c21, clf.c22)
    result = run_configuration(clf, Xt, yt, costs, p1, p2)
    # Expected risk under the NEW population (the old test set is 20% malignant).
    result["new_population_risk"] = result["theoretical"]["avg_cost"]
    return result


def extension_haykin(clf, Xt, yt) -> dict:
    """Exercise 3 — Haykin's assumptions (equal priors, 0-1 costs)."""
    costs = (0.0, 1.0, 1.0, 0.0)
    result = run_configuration(clf, Xt, yt, costs, 0.5, 0.5)
    result["b_geom"] = clf.b_geom_
    result["reduces_to_geometric"] = bool(np.isclose(result["b"], clf.b_geom_))
    return result


def extension_feature_importance(clf) -> dict:
    """Exercise 4 — increase the malignant size mean so its difference doubles.

    Original difference is 0.2 (0.6 vs 0.4); increasing it to 0.4 keeps the
    covariance unchanged, so the size weight doubles from 5 to 10.
    """
    mu1_new = np.array(clf.mu1_, dtype=float)
    mu1_new[0] = clf.mu2_[0] + 0.4   # set size mean difference to exactly 0.4
    w_new = compute_weights(mu1_new, clf.mu2_, clf.Cinv_)
    return {
        "mu1_original": np.array(clf.mu1_, dtype=float),
        "mu2": np.array(clf.mu2_, dtype=float),
        "mu1_modified": mu1_new,
        "mean_diff_original": np.array(clf.mu1_, dtype=float) - np.array(clf.mu2_, dtype=float),
        "mean_diff_modified": mu1_new - np.array(clf.mu2_, dtype=float),
        "w_original": np.array(clf.w_, dtype=float),
        "w_new": w_new,
        "C": np.array(clf.C_, dtype=float),
    }


def extension_correlated(clf) -> dict:
    """Exercise 5 — introduce correlation 0.5 between size and growth."""
    C = np.array([
        [0.04, 0.0, 0.02],
        [0.0, 0.04, 0.0],
        [0.02, 0.0, 0.04],
    ], dtype=float)
    Cinv = np.linalg.inv(C)
    dmu = np.array(clf.mu1_, dtype=float) - np.array(clf.mu2_, dtype=float)
    w_new = Cinv @ dmu
    return {
        "C_original": np.array(clf.C_, dtype=float),
        "C_modified": C,
        "Cinv_modified": Cinv,
        "mean_diff": dmu,
        "w_original": np.array(clf.w_, dtype=float),
        "w_new": w_new,
    }
