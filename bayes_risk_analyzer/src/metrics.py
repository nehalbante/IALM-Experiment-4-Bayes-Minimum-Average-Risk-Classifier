"""Performance metrics: confusion counts, rates, cost, and theoretical values.

Convention: malignant is the *positive* class (label 1), benign the negative
class (label 0). The lab vocabulary ("caught", "missed", "false alarms",
"cleared") maps onto TP/FN/FP/TN respectively.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def confusion_counts(y_true, y_pred) -> dict:
    """Return the four confusion counts plus the total number of samples."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    caught = int(np.sum((y_pred == 1) & (y_true == 1)))   # TP
    missed = int(np.sum((y_pred == 0) & (y_true == 1)))   # FN
    false_alarms = int(np.sum((y_pred == 1) & (y_true == 0)))  # FP
    cleared = int(np.sum((y_pred == 0) & (y_true == 0)))  # TN
    return {"caught": caught, "missed": missed, "false_alarms": false_alarms, "cleared": cleared, "n": len(y_true)}


def average_cost(counts: dict, costs) -> float:
    """Average cost per patient using the cost matrix ``(c11, c12, c21, c22)``."""
    c11, c12, c21, c22 = costs
    n = counts["n"]
    return (c11 * counts["caught"] + c21 * counts["missed"]
            + c12 * counts["false_alarms"] + c22 * counts["cleared"]) / n


def scorecard(y_true, y_pred, costs) -> dict:
    """Full scorecard: counts, miss/FA/error rates and average cost."""
    counts = confusion_counts(y_true, y_pred)
    caught, missed, fa, cleared = counts["caught"], counts["missed"], counts["false_alarms"], counts["cleared"]
    n = counts["n"]
    return {
        **counts,
        "miss_rate": missed / max(caught + missed, 1),
        "false_alarm_rate": fa / max(fa + cleared, 1),
        "error_rate": (missed + fa) / n,
        "avg_cost": average_cost(counts, costs),
    }


def classification_metrics(y_true, y_pred) -> dict:
    """Standard binary classification metrics (accuracy, precision, ...)."""
    counts = confusion_counts(y_true, y_pred)
    tp, fn, fp, tn = counts["caught"], counts["missed"], counts["false_alarms"], counts["cleared"]
    n = counts["n"]
    accuracy = (tp + tn) / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0       # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "specificity": specificity, "f1": f1,
        "miss_rate": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
        "false_alarm_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "error_rate": (fn + fp) / n,
    }


def theoretical_performance(mu1, mu2, Cinv, p1, p2, costs, xi) -> dict:
    """Exact miss/FA/error rates and average risk of the Gaussian model.

    Uses the normal-table formulas of the slides:
    ``ln Lambda | C1 ~ N(+d^2/2, d^2)``, ``ln Lambda | C2 ~ N(-d^2/2, d^2)``.
    """
    dmu = np.asarray(mu1, dtype=float) - np.asarray(mu2, dtype=float)
    Cinv = np.asarray(Cinv, dtype=float)
    d2 = float(dmu @ Cinv @ dmu)
    d = np.sqrt(d2)
    lx = np.log(xi)
    miss = float(norm.cdf((lx - d2 / 2) / d))
    fa = float(1 - norm.cdf((lx + d2 / 2) / d))
    c11, c12, c21, c22 = costs
    risk = c11 * p1 * (1 - miss) + c21 * p1 * miss + c12 * p2 * fa + c22 * p2 * (1 - fa)
    return {
        "d2": d2, "d": d, "miss_rate": miss, "false_alarm_rate": fa,
        "error_rate": p1 * miss + p2 * fa, "avg_cost": risk,
    }
