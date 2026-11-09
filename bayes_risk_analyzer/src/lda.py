"""Linear Discriminant Analysis (LDA) helpers for Part 9 comparison."""
from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fit_lda(X, y) -> LinearDiscriminantAnalysis:
    """Fit sklearn's LDA (MAP rule for Gaussian classes with shared covariance)."""
    return LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)


def lda_params(lda) -> dict:
    """Extract the LDA weight vector and intercept."""
    return {
        "w": np.asarray(lda.coef_[0], dtype=float),
        "b": float(lda.intercept_[0]),
    }
