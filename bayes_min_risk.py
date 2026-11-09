"""
Bayes minimum-risk classifier for two Gaussian classes with a shared covariance
(Haykin, Neural Networks and Learning Machines, Sec. 1.4), written as a perceptron.

Notation follows the slides:
    C1 = malignant (label 1), C2 = benign (label 0)
    p1, p2         priors
    mu1, mu2, C    class means and shared covariance
    c[i][j]        cost of deciding class i when class j is true   (c11, c12, c21, c22)

Decision rule (likelihood ratio test):
    Lambda(x) = p(x|C1) / p(x|C2)  >  xi = p2 (c12 - c22) / (p1 (c21 - c11))   ->  C1

With Gaussian classes and shared C, log Lambda(x) is linear in x, so the rule is a
single perceptron  y = w.x + b,  decide C1 if y > 0, with
    w = C^-1 (mu1 - mu2)
    b = 1/2 (mu2' C^-1 mu2 - mu1' C^-1 mu1) - ln xi
"""
import numpy as np
from scipy.stats import norm


class BayesMinRiskClassifier:
    def __init__(self, c11=8.0, c12=3.0, c21=40.0, c22=1.0, priors=None):
        """priors=None -> estimate p1, p2 from class counts in fit()."""
        self.c11, self.c12, self.c21, self.c22 = c11, c12, c21, c22
        self.priors = priors

    # ---------------------------------------------------------------- learning
    def fit(self, X, y):
        """Maximum-likelihood estimates of the priors, means and the shared covariance."""
        X, y = np.asarray(X, float), np.asarray(y)
        X1, X2 = X[y == 1], X[y == 0]
        n = len(X)
        self.p1_, self.p2_ = (len(X1) / n, len(X2) / n) if self.priors is None else self.priors
        self.mu1_, self.mu2_ = X1.mean(axis=0), X2.mean(axis=0)
        # pooled ML covariance: every point measured from its own class mean, divided by n
        D = np.vstack([X1 - self.mu1_, X2 - self.mu2_])
        self.C_ = D.T @ D / n
        self.Cinv_ = np.linalg.inv(self.C_)
        self._weights()
        return self

    def _weights(self):
        self.w_ = self.Cinv_ @ (self.mu1_ - self.mu2_)
        self.b_geom_ = 0.5 * (self.mu2_ @ self.Cinv_ @ self.mu2_ - self.mu1_ @ self.Cinv_ @ self.mu1_)
        self.xi_ = self.threshold()
        self.b_ = self.b_geom_ - np.log(self.xi_)

    def threshold(self, rule="min_risk"):
        """xi for the three decision rules of the slides."""
        if rule == "ml":                       # equal priors, 0-1 costs
            return 1.0
        if rule == "map":                      # real priors, 0-1 costs
            return self.p2_ / self.p1_
        return self.p2_ * (self.c12 - self.c22) / (self.p1_ * (self.c21 - self.c11))

    # ---------------------------------------------------------------- deciding
    def log_likelihood_ratio(self, X):
        """ln Lambda(x) = w.x + b_geom  (no priors, no costs)."""
        return np.asarray(X, float) @ self.w_ + self.b_geom_

    def decision_function(self, X, rule="min_risk"):
        """Perceptron output y = ln Lambda(x) - ln xi; positive -> malignant."""
        return self.log_likelihood_ratio(X) - np.log(self.threshold(rule))

    def predict(self, X, rule="min_risk"):
        return (self.decision_function(X, rule) > 0).astype(int)

    def posterior(self, X):
        """P(C1 | x) = sigmoid(ln Lambda + ln(p1/p2))."""
        z = self.log_likelihood_ratio(X) + np.log(self.p1_ / self.p2_)
        return 1.0 / (1.0 + np.exp(-z))

    def conditional_risk(self, X):
        """Expected cost of each decision: (R(say C1|x), R(say C2|x))."""
        P1 = self.posterior(X)
        P2 = 1 - P1
        return self.c11 * P1 + self.c12 * P2, self.c21 * P1 + self.c22 * P2

    # ---------------------------------------------------------------- evaluating
    def cost_matrix(self):
        return np.array([[self.c11, self.c12], [self.c21, self.c22]])

    def evaluate(self, X, y, rule="min_risk", y_pred=None):
        """Counts, rates, error rate and average cost per patient on labelled data."""
        y = np.asarray(y)
        yp = self.predict(X, rule) if y_pred is None else np.asarray(y_pred)
        caught = int(np.sum((yp == 1) & (y == 1)))
        missed = int(np.sum((yp == 0) & (y == 1)))
        false_alarm = int(np.sum((yp == 1) & (y == 0)))
        cleared = int(np.sum((yp == 0) & (y == 0)))
        n = len(y)
        cost = (self.c11 * caught + self.c21 * missed + self.c12 * false_alarm + self.c22 * cleared) / n
        return {"caught": caught, "missed": missed, "false_alarms": false_alarm, "cleared": cleared,
                "miss_rate": missed / max(caught + missed, 1), "false_alarm_rate": false_alarm / max(false_alarm + cleared, 1),
                "error_rate": (missed + false_alarm) / n, "avg_cost": cost}

    def theoretical(self, rule="min_risk"):
        """Exact rates and risk for the fitted Gaussian model (normal-table formulas of the slides)."""
        dmu = self.mu1_ - self.mu2_
        d2 = dmu @ self.Cinv_ @ dmu                 # squared Mahalanobis distance
        d = np.sqrt(d2)
        lx = np.log(self.threshold(rule))
        miss = norm.cdf((lx - d2 / 2) / d)           # ln Lambda | C1 ~ N(+d2/2, d2)
        fa = 1 - norm.cdf((lx + d2 / 2) / d)         # ln Lambda | C2 ~ N(-d2/2, d2)
        p1, p2 = self.p1_, self.p2_
        risk = (self.c11 * p1 * (1 - miss) + self.c21 * p1 * miss + self.c12 * p2 * fa + self.c22 * p2 * (1 - fa))
        return {"d2": d2, "miss_rate": miss, "false_alarm_rate": fa,
                "error_rate": p1 * miss + p2 * fa, "avg_cost": risk}


def perceptron_train(X, y, epochs=50, eta=1.0, seed=0):
    """Rosenblatt's error-correction rule (Haykin Sec. 1.3) on augmented inputs [1, x].
    Returns weights (bias first) and the number of mistakes in each epoch."""
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
