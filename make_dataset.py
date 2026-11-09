"""
Generate the tumor dataset for the Bayes minimum-risk lab.

The training set is built so that its maximum-likelihood estimates reproduce the
slide parameters EXACTLY (moment matching):

    malignant (C1):  mean mu1 = [0.6, 0.7, 0.6],  200 tumors  -> prior p1 = 0.2
    benign    (C2):  mean mu2 = [0.4, 0.3, 0.3],  800 tumors  -> prior p2 = 0.8
    shared covariance C = 0.04 I  (sigma = 0.2 for every feature, uncorrelated)

Hence the Bayes minimum-risk perceptron learned from the data has
    w = C^-1 (mu1 - mu2) = [5, 10, 7.5],   b = -10.875 - ln(0.25) = -9.489

The test set is an ordinary random sample from the same model (NOT moment matched),
so test results fluctuate around the theoretical values, as real data would.

Features are normalised scores: size, border irregularity, growth rate.
Because the classes are Gaussian, a few values fall slightly outside [0, 1].
"""
import numpy as np
import pandas as pd

MU1 = np.array([0.6, 0.7, 0.6])     # malignant
MU2 = np.array([0.4, 0.3, 0.3])     # benign
SIGMA = 0.2                          # C = SIGMA^2 I = 0.04 I
FEATURES = ["size", "border", "growth"]


def exact_sample(rng, n, mu, sigma):
    """n points whose sample mean is exactly mu and ML covariance exactly sigma^2 I."""
    z = rng.standard_normal((n, len(mu)))
    z -= z.mean(axis=0)                                  # sample mean -> 0
    s = z.T @ z / n                                      # ML (divide-by-n) covariance
    l = np.linalg.cholesky(s)
    z = z @ np.linalg.inv(l).T                           # sample covariance -> I
    return mu + sigma * z


def random_sample(rng, n, mu, sigma):
    return mu + sigma * rng.standard_normal((n, len(mu)))


def frame(x1, x2, prefix):
    x = np.vstack([x1, x2])
    lab = np.r_[np.ones(len(x1), int), np.zeros(len(x2), int)]
    df = pd.DataFrame(x, columns=FEATURES)
    df.insert(0, "tumor_id", [f"{prefix}{i:04d}" for i in range(1, len(df) + 1)])
    df["diagnosis"] = np.where(lab == 1, "malignant", "benign")
    df["label"] = lab                                    # 1 = malignant (C1), 0 = benign (C2)
    return df


if __name__ == "__main__":
    rng = np.random.default_rng(2026)
    train = frame(exact_sample(rng, 200, MU1, SIGMA), exact_sample(rng, 800, MU2, SIGMA), "TR")
    train = train.sample(frac=1, random_state=7).reset_index(drop=True)
    test = frame(random_sample(rng, 2000, MU1, SIGMA), random_sample(rng, 8000, MU2, SIGMA), "TE")
    test = test.sample(frac=1, random_state=11).reset_index(drop=True)
    slide = pd.DataFrame({"tumor_id": ["A", "B", "C"], "size": [0.25, 0.75, 0.45],
                          "border": [0.10, 0.90, 0.45], "growth": [0.05, 0.80, 0.40]})
    train.to_csv("data/tumor_train.csv", index=False, float_format="%.8f")
    test.to_csv("data/tumor_test.csv", index=False, float_format="%.8f")
    slide.to_csv("data/slide_tumors.csv", index=False)
    print("train:", train.shape, train.label.value_counts().to_dict())
    print("test :", test.shape, test.label.value_counts().to_dict())
