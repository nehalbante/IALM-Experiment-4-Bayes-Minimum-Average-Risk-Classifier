"""Bayes minimum-risk lab: plain-Python version of the solutions notebook (run from the lab folder)."""

# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import norm

np.set_printoptions(precision=4, suppress=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
MAL, BEN = "#C8473A", "#1F7A70"
F = ["size", "border", "growth"]

train = pd.read_csv("data/tumor_train.csv")
test = pd.read_csv("data/tumor_test.csv")
slide = pd.read_csv("data/slide_tumors.csv")
print(train.head())

# %%
print(train["diagnosis"].value_counts())
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
    for lab, col, name in [(0, BEN, "benign"), (1, MAL, "malignant")]:
        d = train[train.label == lab]
        ax.scatter(d[F[i]], d[F[j]], s=8, alpha=0.5, color=col, label=name)
    ax.set_xlabel(F[i]); ax.set_ylabel(F[j])
axes[0].legend()
fig.suptitle("Training tumors: malignant tumors sit higher on every feature")
plt.tight_layout(); plt.show()

# %%
X = train[F].to_numpy()
y = train["label"].to_numpy()
X1, X2 = X[y == 1], X[y == 0]          # malignant, benign
N = len(X)

p1, p2 = len(X1) / N, len(X2) / N
mu1, mu2 = X1.mean(axis=0), X2.mean(axis=0)
D = np.vstack([X1 - mu1, X2 - mu2])
C = D.T @ D / N
Cinv = np.linalg.inv(C)

print("p1, p2 =", p1, p2)
print("mu1 =", mu1, "  mu2 =", mu2)
print("C =\n", C)
print("C^-1 =\n", Cinv)

# %%
c11, c12, c21, c22 = 8, 3, 40, 1

regret_miss = c21 - c11          # extra cost of missing a cancer
regret_fa = c12 - c22            # extra cost of a needless biopsy
xi = p2 * regret_fa / (p1 * regret_miss)

print(f"regrets: miss {regret_miss}, false alarm {regret_fa}")
print(f"prior factor {p2/p1:.2f} x cost factor {regret_fa/regret_miss:.4f} = xi = {xi:.4f},  ln xi = {np.log(xi):.4f}")

# %%
w = Cinv @ (mu1 - mu2)
b_geom = 0.5 * (mu2 @ Cinv @ mu2 - mu1 @ Cinv @ mu1)   # class geometry only
b = b_geom - np.log(xi)                                 # priors and costs enter here

print("w      =", w, "   (slides: [5, 10, 7.5])")
print(f"b_geom = {b_geom:.4f}  (slides: 0.5(8.5 - 30.25) = -10.875)")
print(f"b      = {b:.4f}  (slides: -10.875 + 1.386 = -9.489)")

# %%
Xs = slide[F].to_numpy()
log_lam = Xs @ w + b_geom
yout = Xs @ w + b
post = 1 / (1 + np.exp(-(log_lam + np.log(p1 / p2))))
R_mal = c11 * post + c12 * (1 - post)
R_ben = c21 * post + c22 * (1 - post)

out = slide.copy()
out["w.x"] = Xs @ w
out["ln_Lambda"] = log_lam
out["y = w.x + b"] = yout
out["P(malignant|x)"] = post
out["R(say mal)"] = R_mal
out["R(say ben)"] = R_ben
out["Haykin (xi=1)"] = np.where(log_lam > 0, "malignant", "benign")
out["min risk (xi=0.25)"] = np.where(yout > 0, "malignant", "benign")
print(out.round(3).to_string())

# %%
Xt, yt = test[F].to_numpy(), test["label"].to_numpy()


def scorecard(pred, truth):
    caught = np.sum((pred == 1) & (truth == 1)); missed = np.sum((pred == 0) & (truth == 1))
    fa = np.sum((pred == 1) & (truth == 0)); cleared = np.sum((pred == 0) & (truth == 0))
    n = len(truth)
    return {"caught": caught, "missed": missed, "false alarms": fa, "cleared": cleared,
            "miss rate": missed / (caught + missed), "false-alarm rate": fa / (fa + cleared),
            "error rate": (missed + fa) / n,
            "avg cost": (c11 * caught + c21 * missed + c12 * fa + c22 * cleared) / n}


pred = (Xt @ w + b > 0).astype(int)
emp = scorecard(pred, yt)

d2 = (mu1 - mu2) @ Cinv @ (mu1 - mu2); d = np.sqrt(d2)
miss_th = norm.cdf((np.log(xi) - d2 / 2) / d)
fa_th = 1 - norm.cdf((np.log(xi) + d2 / 2) / d)
risk_th = c11 * p1 * (1 - miss_th) + c21 * p1 * miss_th + c12 * p2 * fa_th + c22 * p2 * (1 - fa_th)

print(f"d^2 = {d2:.3f}")
print(f"{'':18s}{'test data':>12s}{'theory':>10s}")
print(f"{'miss rate':18s}{emp['miss rate']:12.4f}{miss_th:10.4f}")
print(f"{'false-alarm rate':18s}{emp['false-alarm rate']:12.4f}{fa_th:10.4f}")
print(f"{'avg cost':18s}{emp['avg cost']:12.4f}{risk_th:10.4f}   (slides: 2.925)")
print("\nconfusion counts:", {k: int(v) for k, v in emp.items() if k in ("caught", "missed", "false alarms", "cleared")})

# %%
rules = {"ML (xi=1)": 1.0, "MAP (xi=4)": p2 / p1, "Min risk (xi=0.25)": xi}
rows = []
for name, x_ in rules.items():
    s = scorecard((Xt @ w + b_geom - np.log(x_) > 0).astype(int), yt)
    s["rule"], s["bias"] = name, b_geom - np.log(x_)
    rows.append(s)
for name, const in [("Everyone benign", 0), ("Everyone malignant", 1)]:
    s = scorecard(np.full(len(yt), const), yt); s["rule"] = name; rows.append(s)
table = pd.DataFrame(rows).set_index("rule")[["bias", "caught", "missed", "false alarms", "error rate", "avg cost"]]
print(table.round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
cols = ["#5B7C99", "#B7791F", "#6A4C93", "#B8C3C5", "#B8C3C5"]
axes[0].bar(table.index, table["error rate"], color=cols); axes[0].set_title("Error rate (lower = fewer mistakes)")
axes[1].bar(table.index, table["avg cost"], color=cols); axes[1].set_title("Average cost per patient (lower = cheaper)")
for ax in axes: ax.tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.show()

# %%
L = Xt @ w + b_geom
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
ax = axes[0]
bins = np.linspace(-12, 12, 60)
ax.hist(L[yt == 0], bins=bins, density=True, alpha=0.55, color=BEN, label="benign")
ax.hist(L[yt == 1], bins=bins, density=True, alpha=0.55, color=MAL, label="malignant")
g = np.linspace(-12, 12, 400)
ax.plot(g, norm.pdf(g, -d2 / 2, d), color=BEN); ax.plot(g, norm.pdf(g, d2 / 2, d), color=MAL)
for name, x_, col in [("ML", 1, "#5B7C99"), ("MAP", p2 / p1, "#B7791F"), ("min risk", xi, "#6A4C93")]:
    ax.axvline(np.log(x_), color=col, lw=2, label=f"{name}: ln xi = {np.log(x_):.2f}")
ax.set_xlabel("ln Lambda(x) = w.x + b_geom"); ax.legend(fontsize=8); ax.set_title("Log-likelihood ratio by class")

ax = axes[1]
ts = np.linspace(-6, 6, 1201)
emp_cost = [scorecard((L > t).astype(int), yt)["avg cost"] for t in ts]
mt = norm.cdf((ts - d2 / 2) / d); ft = 1 - norm.cdf((ts + d2 / 2) / d)
th_cost = c11 * p1 * (1 - mt) + c21 * p1 * mt + c12 * p2 * ft + c22 * p2 * (1 - ft)
ax.plot(ts, emp_cost, color="#B8C3C5", lw=3, label="test data")
ax.plot(ts, th_cost, color="#6A4C93", lw=2, label="theory")
ax.axvline(np.log(xi), color="#6A4C93", ls="--")
best = ts[np.argmin(th_cost)]
ax.set_title(f"Average cost vs threshold (theory minimum at {best:.2f})")
ax.set_xlabel("threshold on ln Lambda"); ax.set_ylabel("cost per patient"); ax.legend()
plt.tight_layout(); plt.show()

# %%
from bayes_min_risk import BayesMinRiskClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = BayesMinRiskClassifier(c11=8, c12=3, c21=40, c22=1).fit(X, y)
print("class:  w =", clf.w_, " b =", round(clf.b_, 4), " xi =", clf.xi_)
print("        test avg cost =", round(clf.evaluate(Xt, yt)["avg_cost"], 4), " theory =", round(clf.theoretical()["avg_cost"], 4))

lda = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)
print("LDA:    w =", lda.coef_[0], " b =", round(lda.intercept_[0], 4))
print("LDA b + ln 16 =", round(lda.intercept_[0] + np.log(16), 4), " (equals our minimum-risk bias)")

# %%
from bayes_min_risk import perceptron_train

wl, mistakes = perceptron_train(X, y, epochs=40, eta=1.0, seed=0)
b_l, w_l = wl[0], wl[1:]
print("learned w =", w_l, " b =", round(b_l, 3))
print("direction, learned :", w_l / np.linalg.norm(w_l))
print("direction, designed:", w / np.linalg.norm(w))
s_l = scorecard((Xt @ w_l + b_l > 0).astype(int), yt)
print(f"learned perceptron on test: error {s_l['error rate']:.3f}, avg cost {s_l['avg cost']:.3f}")

plt.figure(figsize=(6, 3.2))
plt.plot(range(1, len(mistakes) + 1), mistakes, marker="o", ms=3, color="#5B7C99")
plt.xlabel("epoch"); plt.ylabel("training mistakes"); plt.title("Rosenblatt rule never reaches zero mistakes")
plt.tight_layout(); plt.show()
