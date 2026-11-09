"""Builds the instructor (solutions, executed) and student (TODO) versions of the lab notebook."""
import nbformat as nbf
from nbclient import NotebookClient

cells = []  # (kind, source, student_source or None)


def md(s):
    cells.append(("md", s.strip("\n"), None))


def code(s, student=None):
    cells.append(("code", s.strip("\n"), student.strip("\n") if student else None))


md(r"""
# Lab: Bayes Minimum-Risk Classifier as a Perceptron
### Tumor classification with three features (size, border irregularity, growth rate)

This lab turns the classroom example into working Python. In class we **assumed** the Gaussian parameters and computed the perceptron by hand. Here we **learn** them from data and check that we get the same perceptron:

| Quantity | Slide value |
|---|---|
| Priors | $p_1 = 0.2$ (malignant, $\mathcal{C}_1$), $p_2 = 0.8$ (benign, $\mathcal{C}_2$) |
| Means | $\boldsymbol\mu_1 = [0.6,\,0.7,\,0.6]$, $\boldsymbol\mu_2 = [0.4,\,0.3,\,0.3]$ |
| Shared covariance | $\mathbf C = 0.04\,\mathbf I$, so $\mathbf C^{-1} = 25\,\mathbf I$ |
| Costs | $c_{11}=8,\ c_{12}=3,\ c_{21}=40,\ c_{22}=1$ |
| Threshold | $\xi = \dfrac{p_2(c_{12}-c_{22})}{p_1(c_{21}-c_{11})} = 0.25$ |
| Perceptron | $\mathbf w = [5,\,10,\,7.5]$, $b = -9.489$ |

**Files**
- `data/tumor_train.csv`: 1000 tumors (200 malignant, 800 benign). Built so that its maximum-likelihood estimates equal the slide values exactly.
- `data/tumor_test.csv`: 10,000 fresh tumors from the same model (an ordinary random sample).
- `data/slide_tumors.csv`: tumors A, B, C from the slides.
- `bayes_min_risk.py`: the classifier as a reusable class (used in Part 9).

**By the end you will** estimate priors, means and covariance by maximum likelihood; compute $\xi$, $\mathbf w$ and $b$; classify the slide tumors; and compare ML, MAP and minimum-risk rules on test data.
""")

code(r"""
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
train.head()
""")

# ---------------------------------------------------------------- Part 1
md(r"""
## Part 1 · Explore the data
Label convention: `label = 1` is malignant ($\mathcal C_1$), `label = 0` is benign ($\mathcal C_2$).
""")

code(r"""
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
""")

# ---------------------------------------------------------------- Part 2
md(r"""
## Part 2 · Learn the model by maximum likelihood
For Gaussian classes the maximum-likelihood estimates are simple averages:

$$\hat p_i = \frac{N_i}{N},\qquad \hat{\boldsymbol\mu}_i = \frac{1}{N_i}\sum_{\mathbf x\in\mathcal C_i}\mathbf x,\qquad \hat{\mathbf C} = \frac1N\sum_{i=1}^{2}\ \sum_{\mathbf x\in\mathcal C_i}(\mathbf x-\hat{\boldsymbol\mu}_i)(\mathbf x-\hat{\boldsymbol\mu}_i)^T$$

The covariance is **pooled**: every tumor is measured from its own class mean, because the model assumes both classes share one $\mathbf C$.
""")

code(r"""
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
""", student=r"""
X = train[F].to_numpy()
y = train["label"].to_numpy()
X1, X2 = X[y == 1], X[y == 0]          # malignant, benign
N = len(X)

# TODO: priors p1, p2 from the class counts
p1, p2 = ...
# TODO: class means mu1, mu2
mu1, mu2 = ...
# TODO: pooled ML covariance C (stack X1 - mu1 and X2 - mu2, then D.T @ D / N) and its inverse
C = ...
Cinv = ...

print("p1, p2 =", p1, p2)
print("mu1 =", mu1, "  mu2 =", mu2)
print("C =\n", C)
""")

md(r"""
**Check against the slides.** You should get $p_1 = 0.2$, $\boldsymbol\mu_1 = [0.6, 0.7, 0.6]$, $\boldsymbol\mu_2 = [0.4, 0.3, 0.3]$, $\mathbf C = 0.04\,\mathbf I$ and $\mathbf C^{-1} = 25\,\mathbf I$ (tiny off-diagonal values like $10^{-9}$ are rounding in the CSV file).
""")

# ---------------------------------------------------------------- Part 3
md(r"""
## Part 3 · Costs and the threshold ξ
Only the **regrets** (cost differences) matter:

$$\xi = \frac{p_2\,(c_{12}-c_{22})}{p_1\,(c_{21}-c_{11})} = \underbrace{\frac{p_2}{p_1}}_{\text{prior factor}}\times\underbrace{\frac{c_{12}-c_{22}}{c_{21}-c_{11}}}_{\text{cost factor}}$$
""")

code(r"""
c11, c12, c21, c22 = 8, 3, 40, 1

regret_miss = c21 - c11          # extra cost of missing a cancer
regret_fa = c12 - c22            # extra cost of a needless biopsy
xi = p2 * regret_fa / (p1 * regret_miss)

print(f"regrets: miss {regret_miss}, false alarm {regret_fa}")
print(f"prior factor {p2/p1:.2f} x cost factor {regret_fa/regret_miss:.4f} = xi = {xi:.4f},  ln xi = {np.log(xi):.4f}")
""", student=r"""
c11, c12, c21, c22 = 8, 3, 40, 1

# TODO: the two regrets and the threshold xi
regret_miss = ...
regret_fa = ...
xi = ...

print(f"xi = {xi:.4f},  ln xi = {np.log(xi):.4f}")
""")

# ---------------------------------------------------------------- Part 4
md(r"""
## Part 4 · From Bayes rule to a perceptron
With a shared covariance the log-likelihood ratio is linear in $\mathbf x$ (Haykin Sec. 1.4):

$$\ln\Lambda(\mathbf x) = \underbrace{(\boldsymbol\mu_1-\boldsymbol\mu_2)^T\mathbf C^{-1}}_{\mathbf w^T}\mathbf x + \tfrac12\left(\boldsymbol\mu_2^T\mathbf C^{-1}\boldsymbol\mu_2-\boldsymbol\mu_1^T\mathbf C^{-1}\boldsymbol\mu_1\right)$$

Decide malignant when $\ln\Lambda(\mathbf x) > \ln\xi$, i.e. when the perceptron output $y=\mathbf w^T\mathbf x + b > 0$ with

$$\mathbf w = \mathbf C^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2),\qquad b = \tfrac12\left(\boldsymbol\mu_2^T\mathbf C^{-1}\boldsymbol\mu_2-\boldsymbol\mu_1^T\mathbf C^{-1}\boldsymbol\mu_1\right) - \ln\xi .$$
""")

code(r"""
w = Cinv @ (mu1 - mu2)
b_geom = 0.5 * (mu2 @ Cinv @ mu2 - mu1 @ Cinv @ mu1)   # class geometry only
b = b_geom - np.log(xi)                                 # priors and costs enter here

print("w      =", w, "   (slides: [5, 10, 7.5])")
print(f"b_geom = {b_geom:.4f}  (slides: 0.5(8.5 - 30.25) = -10.875)")
print(f"b      = {b:.4f}  (slides: -10.875 + 1.386 = -9.489)")
""", student=r"""
# TODO: weights and bias of the Bayes minimum-risk perceptron
w = ...
b_geom = ...        # 0.5 * (mu2' Cinv mu2 - mu1' Cinv mu1)
b = ...             # b_geom - ln(xi)

print("w =", w, "  b =", b)
""")

# ---------------------------------------------------------------- Part 5
md(r"""
## Part 5 · Classify the slide tumors A, B, C
For each tumor we compute the perceptron output, the posterior $P(\mathcal C_1\mid\mathbf x)=\sigma\big(\ln\Lambda + \ln\tfrac{p_1}{p_2}\big)$ and the expected cost of each decision:

$$R(\text{say malignant}\mid\mathbf x) = c_{11}P_1 + c_{12}P_2,\qquad R(\text{say benign}\mid\mathbf x) = c_{21}P_1 + c_{22}P_2$$
""")

code(r"""
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
out.round(3)
""", student=r"""
Xs = slide[F].to_numpy()
# TODO: ln Lambda, perceptron output y, posterior P(malignant|x), and both conditional risks
log_lam = ...
yout = ...
post = ...
R_mal = ...
R_ben = ...
""")

md(r"""
**Expected:** A: $y=-6.86$ (benign), B: $y=+9.26$ (malignant), C: $y=+0.26$ (malignant).
Tumor C is the interesting one: $\ln\Lambda=-1.125$, so the evidence alone says benign (Haykin's $\xi=1$ rule) and its cancer probability is only 7.5%. But 7.5% is above the 1/17 ≈ 5.9% action threshold set by the costs, so the minimum-risk perceptron calls it malignant. Check that $R(\text{say mal}) < R(\text{say ben})$ for C.
""")

# ---------------------------------------------------------------- Part 6
md(r"""
## Part 6 · Test on 10,000 new tumors
The training set reproduces the slide parameters exactly; the test set is an ordinary random sample, so its results should be **close to**, not equal to, the theoretical values.

Theory for this model: the squared Mahalanobis distance is $d^2 = (\boldsymbol\mu_1-\boldsymbol\mu_2)^T\mathbf C^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2) = 7.25$, and
$\ln\Lambda\mid\mathcal C_1 \sim N(+d^2/2,\ d^2)$, $\ln\Lambda\mid\mathcal C_2 \sim N(-d^2/2,\ d^2)$. Hence

$$P(\text{miss}) = \Phi\!\left(\frac{\ln\xi - d^2/2}{d}\right),\qquad P(\text{false alarm}) = 1-\Phi\!\left(\frac{\ln\xi + d^2/2}{d}\right).$$
""")

code(r"""
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
""")

# ---------------------------------------------------------------- Part 7
md(r"""
## Part 7 · ML vs MAP vs minimum risk
All three rules use the **same** weights $\mathbf w$; only the bias changes, because only $\ln\xi$ changes:

| Rule | $\xi$ | Bias $b = b_{geom} - \ln\xi$ |
|---|---|---|
| ML (Haykin's case) | 1 | −10.875 |
| MAP | $p_2/p_1 = 4$ | −12.261 |
| Minimum risk | 0.25 | −9.489 |

We add two trivial policies for reference: call everyone benign, and call everyone malignant.
""")

code(r"""
rules = {"ML (xi=1)": 1.0, "MAP (xi=4)": p2 / p1, "Min risk (xi=0.25)": xi}
rows = []
for name, x_ in rules.items():
    s = scorecard((Xt @ w + b_geom - np.log(x_) > 0).astype(int), yt)
    s["rule"], s["bias"] = name, b_geom - np.log(x_)
    rows.append(s)
for name, const in [("Everyone benign", 0), ("Everyone malignant", 1)]:
    s = scorecard(np.full(len(yt), const), yt); s["rule"] = name; rows.append(s)
table = pd.DataFrame(rows).set_index("rule")[["bias", "caught", "missed", "false alarms", "error rate", "avg cost"]]
display(table.round(3))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
cols = ["#5B7C99", "#B7791F", "#6A4C93", "#B8C3C5", "#B8C3C5"]
axes[0].bar(table.index, table["error rate"], color=cols); axes[0].set_title("Error rate (lower = fewer mistakes)")
axes[1].bar(table.index, table["avg cost"], color=cols); axes[1].set_title("Average cost per patient (lower = cheaper)")
for ax in axes: ax.tick_params(axis="x", rotation=25)
plt.tight_layout(); plt.show()
""")

md(r"""
**Questions**
1. Which rule makes the fewest mistakes? Which is cheapest? Why are they different rules?
2. How many of the 2000 cancers does MAP miss? How many does minimum risk miss?
3. "Everyone malignant" has an 80% error rate. Why is its cost still competitive?
""")

# ---------------------------------------------------------------- Part 8
md(r"""
## Part 8 · See the decision in one dimension
The perceptron squeezes three features into one number, $\ln\Lambda(\mathbf x)$. Plot it for each class and draw the three thresholds $\ln\xi$. Then sweep the threshold and watch the average cost: its minimum should sit at $\ln\xi = \ln 0.25 = -1.386$.
""")

code(r"""
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
""")

# ---------------------------------------------------------------- Part 9
md(r"""
## Part 9 · Cross-checks
**(a) The reusable class.** `bayes_min_risk.py` packages Parts 2–7. It should give the same numbers.

**(b) scikit-learn's LDA** fits the same Gaussian model but uses the MAP rule ($\xi = p_2/p_1$). Its weights equal ours, and its intercept differs by exactly the cost factor: $b_{\text{min risk}} = b_{\text{LDA}} + \ln\dfrac{c_{21}-c_{11}}{c_{12}-c_{22}} = b_{\text{LDA}} + \ln 16$.
""")

code(r"""
from bayes_min_risk import BayesMinRiskClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

clf = BayesMinRiskClassifier(c11=8, c12=3, c21=40, c22=1).fit(X, y)
print("class:  w =", clf.w_, " b =", round(clf.b_, 4), " xi =", clf.xi_)
print("        test avg cost =", round(clf.evaluate(Xt, yt)["avg_cost"], 4), " theory =", round(clf.theoretical()["avg_cost"], 4))

lda = LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)
print("LDA:    w =", lda.coef_[0], " b =", round(lda.intercept_[0], 4))
print("LDA b + ln 16 =", round(lda.intercept_[0] + np.log(16), 4), " (equals our minimum-risk bias)")
""")

# ---------------------------------------------------------------- Part 10
md(r"""
## Part 10 · Designed vs learned perceptron
Our perceptron was **designed** from the Gaussian model. Rosenblatt's perceptron **learns** its weights from labelled examples with the error-correction rule
$\mathbf w(n+1) = \mathbf w(n) + \eta\, d(n)\,\mathbf x(n)$ whenever a training tumor is misclassified.
The convergence theorem only holds for linearly separable classes. Our classes overlap, so watch what happens.
""")

code(r"""
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
""")

md(r"""
**What to notice**
- The mistakes never reach zero: overlapping classes cannot be separated, so the learning rule keeps correcting and never settles.
- The learned weights point in a broadly similar direction to $\mathbf w=[5,10,7.5]$ but are not optimal, and the rule knows nothing about priors or costs.
- The designed perceptron uses the model to place the boundary where the **expected cost** is lowest.
""")

# ---------------------------------------------------------------- Exercises
md(r"""
## Exercises
1. **Cheaper misses.** Set $c_{21}=11$ (other costs unchanged). Recompute $\xi$, $b$ and the test average cost. How do the decisions for tumor C and the miss rate change?
2. **Screening population.** In a screening programme $p_1 = 0.05$. Pass `priors=(0.05, 0.95)` to `BayesMinRiskClassifier` and recompute $\xi$ and $b$. Which way does the boundary move, and why?
3. **Haykin's assumptions.** Show that with $p_1=p_2=0.5$, $c_{11}=c_{22}=0$ and $c_{12}=c_{21}=1$ the bias becomes $b_{geom}$ alone ($\xi=1$).
4. **Feature importance.** Why does *border* get the largest weight (10) and *size* the smallest (5)? Change $\boldsymbol\mu_1$ for size to 0.8 in `make_dataset.py`, regenerate the data and predict the new weight before running the code.
5. **Correlated features (challenge).** Change the covariance in `make_dataset.py` so that size and growth have correlation 0.5. Do the weights still equal $25(\boldsymbol\mu_1-\boldsymbol\mu_2)$? Explain using $\mathbf w=\mathbf C^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2)$.
""")


def build(student, path, execute):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    for kind, src, stu in cells:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(stu if (student and stu) else src))
    if execute:
        NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    nbf.write(nb, path)


if __name__ == "__main__":
    build(False, "Bayes_MinRisk_Lab_solutions.ipynb", execute=True)
    build(True, "Bayes_MinRisk_Lab_student.ipynb", execute=False)
    print("notebooks written")
