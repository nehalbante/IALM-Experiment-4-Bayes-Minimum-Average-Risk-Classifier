"""Mathematical reference content for the dedicated reference page.

Each entry documents the symbols, the general formula, the formula with the
practical's values, and its interpretation. LaTeX strings are rendered with
Streamlit's ``st.latex``.
"""
from __future__ import annotations

# fmt: off
TOPICS = [
    {
        "title": "1. Bayes' theorem",
        "intro": "Bayes' theorem converts the class-conditional likelihood and the prior into the posterior probability that a tumor belongs to a class given its features.",
        "symbols": [
            ("P(C_i)", "prior probability of class i"),
            ("p(x | C_i)", "class-conditional likelihood of observing x in class i"),
            ("P(C_i | x)", "posterior probability of class i given x"),
            ("p(x)", "unconditional (marginal) density of x"),
        ],
        "general": [r"P(C_i\mid\mathbf{x}) = \frac{p(\mathbf{x}\mid C_i)\,P(C_i)}{p(\mathbf{x})}"],
        "practical": [r"P(\mathcal C_1\mid\mathbf{x}) = \frac{p(\mathbf{x}\mid\mathcal C_1)\,p_1}{p(\mathbf{x})}"],
        "interpretation": "The posterior is proportional to likelihood × prior; the denominator normalises the two class posteriors so they sum to one.",
    },
    {
        "title": "2. Prior probabilities",
        "intro": "The priors encode how common each class is before any feature is observed.",
        "symbols": [
            ("p_1", "prior probability of malignant (C1)"),
            ("p_2", "prior probability of benign (C2)"),
        ],
        "general": [r"p_i = P(\mathcal C_i),\qquad p_1 + p_2 = 1"],
        "practical": [r"p_1 = 0.2,\qquad p_2 = 0.8"],
        "interpretation": "Only 20% of tumors are malignant in the original population. Priors are estimated from training class counts: p̂_i = N_i / N.",
    },
    {
        "title": "3. Likelihood functions",
        "intro": "Each class is a multivariate Gaussian. Malignant and benign share one covariance matrix but have different means.",
        "symbols": [
            ("μ_i", "mean vector of class i"),
            ("C", "shared covariance matrix"),
            ("d", "feature dimension (3 here)"),
        ],
        "general": [r"p(\mathbf{x}\mid\mathcal C_i) = \frac{1}{(2\pi)^{d/2}|\mathbf C|^{1/2}}\exp\!\left(-\tfrac12(\mathbf{x}-\boldsymbol\mu_i)^T\mathbf C^{-1}(\mathbf{x}-\boldsymbol\mu_i)\right)"],
        "practical": [
            r"\boldsymbol\mu_1=[0.6,\,0.7,\,0.6],\quad \boldsymbol\mu_2=[0.4,\,0.3,\,0.3]",
            r"\mathbf C = 0.04\,\mathbf I,\quad \mathbf C^{-1}=25\,\mathbf I",
        ],
        "interpretation": "Malignant tumors score higher on every feature. A shared covariance means both classes have equal spread, so the decision boundary is a straight line.",
    },
    {
        "title": "4. Likelihood ratio",
        "intro": "The likelihood ratio compares how much more likely the observed features are under malignant than under benign.",
        "symbols": [("Λ(x)", "likelihood ratio")],
        "general": [r"\Lambda(\mathbf{x}) = \frac{p(\mathbf{x}\mid\mathcal C_1)}{p(\mathbf{x}\mid\mathcal C_2)}"],
        "practical": [r"\Lambda(\mathbf{x}) = \exp\!\big(\mathbf w^T\mathbf x + b_{\mathrm{geom}}\big)"],
        "interpretation": "Λ(x) > 1 means the data favour malignant. With equal costs and priors, deciding C1 iff Λ(x) > 1 is the maximum-likelihood rule.",
    },
    {
        "title": "5. Log-likelihood ratio",
        "intro": "Taking logs turns the Gaussian ratio into a linear function of x — the discriminant.",
        "symbols": [("ln Λ(x)", "log-likelihood ratio (linear in x)")],
        "general": [r"\ln\Lambda(\mathbf{x}) = \mathbf w^T\mathbf x + b_{\mathrm{geom}}"],
        "practical": [r"\ln\Lambda(\mathbf{x}) = 5\,\text{size} + 10\,\text{border} + 7.5\,\text{growth} - 10.875"],
        "interpretation": "Because the covariance is shared, the quadratic terms cancel and only a linear form remains — the Bayes classifier is a single perceptron.",
    },
    {
        "title": "6. Geometric decision boundary",
        "intro": "The geometric bias depends only on the class geometry (means and covariance), not on priors or costs.",
        "symbols": [("b_geom", "geometric bias of the discriminant")],
        "general": [r"b_{\mathrm{geom}} = \tfrac12\big(\boldsymbol\mu_2^T\mathbf C^{-1}\boldsymbol\mu_2 - \boldsymbol\mu_1^T\mathbf C^{-1}\boldsymbol\mu_1\big)"],
        "practical": [r"b_{\mathrm{geom}} = \tfrac12(8.5 - 30.25) = -10.875"],
        "interpretation": "It is where the two Gaussian log-likelihoods are equal — the midpoint between the class centres, weighted by C⁻¹.",
    },
    {
        "title": "7. Maximum-likelihood (ML) classifier",
        "intro": "The ML rule ignores priors and costs and picks the class that makes the data most likely.",
        "symbols": [("ξ_ML", "ML likelihood-ratio threshold (= 1)")],
        "general": [r"\text{decide } \mathcal C_1 \iff \Lambda(\mathbf{x}) > 1,\quad \xi_{ML}=1"],
        "practical": [r"\text{decide malignant } \iff \ln\Lambda(\mathbf{x}) > 0,\quad b_{ML}=b_{\mathrm{geom}}=-10.875"],
        "interpretation": "Equivalent to Haykin's assumption of equal priors and equal (0-1) costs; it minimises error only in that symmetric case.",
    },
    {
        "title": "8. Maximum-a-posteriori (MAP) classifier",
        "intro": "The MAP rule incorporates priors and chooses the class with the larger posterior; it minimises the probability of error.",
        "symbols": [("ξ_MAP", "MAP likelihood-ratio threshold = p2 / p1")],
        "general": [r"\text{decide } \mathcal C_1 \iff \Lambda(\mathbf{x}) > \frac{p_2}{p_1} = \xi_{MAP}"],
        "practical": [r"\xi_{MAP} = \frac{0.8}{0.2} = 4,\quad b_{MAP} = -10.875 - \ln 4 = -12.261"],
        "interpretation": "Because benign tumors are 4× more common, the evidence must be 4× stronger before calling a tumor malignant.",
    },
    {
        "title": "9. Bayes minimum-average-risk classifier",
        "intro": "The minimum-risk rule chooses the decision with the smallest expected cost, using the full cost matrix instead of the 0-1 error.",
        "symbols": [
            ("c_ij", "cost of deciding class i when the true class is j"),
            ("R(decide C_i | x)", "conditional risk of deciding class i"),
        ],
        "general": [
            r"R(\text{say }\mathcal C_1\mid\mathbf{x}) = c_{11}P_1 + c_{12}P_2",
            r"R(\text{say }\mathcal C_2\mid\mathbf{x}) = c_{21}P_1 + c_{22}P_2",
        ],
        "practical": [r"R(\text{say mal})=8P_1+3P_2,\qquad R(\text{say ben})=40P_1+1P_2"],
        "interpretation": "We pick malignant when R(say mal) < R(say ben). This minimises expected cost, which is different from minimising the number of errors.",
    },
    {
        "title": "10. Cost matrix",
        "intro": "The cost matrix penalises the four outcomes. Only the regrets (differences) affect the decision rule.",
        "symbols": [
            ("c11 = 8", "correctly flagging a malignant tumor (biopsy cost)"),
            ("c12 = 3", "false alarm: unnecessary biopsy of a benign tumor"),
            ("c21 = 40", "miss: failing to detect a malignant tumor"),
            ("c22 = 1", "correctly clearing a benign tumor"),
        ],
        "general": [r"\mathbf C = \begin{bmatrix} c_{11} & c_{12} \\ c_{21} & c_{22} \end{bmatrix}"],
        "practical": [r"\mathbf C = \begin{bmatrix} 8 & 3 \\ 40 & 1 \end{bmatrix},\quad \text{regret}_{\text{miss}}=32,\ \text{regret}_{\text{FA}}=2"],
        "interpretation": "Missing a cancer is far more costly (40 vs 8) than a needless biopsy (3 vs 1), so borderline cases are called malignant.",
    },
    {
        "title": "11. Decision threshold",
        "intro": "The threshold ξ folds priors and cost regrets into a single likelihood-ratio value.",
        "symbols": [("ξ", "minimum-risk likelihood-ratio threshold")],
        "general": [r"\xi = \frac{p_2}{p_1}\cdot\frac{c_{12}-c_{22}}{c_{21}-c_{11}}"],
        "practical": [r"\xi = \frac{0.8}{0.2}\cdot\frac{3-1}{40-8} = 4\cdot\frac{2}{32} = 0.25"],
        "interpretation": "ξ = 0.25 < 1 means even evidence 4× against malignant (Λ = 0.25) still triggers a malignant call, because misses are expensive.",
    },
    {
        "title": "12. Discriminant function",
        "intro": "The discriminant is the perceptron output g(x) = wᵀx + b; its sign gives the decision.",
        "symbols": [
            ("w", "weight vector = C⁻¹(μ₁ − μ₂)"),
            ("b", "bias = b_geom − ln ξ"),
        ],
        "general": [
            r"\mathbf w = \mathbf C^{-1}(\boldsymbol\mu_1-\boldsymbol\mu_2)",
            r"b = b_{\mathrm{geom}} - \ln\xi",
            r"g(\mathbf{x}) = \mathbf w^T\mathbf{x} + b > 0 \Rightarrow \mathcal C_1",
        ],
        "practical": [
            r"\mathbf w = 25\,[0.2,\,0.4,\,0.3] = [5,\,10,\,7.5]",
            r"b = -10.875 - \ln(0.25) = -10.875 + 1.386 = -9.489",
        ],
        "interpretation": "The weights reward informative features; the bias shifts the boundary to account for priors and costs.",
    },
    {
        "title": "13. Linear Discriminant Analysis (LDA)",
        "intro": "sklearn's LDA fits the same shared-covariance Gaussian model but applies the MAP rule (ξ = p2/p1), so it differs from the minimum-risk classifier only by the cost factor.",
        "symbols": [("b_LDA", "LDA intercept (MAP bias)")],
        "general": [
            r"b_{\mathrm{LDA}} = b_{\mathrm{geom}} - \ln\frac{p_2}{p_1}",
            r"b_{\mathrm{min risk}} = b_{\mathrm{LDA}} + \ln\frac{c_{21}-c_{11}}{c_{12}-c_{22}}",
        ],
        "practical": [
            r"b_{\mathrm{LDA}} = -12.2613",
            r"b_{\mathrm{min risk}} = -12.2613 + \ln 16 = -9.4887",
        ],
        "interpretation": "LDA and the Bayes classifier share the same weights; only the bias differs by the cost-ratio logarithm.",
    },
    {
        "title": "14. Rosenblatt perceptron learning rule",
        "intro": "The perceptron learns its weights from labelled examples by correcting every misclassified sample.",
        "symbols": [
            ("w(n)", "weight vector at step n"),
            ("η", "learning rate"),
            ("d(n)", "desired output: +1 malignant, −1 benign"),
            ("x(n)", "augmented input [1, x]"),
        ],
        "general": [r"\mathbf w(n+1) = \mathbf w(n) + \eta\, d(n)\,\mathbf x(n)\ \ \text{(on a mistake)}"],
        "practical": [r"\mathbf w(n+1) = \mathbf w(n) + d(n)\,\mathbf x(n)\quad (\eta=1)"],
        "interpretation": "The rule converges only for linearly separable data. The overlapping tumor classes are not separable, so mistakes never reach zero.",
    },
    {
        "title": "15. Confusion matrix and performance metrics",
        "intro": "Performance is measured both by the error rate and by the average cost — the quantity the minimum-risk rule actually optimises.",
        "symbols": [
            ("Caught (TP)", "malignant correctly classified"),
            ("Missed (FN)", "malignant classified benign"),
            ("False alarm (FP)", "benign classified malignant"),
            ("Cleared (TN)", "benign correctly classified"),
        ],
        "general": [
            r"\text{miss rate} = \frac{FN}{TP+FN}",
            r"\text{false-alarm rate} = \frac{FP}{FP+TN}",
            r"\text{error rate} = \frac{FN+FP}{N}",
            r"\text{avg cost} = \frac{c_{11}TP + c_{21}FN + c_{12}FP + c_{22}TN}{N}",
        ],
        "practical": [
            r"\text{avg cost} = \frac{8\cdot1947 + 40\cdot53 + 3\cdot1618 + 1\cdot6382}{10000} = 2.8932",
        ],
        "interpretation": "Accuracy and average cost are different criteria: a rule with a worse error rate can have a lower cost if it avoids expensive misses.",
    },
]
# fmt: on
