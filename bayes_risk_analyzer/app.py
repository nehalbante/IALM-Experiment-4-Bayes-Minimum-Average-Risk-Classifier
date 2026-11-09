# command to start - "streamlit run app.py"

"""Bayes Risk Analyzer — Streamlit application.

Launch with ``streamlit run app.py`` from the ``bayes_risk_analyzer`` folder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import bayes, config, data_loader, extensions, lda, metrics, perceptron, ui, visualization
from src import theory as theory_module

st.set_page_config(
    page_title="Bayes Risk Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_css()


# ---------------------------------------------------------------- helpers
def fmt(x, digits: int = 4) -> str:
    return f"{float(x):.{digits}f}"


def fmt_vec(v, digits: int = 4) -> str:
    return "[" + ", ".join(f"{float(x):.{digits}f}" for x in v) + "]"


def fmt_costs(costs) -> str:
    return "[" + ", ".join(f"{c:g}" for c in costs) + "]"


# ---------------------------------------------------------------- cached data/model
@st.cache_resource(show_spinner=False)
def load_bundle() -> dict:
    """Load datasets and fit the default classifier once per session."""
    train = data_loader.load_train()
    test = data_loader.load_test()
    slide = data_loader.load_slide()
    X, y = data_loader.split_features_labels(train)
    Xt, yt = data_loader.split_features_labels(test)
    clf = bayes.fit_default_classifier(X, y)
    lda_model = lda.fit_lda(X, y)
    lda_p = lda.lda_params(lda_model)
    return {
        "train": train, "test": test, "slide": slide,
        "X": X, "y": y, "Xt": Xt, "yt": yt,
        "clf": clf, "lda_model": lda_model, "lda": lda_p,
    }


# ---------------------------------------------------------------- sidebar
NAV = [
    "Experiment Overview",
    "Part 1 – Part 10",
    "Dataset Explorer",
    "Classifier Playground",
    "Model Evaluation",
    "Decision Boundary Visualizer",
    "Perceptron Laboratory",
    "Extension Experiments",
    "Mathematical Reference",
]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 🎯 Bayes Risk Analyzer")
        st.caption("IALM Experiment 4 · Bayes Minimum Average Risk Classifier")
        page = st.radio("Navigation", NAV)
        st.markdown("---")
        st.caption(
            "Features: `size`, `border`, `growth` · Label: 1 = malignant, 0 = benign. "
            "Original datasets are read-only."
        )
    return page


# ================================================================ OVERVIEW
def render_overview(B: dict) -> None:
    ui.render_title()
    clf = B["clf"]
    st.markdown(
        "This interactive system implements the **Bayes minimum-average-risk classifier** as a single "
        "perceptron for tumor classification (Haykin, *Neural Networks and Learning Machines*, Sec. 1.4). "
        "It reproduces every part of the original laboratory and adds five extension experiments."
    )

    ui.section_header("Slide parameters (theoretical model)")
    params = [
        ("Priors", "$p_1 = 0.2$ (malignant), $p_2 = 0.8$ (benign)"),
        ("Means", "$\\mu_1 = [0.6, 0.7, 0.6]$, $\\mu_2 = [0.4, 0.3, 0.3]$"),
        ("Covariance", "$\\mathbf C = 0.04\\,\\mathbf I$, $\\mathbf C^{-1} = 25\\,\\mathbf I$"),
        ("Costs", "$c_{11}=8,\\ c_{12}=3,\\ c_{21}=40,\\ c_{22}=1$"),
        ("Threshold", "$\\xi = 0.25$"),
        ("Perceptron", "$\\mathbf w = [5, 10, 7.5]$, $b = -9.489$"),
    ]
    st.table(pd.DataFrame(params, columns=["Quantity", "Value"]))

    ui.section_header("Model learned from the training set")
    ui.metric_row([
        ("Malignant prior p₁", fmt(clf.p1_)),
        ("Benign prior p₂", fmt(clf.p2_)),
        ("Threshold ξ", fmt(clf.xi_)),
        ("Geometric bias b_geom", fmt(clf.b_geom_)),
        ("Minimum-risk bias b", fmt(clf.b_)),
    ])
    st.markdown(f"**Weights** `w = {fmt_vec(clf.w_)}`")

    ui.section_header("Datasets")
    st.markdown(
        f"- **Training** — {B['train'].shape[0]} tumors "
        f"({int((B['y'] == 1).sum())} malignant, {int((B['y'] == 0).sum())} benign).\n"
        f"- **Test** — {B['test'].shape[0]} fresh tumors from the same model.\n"
        f"- **Slides** — tumors A, B, C from the lecture slides."
    )

    ui.section_header("How to use")
    st.markdown(
        "Use the **sidebar** to move between sections. Start with **Part 1 – Part 10** to follow the "
        "laboratory, then explore the interactive **Playground**, **Decision Boundary Visualizer**, "
        "**Perceptron Laboratory** and **Extension Experiments**."
    )


# ---------------------------------------------------------------- main
def main() -> None:
    page = render_sidebar()
    try:
        B = load_bundle()
    except FileNotFoundError as exc:
        st.error(f"Dataset files not found: {exc}")
        st.info(
            f"Expected a `data/` folder containing `tumor_train.csv`, `tumor_test.csv` and "
            f"`slide_tumors.csv` in `{config.find_data_dir()}`."
        )
        return

    if page == "Experiment Overview":
        render_overview(B)
    elif page == "Part 1 – Part 10":
        render_parts(B)
    elif page == "Dataset Explorer":
        render_dataset_explorer(B)
    elif page == "Classifier Playground":
        render_playground(B)
    elif page == "Model Evaluation":
        render_evaluation(B)
    elif page == "Decision Boundary Visualizer":
        render_decision_boundary(B)
    elif page == "Perceptron Laboratory":
        render_perceptron_lab(B)
    elif page == "Extension Experiments":
        render_extensions(B)
    elif page == "Mathematical Reference":
        render_math_reference(B)


# ================================================================ PARTS 1-10
def render_parts(B: dict) -> None:
    ui.render_title()
    clf = B["clf"]
    X, y, Xt, yt = B["X"], B["y"], B["Xt"], B["yt"]
    train, slide = B["train"], B["slide"]
    c11, c12, c21, c22 = clf.c11, clf.c12, clf.c21, clf.c22

    tabs = st.tabs([f"Part {i}" for i in range(1, 11)])

    # ---------------- Part 1 ----------------
    with tabs[0]:
        ui.render_part_block(
            "Part 1 · Explore the data",
            aim="Load the training set and inspect class counts and the feature space.",
            theory="Label convention: `label = 1` is malignant (C₁), `label = 0` is benign (C₂).",
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Class distribution**")
            st.dataframe(train["diagnosis"].value_counts().rename("count"))
        with c2:
            st.plotly_chart(visualization.scatter_2d(train, "size", "border"), width="stretch")
        st.plotly_chart(visualization.scatter_2d(train, "size", "growth"), width="stretch")
        st.plotly_chart(visualization.scatter_2d(train, "border", "growth"), width="stretch")
        ui.card(
            "**Observation.** Malignant tumors sit higher on every feature; the classes overlap, so no "
            "linear boundary can separate them perfectly. **Result.** 200 malignant / 800 benign samples."
        )

    # ---------------- Part 2 ----------------
    with tabs[1]:
        ui.render_part_block(
            "Part 2 · Learn the model by maximum likelihood",
            aim="Estimate priors, class means and the pooled covariance from the training set.",
            theory="For Gaussian classes the ML estimates are simple averages; the covariance is pooled because both classes share one C.",
            equations=[
                r"\hat p_i = \frac{N_i}{N},\qquad \hat{\mu}_i = \frac{1}{N_i}\sum_{x\in C_i} x",
                r"\hat{\mathbf C} = \frac{1}{N}\sum_{i=1}^{2}\sum_{x\in C_i}(x-\hat{\mu}_i)(x-\hat{\mu}_i)^T",
            ],
        )
        ui.metric_row([("p₁", fmt(clf.p1_)), ("p₂", fmt(clf.p2_)), ("N", str(len(X)))])
        st.markdown(f"**μ₁ (malignant)** = `{fmt_vec(clf.mu1_)}`")
        st.markdown(f"**μ₂ (benign)** = `{fmt_vec(clf.mu2_)}`")
        st.markdown("**Covariance C**")
        st.dataframe(pd.DataFrame(np.round(clf.C_, 6), columns=config.FEATURES, index=config.FEATURES))
        st.markdown("**Inverse covariance C⁻¹**")
        st.dataframe(pd.DataFrame(np.round(clf.Cinv_, 6), columns=config.FEATURES, index=config.FEATURES))
        ui.card(
            "**Result.** Estimates reproduce the slides: $p_1=0.2$, $\\mu_1=[0.6,0.7,0.6]$, "
            "$\\mu_2=[0.4,0.3,0.3]$, $\\mathbf C=0.04\\,\\mathbf I$."
        )

    # ---------------- Part 3 ----------------
    with tabs[2]:
        ui.render_part_block(
            "Part 3 · Costs and the threshold ξ",
            aim="Compute the minimum-risk threshold from the two cost regrets.",
            theory="Only the regrets matter: the regret of a miss (c21 − c11) and of a false alarm (c12 − c22).",
            equations=[r"\xi = \frac{p_2(c_{12}-c_{22})}{p_1(c_{21}-c_{11})}"],
        )
        ui.metric_row([
            ("Regret (miss) c21−c11", f"{c21 - c11:g}"),
            ("Regret (false alarm) c12−c22", f"{c12 - c22:g}"),
            ("ξ", fmt(clf.xi_)),
            ("ln ξ", fmt(np.log(clf.xi_))),
        ])
        st.latex(rf"\xi = \frac{{0.8}}{{0.2}}\times\frac{{3-1}}{{40-8}} = 4\times\frac{{2}}{{32}} = {clf.xi_:.4f}")
        ui.card(
            "**Result.** $\\xi = 0.25$, so $\\ln\\xi = -1.3863$. A threshold below 1 means borderline "
            "tumors are called malignant even when the evidence alone slightly favours benign."
        )


    # ---------------- Part 4 ----------------
    with tabs[3]:
        ui.render_part_block(
            "Part 4 · From Bayes rule to a perceptron",
            aim="Derive the weight vector and bias of the minimum-risk perceptron.",
            theory="With a shared covariance the log-likelihood ratio is linear in x; deciding malignant when ln Λ(x) > ln ξ gives a single perceptron.",
            equations=[
                r"\mathbf w = \mathbf C^{-1}(\mu_1 - \mu_2)",
                r"b = \tfrac12\big(\mu_2^T\mathbf C^{-1}\mu_2 - \mu_1^T\mathbf C^{-1}\mu_1\big) - \ln\xi",
                r"g(\mathbf{x}) = \mathbf w^T\mathbf{x} + b > 0 \Rightarrow \text{malignant}",
            ],
        )
        ui.metric_row([("w", fmt_vec(clf.w_)), ("b_geom", fmt(clf.b_geom_)), ("b", fmt(clf.b_))])
        st.latex(r"\mathbf w = 25\,[0.2,\,0.4,\,0.3] = [5,\,10,\,7.5]")
        st.latex(rf"b = {clf.b_geom_:.4f} - \ln({clf.xi_:.3f}) = {clf.b_:.4f}")
        ui.card(
            "**Result.** $\\mathbf w=[5,10,7.5]$ and $b=-9.489$. Border (weight 10) is the most "
            "discriminative feature; size (weight 5) the least."
        )

    # ---------------- Part 5 ----------------
    with tabs[4]:
        ui.render_part_block(
            "Part 5 · Classify the slide tumors A, B, C",
            aim="Compute the perceptron output, posterior and conditional risks for the slide tumors.",
            theory="For each tumor compute y = w·x + b, the posterior P(C₁|x) and the expected cost of each decision.",
            equations=[
                r"P(\mathcal C_1\mid x) = \sigma\big(\ln\Lambda + \ln\tfrac{p_1}{p_2}\big)",
                r"R(\text{say mal})=c_{11}P_1 + c_{12}P_2,\qquad R(\text{say ben})=c_{21}P_1 + c_{22}P_2",
            ],
        )
        Xs = slide[config.FEATURES].to_numpy()
        w, b_geom, b = clf.w_, clf.b_geom_, clf.b_
        log_lam = Xs @ w + b_geom
        yout = Xs @ w + b
        post = bayes.posterior(Xs, w, b_geom, clf.p1_, clf.p2_)
        r_mal, r_ben = bayes.conditional_risks(Xs, w, b_geom, clf.p1_, clf.p2_, c11, c12, c21, c22)
        out = pd.DataFrame({
            "tumor": slide["tumor_id"],
            "w·x": Xs @ w,
            "ln Λ": log_lam,
            "y = w·x + b": yout,
            "P(malignant|x)": post,
            "R(say mal)": r_mal,
            "R(say ben)": r_ben,
            "min-risk decision": np.where(yout > 0, "malignant", "benign"),
        })
        st.dataframe(out.round(4))
        ui.card(
            "**Observation.** A: $y=-6.86$ (benign), B: $y=+9.26$ (malignant), C: $y=+0.26$ (malignant). "
            "Tumor C has $\\ln\\Lambda=-1.125$ (evidence says benign) and only a 7.5% cancer probability, "
            "but that is above the ≈5.9% action threshold set by the costs, so minimum risk calls it "
            "malignant. Check $R(\\text{say mal}) < R(\\text{say ben})$ for C."
        )


    # ---------------- Part 6 ----------------
    with tabs[5]:
        ui.render_part_block(
            "Part 6 · Test on 10,000 new tumors",
            aim="Evaluate the classifier on the test set and compare with theoretical values.",
            theory="The test set is an ordinary random sample, so results are close to (not equal to) the theory. d² = (μ₁−μ₂)ᵀC⁻¹(μ₁−μ₂) = 7.25.",
            equations=[
                r"P(\text{miss}) = \Phi\!\left(\frac{\ln\xi - d^2/2}{d}\right)",
                r"P(\text{false alarm}) = 1 - \Phi\!\left(\frac{\ln\xi + d^2/2}{d}\right)",
            ],
        )
        ev = clf.evaluate(Xt, yt)
        th = clf.theoretical()
        ui.metric_row([
            ("Caught", str(ev["caught"])),
            ("Missed", str(ev["missed"])),
            ("False alarms", str(ev["false_alarms"])),
            ("Cleared", str(ev["cleared"])),
        ])
        st.markdown(f"**Squared Mahalanobis distance** d² = `{fmt(th['d2'], 3)}`")
        comp = pd.DataFrame({
            "metric": ["miss rate", "false-alarm rate", "error rate", "avg cost"],
            "test data": [ev["miss_rate"], ev["false_alarm_rate"], ev["error_rate"], ev["avg_cost"]],
            "theory": [th["miss_rate"], th["false_alarm_rate"], th["error_rate"], th["avg_cost"]],
        })
        st.dataframe(comp.round(4))
        st.plotly_chart(visualization.prediction_distribution(yt, clf.predict(Xt)), width="stretch")
        ui.card(
            "**Result.** Test data: miss rate 0.0265, false-alarm rate 0.2023, error rate 0.1671, "
            "average cost 2.8932 — very close to theoretical miss 0.0314, FA 0.2029, cost 2.9253."
        )

    # ---------------- Part 7 ----------------
    with tabs[6]:
        ui.render_part_block(
            "Part 7 · ML vs MAP vs minimum risk",
            aim="Compare the three Bayes rules and two trivial policies on the test set.",
            theory="All rules share the same weights; only the bias changes because only ln ξ changes.",
            equations=[r"b = b_{\mathrm{geom}} - \ln\xi"],
        )
        rows = []
        for name, xi_ in [("ML (ξ=1)", 1.0), ("MAP (ξ=4)", clf.p2_ / clf.p1_), ("Min risk (ξ=0.25)", clf.xi_)]:
            bias = clf.b_geom_ - np.log(xi_)
            s = metrics.scorecard(yt, bayes.predict(Xt, clf.w_, bias), (c11, c12, c21, c22))
            s["rule"], s["bias"] = name, bias
            rows.append(s)
        for name, const in [("Everyone benign", 0), ("Everyone malignant", 1)]:
            s = metrics.scorecard(yt, np.full(len(yt), const), (c11, c12, c21, c22))
            s["rule"], s["bias"] = name, None
            rows.append(s)
        table = pd.DataFrame(rows).set_index("rule")[
            ["bias", "caught", "missed", "false_alarms", "cleared", "miss_rate", "false_alarm_rate", "error_rate", "avg_cost"]
        ]
        st.dataframe(table.round(4))
        plot_df = table.rename(columns={"error_rate": "error rate", "avg_cost": "avg cost"})
        st.plotly_chart(visualization.model_comparison_bars(plot_df), width="stretch")
        ui.card(
            "**Observation.** MAP makes the fewest mistakes (it minimises error); minimum risk is the "
            "cheapest (it minimises expected cost). 'Everyone malignant' has an 80% error rate yet a "
            "competitive cost because it never pays the expensive miss cost (c21=40)."
        )

    # ---------------- Part 8 ----------------
    with tabs[7]:
        ui.render_part_block(
            "Part 8 · See the decision in one dimension",
            aim="Visualise the log-likelihood ratio and sweep the threshold to find the minimum cost.",
            theory="The perceptron squeezes three features into ln Λ(x); the three thresholds are ln ξ.",
        )
        from scipy.stats import norm
        L = clf.log_likelihood_ratio(Xt)
        thresh_list = [
            ("ML", 1.0, config.ML_COLOR),
            ("MAP", clf.p2_ / clf.p1_, config.MAP_COLOR),
            ("min risk", clf.xi_, config.RISK_COLOR),
        ]
        st.plotly_chart(visualization.log_likelihood_hist(L, yt, thresh_list, th["d2"]), width="stretch")
        ts = np.linspace(-6, 6, 1201)
        emp_cost = [metrics.scorecard(yt, (L > t).astype(int), (c11, c12, c21, c22))["avg_cost"] for t in ts]
        d2 = th["d2"]
        d = np.sqrt(d2)
        miss_t = norm.cdf((ts - d2 / 2) / d)
        fa_t = 1 - norm.cdf((ts + d2 / 2) / d)
        th_cost = c11 * clf.p1_ * (1 - miss_t) + c21 * clf.p1_ * miss_t + c12 * clf.p2_ * fa_t + c22 * clf.p2_ * (1 - fa_t)
        st.plotly_chart(visualization.cost_vs_threshold(ts, emp_cost, th_cost, clf.xi_), width="stretch")
        ui.card(
            "**Result.** The empirical cost curve (test data) tracks the theoretical curve, and both reach "
            "their minimum at $\\ln\\xi = \\ln 0.25 = -1.386$."
        )


    # ---------------- Part 9 ----------------
    with tabs[8]:
        ui.render_part_block(
            "Part 9 · Cross-checks (reusable class + LDA)",
            aim="Verify the reusable class against scikit-learn's LDA.",
            theory="sklearn's LDA fits the same Gaussian model but uses the MAP rule; its intercept differs from our minimum-risk bias by the cost factor ln 16.",
            equations=[r"b_{\text{min risk}} = b_{\text{LDA}} + \ln\frac{c_{21}-c_{11}}{c_{12}-c_{22}} = b_{\text{LDA}} + \ln 16"],
        )
        lda_p = B["lda"]
        ui.metric_row([
            ("Bayes w", fmt_vec(clf.w_)),
            ("Bayes b", fmt(clf.b_)),
            ("LDA w", fmt_vec(lda_p["w"])),
            ("LDA intercept", fmt(lda_p["b"])),
            ("LDA b + ln 16", fmt(lda_p["b"] + np.log(16))),
        ])
        st.markdown(
            f"**Class test avg cost** = `{fmt(clf.evaluate(Xt, yt)['avg_cost'])}` · "
            f"**theory** = `{fmt(clf.theoretical()['avg_cost'])}`"
        )
        ui.card(
            "**Result.** LDA weights equal ours; the intercept $b_{LDA}=-12.2613$ plus $\\ln 16$ equals "
            f"our minimum-risk bias $b={clf.b_:.4f}$ — the difference is exactly the cost factor."
        )

    # ---------------- Part 10 ----------------
    with tabs[9]:
        ui.render_part_block(
            "Part 10 · Designed vs learned perceptron",
            aim="Compare the theoretically designed perceptron with one learned by Rosenblatt's rule.",
            theory="The designed perceptron uses the Gaussian model; the learned one adjusts weights from labelled examples via the error-correction rule.",
            equations=[r"\mathbf w(n+1) = \mathbf w(n) + \eta\, d(n)\,\mathbf x(n)\ \ \text{(on a mistake)}"],
        )
        wl, mistakes = perceptron.perceptron_train(X, y, epochs=40, eta=1.0, seed=0)
        b_l, w_l = perceptron.perceptron_weights(wl)
        s_l = metrics.scorecard(yt, perceptron.perceptron_predict(Xt, wl), (c11, c12, c21, c22))
        cA, cB = st.columns(2)
        with cA:
            st.markdown(f"**Learned w** = `{fmt_vec(w_l, 3)}` · **b** = `{fmt(b_l, 3)}`")
            st.markdown(f"**Direction (learned)** = `{fmt_vec(w_l / np.linalg.norm(w_l), 3)}`")
            st.markdown(f"**Direction (designed)** = `{fmt_vec(clf.w_ / np.linalg.norm(clf.w_), 3)}`")
        with cB:
            st.markdown(f"**Test error rate** = `{s_l['error_rate']:.3f}`")
            st.markdown(f"**Test avg cost** = `{s_l['avg_cost']:.3f}`")
            st.markdown(f"**Mistakes in final epoch** = `{mistakes[-1]}` (never reaches zero)")
        st.plotly_chart(visualization.mistakes_per_epoch(mistakes), width="stretch")
        ui.card(
            "**Observation.** The mistakes never reach zero because the overlapping classes are not linearly "
            "separable. The learned weights point in a broadly similar direction to $[5,10,7.5]$ but are "
            "not optimal and know nothing about priors or costs. See the **Perceptron Laboratory** for an "
            "interactive version."
        )


# ================================================================ DATASET EXPLORER
def _display_dataset_info(df: pd.DataFrame, name: str) -> None:
    has_label = config.LABEL_COL in df.columns
    info = data_loader.dataset_summary(df)
    st.markdown(f"**{name}** — `{df.shape[0]}` rows × `{df.shape[1]}` columns")
    ui.metric_row([
        ("Rows", str(df.shape[0])),
        ("Columns", str(df.shape[1])),
        ("Missing values", str(info["missing"])),
    ])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Column info**")
        st.dataframe(pd.DataFrame({"dtype": info["dtypes"]}).rename_axis("column"))
        st.markdown("**First 5 rows**")
        st.dataframe(df.head())
    with c2:
        if has_label:
            st.markdown("**Class distribution**")
            st.dataframe(df[config.LABEL_COL].value_counts().rename("count"))
        st.markdown("**Last 5 rows**")
        st.dataframe(df.tail())
        st.markdown("**Statistical summary**")
        st.dataframe(df[config.FEATURES].describe().T)


def render_dataset_explorer(B: dict) -> None:
    ui.render_title()
    ui.section_header("Dataset Explorer")

    source = st.radio("Data source", ["tumor_train.csv", "tumor_test.csv", "slide_tumors.csv", "Upload CSV"], horizontal=True)

    df = None
    name = source
    if source == "tumor_train.csv":
        df, name = B["train"], "tumor_train.csv"
    elif source == "tumor_test.csv":
        df, name = B["test"], "tumor_test.csv"
    elif source == "slide_tumors.csv":
        df, name = B["slide"], "slide_tumors.csv"
    else:
        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            norm, errors = data_loader.validate_uploaded_dataset(raw)
            if errors:
                for e in errors:
                    st.error(e)
            else:
                df, name = norm, uploaded.name
                st.success("Dataset validated: required features and label column present.")

    if df is None:
        st.info("Select a dataset or upload a CSV to begin.")
        return

    has_label = config.LABEL_COL in df.columns
    _display_dataset_info(df, name)

    ui.section_header("Visualizations")
    tab_a, tab_b, tab_c = st.tabs(["Univariate", "Class distributions", "Relationships"])
    with tab_a:
        st.plotly_chart(visualization.feature_histograms(df), width="stretch")
    with tab_b:
        if has_label:
            feat = st.selectbox("Feature", config.FEATURES, key="dist_feat")
            st.plotly_chart(visualization.feature_dist_by_class(df, feat), width="stretch")
        else:
            st.info("This dataset has no label column, so class distributions are unavailable.")
    with tab_c:
        if has_label:
            st.plotly_chart(visualization.pairwise_scatter(df), width="stretch")
            st.plotly_chart(visualization.correlation_heatmap(df), width="stretch")
        else:
            st.info("This dataset has no label column; showing the correlation heatmap only.")
            st.plotly_chart(visualization.correlation_heatmap(df), width="stretch")


# ================================================================ PLAYGROUND
def render_playground(B: dict) -> None:
    ui.render_title()
    ui.section_header("Classifier Playground")
    clf = B["clf"]
    Xt, yt = B["Xt"], B["yt"]
    st.markdown("Adjust priors, costs, weights and bias — every result below updates live.")

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            p1 = st.slider("Malignant prior p₁", 0.01, 0.99, float(clf.p1_), 0.01)
            p2 = 1.0 - p1
            st.caption(f"p₂ = 1 − p₁ = **{p2:.4f}**")
        with c2:
            c11 = st.number_input("c₁₁ · correct malignant", value=float(clf.c11), step=1.0)
            c12 = st.number_input("c₁₂ · false alarm", value=float(clf.c12), step=1.0)
        with c3:
            c21 = st.number_input("c₂₁ · miss", value=float(clf.c21), step=1.0)
            c22 = st.number_input("c₂₂ · correct benign", value=float(clf.c22), step=1.0)

        override_xi = st.checkbox("Override ξ manually")
        xi_manual = None
        if override_xi:
            xi_manual = st.number_input("ξ (manual)", value=float(clf.xi_), min_value=0.0001, step=0.05)

        w = list(map(float, clf.w_))
        edit_w = st.checkbox("Edit feature weights")
        if edit_w:
            wc = st.columns(3)
            w[0] = wc[0].number_input("w · size", value=float(clf.w_[0]), step=0.5)
            w[1] = wc[1].number_input("w · border", value=float(clf.w_[1]), step=0.5)
            w[2] = wc[2].number_input("w · growth", value=float(clf.w_[2]), step=0.5)
        w = np.asarray(w, float)

        b_geom = float(clf.b_geom_)
        edit_bgeom = st.checkbox("Edit geometric bias")
        if edit_bgeom:
            b_geom = st.number_input("b_geom", value=float(clf.b_geom_), step=0.5)

    costs = (c11, c12, c21, c22)
    try:
        xi_val = xi_manual if override_xi else bayes.compute_threshold(p1, p2, c11, c12, c21, c22)
        b = bayes.compute_bias(b_geom, xi_val)
    except ValueError as exc:
        st.error(f"Invalid parameters: {exc}")
        return

    y_pred = bayes.predict(Xt, w, b)
    emp = metrics.scorecard(yt, y_pred, costs)
    cls = metrics.classification_metrics(yt, y_pred)
    theo = metrics.theoretical_performance(clf.mu1_, clf.mu2_, clf.Cinv_, p1, p2, costs, xi_val)

    ui.section_header("Derived quantities")
    ui.metric_row([
        ("ξ", fmt(xi_val)),
        ("ln ξ", fmt(np.log(xi_val))),
        ("b_geom", fmt(b_geom)),
        ("b = b_geom − ln ξ", fmt(b)),
    ])
    st.latex(rf"\xi = \frac{{p_2}}{{p_1}}\cdot\frac{{c_{{12}}-c_{{22}}}}{{c_{{21}}-c_{{11}}}} = \frac{{{p2:.4f}}}{{{p1:.4f}}}\cdot\frac{{{c12 - c22:g}}}{{{c21 - c11:g}}} = {xi_val:.4f}")
    st.latex(rf"b = {b_geom:.4f} - \ln({xi_val:.4f}) = {b:.4f}")
    st.markdown(f"**Discriminant** `g(x) = x · {fmt_vec(w)} + {fmt(b)}` — `g(x) > 0` ⇒ malignant.")

    ui.section_header("Performance on the test set")
    ui.metric_row([
        ("Caught", str(emp["caught"])),
        ("Missed", str(emp["missed"])),
        ("False alarms", str(emp["false_alarms"])),
        ("Cleared", str(emp["cleared"])),
    ])
    ui.metric_row([
        ("Miss rate", fmt(emp["miss_rate"])),
        ("False-alarm rate", fmt(emp["false_alarm_rate"])),
        ("Error rate", fmt(emp["error_rate"])),
        ("Average cost", fmt(emp["avg_cost"])),
        ("Accuracy", fmt(cls["accuracy"])),
    ])

    col_hm, col_bd = st.columns([1, 1])
    with col_hm:
        st.plotly_chart(visualization.confusion_heatmap(emp, costs), width="stretch")
    with col_bd:
        st.plotly_chart(
            visualization.decision_boundary_2d(B["test"], "size", "border", w, b),
            width="stretch",
        )

    ui.section_header("How the miss cost c₂₁ changes the classifier")
    st.markdown(
        "Raising $c_{21}$ (the cost of missing a cancer) **decreases ξ**, which **increases** the bias "
        "$b = b_{geom} - \\ln\\xi$ and shifts the boundary so that a **larger** region is classified as "
        "malignant. More malignant tumors are caught, but more benign tumors are flagged (false alarms). "
        "Lowering $c_{21}$ does the opposite. The minimum-risk rule therefore trades the two kinds of "
        "mistake according to their relative cost."
    )
    st.caption(
        f"Theoretical reference under this configuration: miss rate {theo['miss_rate']:.4f}, "
        f"false-alarm rate {theo['false_alarm_rate']:.4f}, average risk {theo['avg_cost']:.4f}."
    )


# ================================================================ MODEL EVALUATION
def render_evaluation(B: dict) -> None:
    ui.render_title()
    ui.section_header("Model Evaluation")
    clf = B["clf"]
    Xt, yt, test = B["Xt"], B["yt"], B["test"]
    costs = (clf.c11, clf.c12, clf.c21, clf.c22)

    y_pred = clf.predict(Xt)
    emp = metrics.scorecard(yt, y_pred, costs)
    cls = metrics.classification_metrics(yt, y_pred)

    ui.metric_row([
        ("Accuracy", fmt(cls["accuracy"])),
        ("Precision", fmt(cls["precision"])),
        ("Recall / sensitivity", fmt(cls["recall"])),
        ("Specificity", fmt(cls["specificity"])),
        ("F1-score", fmt(cls["f1"])),
    ])
    ui.metric_row([
        ("Miss rate", fmt(emp["miss_rate"])),
        ("False-alarm rate", fmt(emp["false_alarm_rate"])),
        ("Error rate", fmt(emp["error_rate"])),
        ("Average cost", fmt(emp["avg_cost"])),
        ("Caught / Missed / FA / Cleared",
         f"{emp['caught']} / {emp['missed']} / {emp['false_alarms']} / {emp['cleared']}"),
    ])

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(visualization.confusion_heatmap(emp, costs), width="stretch")
    with col2:
        scores = clf.log_likelihood_ratio(Xt)
        st.plotly_chart(visualization.roc_curve_fig(yt, scores), width="stretch")

    st.plotly_chart(visualization.prediction_distribution(yt, y_pred), width="stretch")

    ui.section_header("Feature distributions (by true class)")
    feat = st.selectbox("Feature", config.FEATURES, key="eval_feat")
    st.plotly_chart(visualization.feature_dist_by_class(test, feat), width="stretch")

    ui.section_header("Model comparison (all rules)")
    rows = []
    for name, xi_ in [("ML (ξ=1)", 1.0), ("MAP (ξ=4)", clf.p2_ / clf.p1_), ("Min risk (ξ=0.25)", clf.xi_)]:
        bias = clf.b_geom_ - np.log(xi_)
        s = metrics.scorecard(yt, bayes.predict(Xt, clf.w_, bias), costs)
        s["rule"], s["bias"] = name, bias
        rows.append(s)
    for name, const in [("Everyone benign", 0), ("Everyone malignant", 1)]:
        s = metrics.scorecard(yt, np.full(len(yt), const), costs)
        s["rule"], s["bias"] = name, None
        rows.append(s)
    table = pd.DataFrame(rows).set_index("rule")
    st.dataframe(table[["bias", "caught", "missed", "false_alarms", "cleared", "miss_rate", "false_alarm_rate", "error_rate", "avg_cost"]].round(4))
    plot_df = table.rename(columns={"error_rate": "error rate", "avg_cost": "avg cost"})
    st.plotly_chart(visualization.model_comparison_bars(plot_df), width="stretch")

    ui.card(
        "**Note.** Classification accuracy and average classification cost are different criteria: accuracy "
        "counts mistakes equally, while average cost weights each outcome by its cost-matrix entry. The "
        "minimum-risk rule optimises the latter, not the former."
    )


# ================================================================ DECISION BOUNDARY
def render_decision_boundary(B: dict) -> None:
    ui.render_title()
    ui.section_header("Decision Boundary Visualizer")
    clf = B["clf"]
    test = B["test"]

    c1, c2, c3 = st.columns(3)
    with c1:
        x_feat = st.selectbox("X-axis feature", config.FEATURES, index=0)
    with c2:
        remaining = [f for f in config.FEATURES if f != x_feat]
        y_feat = st.selectbox("Y-axis feature", remaining, index=0)
    third = [f for f in config.FEATURES if f not in (x_feat, y_feat)][0]
    with c3:
        fixed_val = st.number_input(f"Fix {third} at", value=float(test[third].mean()), step=0.05)

    with st.expander("Priors & costs", expanded=True):
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            p1 = st.slider("Malignant prior p₁", 0.01, 0.99, float(clf.p1_), 0.01)
            p2 = 1.0 - p1
            show_all = st.checkbox("Overlay ML & MAP boundaries", value=True)
        with cc2:
            c11 = st.number_input("c₁₁", value=float(clf.c11), step=1.0, key="db_c11")
            c12 = st.number_input("c₁₂", value=float(clf.c12), step=1.0, key="db_c12")
        with cc3:
            c21 = st.number_input("c₂₁", value=float(clf.c21), step=1.0, key="db_c21")
            c22 = st.number_input("c₂₂", value=float(clf.c22), step=1.0, key="db_c22")

    try:
        xi = bayes.compute_threshold(p1, p2, c11, c12, c21, c22)
    except ValueError as exc:
        st.error(f"Invalid parameters: {exc}")
        return

    b_minrisk = bayes.compute_bias(clf.b_geom_, xi)
    b_ml = clf.b_geom_
    b_map = clf.b_geom_ - np.log(p2 / p1)

    boundaries = [("min risk", b_minrisk, config.RISK_COLOR)]
    if show_all:
        boundaries = [("min risk", b_minrisk, config.RISK_COLOR),
                      ("ML", b_ml, config.ML_COLOR),
                      ("MAP", b_map, config.MAP_COLOR)]

    ui.metric_row([
        ("ξ (min risk)", fmt(xi)),
        ("b min risk", fmt(b_minrisk)),
        ("b ML", fmt(b_ml)),
        ("b MAP", fmt(b_map)),
    ])

    fig2d = visualization.decision_boundary_multi(
        test, x_feat, y_feat, clf.w_, boundaries, fixed_vals={third: fixed_val}
    )
    st.plotly_chart(fig2d, width="stretch")
    st.plotly_chart(visualization.decision_boundary_3d(test, clf.w_, b_minrisk), width="stretch")

    st.info(
        "A 2D projection of a three-feature classifier shows only one **slice** of the boundary: the third "
        f"feature (`{third}`) is fixed at `{fixed_val:.3f}`. The complete boundary is the plane "
        "`g(x) = w·x + b = 0` in 3D (shown above)."
    )


# ================================================================ PERCEPTRON LAB
def render_perceptron_lab(B: dict) -> None:
    ui.render_title()
    ui.section_header("Perceptron Laboratory (Rosenblatt)")
    clf = B["clf"]
    X, y, Xt, yt = B["X"], B["y"], B["Xt"], B["yt"]
    costs = (clf.c11, clf.c12, clf.c21, clf.c22)

    c1, c2, c3 = st.columns(3)
    with c1:
        eta = st.number_input("Learning rate η", value=1.0, min_value=0.01, step=0.1)
    with c2:
        epochs = st.slider("Maximum epochs", 5, 200, 40)
    with c3:
        seed = st.number_input("Random seed", value=0, step=1)

    wl, mistakes = perceptron.perceptron_train(X, y, epochs=int(epochs), eta=float(eta), seed=int(seed))
    b_l, w_l = perceptron.perceptron_weights(wl)

    y_train = perceptron.perceptron_predict(X, wl)
    y_test = perceptron.perceptron_predict(Xt, wl)
    tr_err = float(np.mean(y_train != y))
    te_metrics = metrics.scorecard(yt, y_test, costs)

    ui.metric_row([
        ("Final bias b", fmt(b_l, 3)),
        ("Final weights w", fmt_vec(w_l, 3)),
        ("Training error", fmt(tr_err)),
        ("Test error", fmt(te_metrics["error_rate"])),
        ("Test avg cost", fmt(te_metrics["avg_cost"])),
    ])

    st.markdown(f"**Initial weights** = `{fmt_vec(np.zeros(4), 3)}` (bias + 3 features)")
    st.markdown(
        f"**Direction (learned)** = `{fmt_vec(w_l / np.linalg.norm(w_l), 3)}` · "
        f"**Direction (designed)** = `{fmt_vec(clf.w_ / np.linalg.norm(clf.w_), 3)}`"
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.plotly_chart(visualization.mistakes_per_epoch(mistakes), width="stretch")
    with col2:
        st.markdown("**Convergence status**")
        if mistakes[-1] == 0:
            st.success("Converged: zero mistakes in the final epoch.")
        else:
            st.warning(
                f"Did not converge — {mistakes[-1]} mistakes in the final epoch. The overlapping classes "
                "are not linearly separable, so Rosenblatt's rule keeps correcting forever."
            )

    ui.section_header("Learned decision boundary")
    st.plotly_chart(
        visualization.decision_boundary_2d(B["test"], "size", "border", w_l, b_l),
        width="stretch",
    )

    ui.section_header("Designed vs learned perceptron")
    st.markdown(
        "The **designed** Bayes perceptron is computed analytically from the Gaussian model "
        "($\\mathbf w = \\mathbf C^{-1}(\\mu_1-\\mu_2)$) and incorporates priors and costs through its bias. "
        "The **learned** Rosenblatt perceptron adjusts its weights sample-by-sample from labelled data only; "
        "it knows nothing about priors or costs and, because the data are not separable, it never settles to "
        "a stable solution."
    )


# ================================================================ EXTENSIONS
def render_extensions(B: dict) -> None:
    ui.render_title()
    ui.section_header("Extension Experiments")
    clf = B["clf"]
    Xt, yt = B["Xt"], B["yt"]
    test = B["test"]

    tabs = st.tabs([
        "Extension 1 · Cheaper misses",
        "Extension 2 · Screening population",
        "Extension 3 · Haykin's assumptions",
        "Extension 4 · Feature importance",
        "Extension 5 · Correlated features",
    ])

    # ---------- Extension 1 ----------
    with tabs[0]:
        st.markdown("### Extension 1 · Cheaper misses")
        st.markdown("Change $c_{21}$ from 40 to **11**; keep other costs and priors unchanged.")
        r = extensions.extension_cheaper_misses(clf, Xt, yt, new_c21=11.0)
        ui.metric_row([
            ("ξ", fmt(r["xi"])),
            ("ln ξ", fmt(np.log(r["xi"]))),
            ("Bias b", fmt(r["b"])),
            ("Miss rate", fmt(r["empirical"]["miss_rate"])),
            ("False-alarm rate", fmt(r["empirical"]["false_alarm_rate"])),
            ("Avg cost (test)", fmt(r["empirical"]["avg_cost"])),
        ])
        st.latex(rf"\xi = \frac{{0.8}}{{0.2}}\cdot\frac{{3-1}}{{11-8}} = 4\cdot\frac{{2}}{{3}} = {r['xi']:.4f}")
        st.latex(rf"b = -10.875 - \ln({r['xi']:.4f}) = {r['b']:.4f}")
        st.caption(
            f"Theoretical: miss rate {r['theoretical']['miss_rate']:.4f}, "
            f"false-alarm rate {r['theoretical']['false_alarm_rate']:.4f}, "
            f"average risk {r['theoretical']['avg_cost']:.4f}."
        )
        st.plotly_chart(
            visualization.decision_boundary_multi(
                test, "size", "border", clf.w_,
                [("original (b=-9.489)", clf.b_, config.RISK_COLOR),
                 ("cheaper misses (b=-11.856)", r["b"], config.MAP_COLOR)],
            ),
            width="stretch",
        )
        ui.card(
            "**Explanation.** Reducing the cost of missing a cancer (40 → 11) makes misses cheaper, so the "
            "classifier becomes **more conservative**: ξ rises to 2.6667 and the bias falls to −11.856, "
            "shrinking the malignant region. The miss rate rises (≈3.1% → ≈16.3%) while the false-alarm "
            "rate falls (≈20.3% → ≈4.4%)."
        )

    # ---------- Extension 2 ----------
    with tabs[1]:
        st.markdown("### Extension 2 · Screening population")
        st.markdown("Change the malignant prior to $p_1 = 0.05$ (screening); keep original costs.")
        r = extensions.extension_screening(clf, Xt, yt, p1=0.05)
        ui.metric_row([
            ("ξ", fmt(r["xi"])),
            ("ln ξ", fmt(np.log(r["xi"]))),
            ("Bias b", fmt(r["b"])),
            ("Miss rate (old test set)", fmt(r["empirical"]["miss_rate"])),
            ("False-alarm rate (old test set)", fmt(r["empirical"]["false_alarm_rate"])),
        ])
        st.latex(rf"\xi = \frac{{0.95}}{{0.05}}\cdot\frac{{2}}{{32}} = 19\cdot 0.0625 = {r['xi']:.4f}")
        st.latex(rf"b = -10.875 - \ln({r['xi']:.4f}) = {r['b']:.4f}")
        st.plotly_chart(
            visualization.decision_boundary_multi(
                test, "size", "border", clf.w_,
                [("original (b=-9.489)", clf.b_, config.RISK_COLOR),
                 ("screening (b=-11.047)", r["b"], config.ML_COLOR)],
            ),
            width="stretch",
        )
        ui.card(
            "**Explanation.** A lower malignant prior means the evidence must be far stronger before calling "
            "a tumor malignant, so the boundary shifts and the malignant region shrinks. "
            "**Important distinction:** the metrics above are evaluated on the *original* 20% malignant test "
            "set; the expected risk under the new 5% population is "
            f"**{r['new_population_risk']:.4f}** — not the old test-set average cost."
        )

    # ---------- Extension 3 ----------
    with tabs[2]:
        st.markdown("### Extension 3 · Haykin's assumptions")
        st.markdown("Set $p_1 = p_2 = 0.5$ and $c_{11}=c_{22}=0$, $c_{12}=c_{21}=1$.")
        r = extensions.extension_haykin(clf, Xt, yt)
        ui.metric_row([
            ("ξ", fmt(r["xi"])),
            ("Bias b", fmt(r["b"])),
            ("b_geom", fmt(r["b_geom"])),
            ("Error rate", fmt(r["empirical"]["error_rate"])),
            ("Avg cost (0-1)", fmt(r["empirical"]["avg_cost"])),
        ])
        st.latex(rf"\xi = \frac{{0.5}}{{0.5}}\cdot\frac{{1-0}}{{1-0}} = 1,\qquad b = b_{{\mathrm{{geom}}}} = {r['b']:.4f}")
        st.success("Verified: the minimum-risk bias equals the geometric bias — the rule reduces to ML.")
        ui.card(
            "**Explanation.** With equal priors and equal (0-1) costs, the prior and cost factors both equal 1, "
            "so ξ = 1 and b = b_geom. Minimum-risk classification reduces to the geometric / "
            "maximum-likelihood decision rule, which depends only on the class distributions."
        )


    # ---------- Extension 4 ----------
    with tabs[3]:
        st.markdown("### Extension 4 · Feature importance")
        st.markdown(
            "Increase the malignant size mean from 0.6 to 0.8 (size mean difference 0.2 → 0.4), keeping the "
            "covariance and other means unchanged. The size weight should double from 5 to 10."
        )
        r = extensions.extension_feature_importance(clf)
        st.dataframe(pd.DataFrame({
            "feature": config.FEATURES,
            "μ₁ original": r["mu1_original"],
            "μ₁ modified": r["mu1_modified"],
            "μ₂": r["mu2"],
        }).round(4))
        ui.metric_row([
            ("w (original)", fmt_vec(r["w_original"])),
            ("w (new)", fmt_vec(r["w_new"])),
        ])
        st.plotly_chart(
            visualization.weights_bar(config.FEATURES, {"Original": r["w_original"], "New": r["w_new"]}),
            width="stretch",
        )
        ui.card(
            "**Explanation.** With $\\mathbf C = 0.04\\,\\mathbf I$ the weights are $25\\times$ the mean "
            "difference. Doubling the size mean difference doubles its weight (5 → 10). A feature receives a "
            "larger weight when its class-mean separation increases, provided the covariance is unchanged."
        )

    # ---------- Extension 5 ----------
    with tabs[4]:
        st.markdown("### Extension 5 · Correlated features")
        st.markdown("Introduce correlation 0.5 between size and growth; keep the original mean difference.")
        r = extensions.extension_correlated(clf)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Modified covariance**")
            st.dataframe(pd.DataFrame(np.round(r["C_modified"], 3), columns=config.FEATURES, index=config.FEATURES))
        with col2:
            st.markdown("**Inverse covariance**")
            st.dataframe(pd.DataFrame(np.round(r["Cinv_modified"], 3), columns=config.FEATURES, index=config.FEATURES))
        ui.metric_row([
            ("w (original)", fmt_vec(r["w_original"])),
            ("w (new)", fmt_vec(r["w_new"])),
        ])
        st.plotly_chart(
            visualization.weights_bar(config.FEATURES, {"Original": r["w_original"], "New": r["w_new"]}),
            width="stretch",
        )
        st.plotly_chart(visualization.covariance_heatmap(r["C_modified"], "Correlated covariance matrix"), width="stretch")
        ui.card(
            "**Explanation.** With correlated features the weights are no longer $25\\times$ the mean "
            "difference. The inverse covariance $\\mathbf C^{-1}$ has off-diagonal terms (−16.667) that "
            "correct for the shared information between size and growth, so their weights drop (5 → 1.667 "
            "and 7.5 → 6.667) while the uncorrelated border weight stays at 10."
        )


# ================================================================ MATH REFERENCE
def render_math_reference(B: dict) -> None:
    ui.render_title()
    ui.section_header("Mathematical Reference")
    st.markdown(
        "Every equation used by the application, with its symbols, general form, practical values and "
        "interpretation. The minimum-risk classifier is a two-class Gaussian Bayes classifier with a shared "
        "covariance, written as a single perceptron."
    )
    for topic in theory_module.TOPICS:
        st.markdown(f"### {topic['title']}")
        st.markdown(topic["intro"])
        ui.render_symbols(topic["symbols"])
        st.markdown("**General formula**")
        for eq in topic["general"]:
            st.latex(eq)
        st.markdown("**With the practical's values**")
        for eq in topic["practical"]:
            st.latex(eq)
        st.markdown(f"**Interpretation.** {topic['interpretation']}")
        st.markdown("---")



if __name__ == "__main__":
    main()

