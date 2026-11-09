"""Plotly figure builders for the dashboard (single source of visualisation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn import metrics as sk_metrics

from . import config

DARK_BG = "#0e1117"
CARD_BG = "#151c2a"
GRID = "rgba(255,255,255,0.08)"
TEXT = "#e6edf3"
MUTED = "#9aa7b5"


def base_layout(fig, title=None, height=380, xlabel=None, ylabel=None, margin=None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT, family="Segoe UI, sans-serif"),
        title=dict(text=title, font=dict(size=16)) if title else None,
        height=height,
        xaxis=dict(title=xlabel, gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(title=ylabel, gridcolor=GRID, zerolinecolor=GRID),
        margin=margin or dict(l=40, r=20, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ---------------------------------------------------------------- dataset explorer
def class_distribution_bar(df: pd.DataFrame) -> go.Figure:
    counts = df[config.LABEL_COL].value_counts().sort_index()
    names = [config.CLASS_NAMES.get(i, i) for i in counts.index]
    colors = [config.MAL_COLOR if i == 1 else config.BEN_COLOR for i in counts.index]
    fig = go.Figure(go.Bar(x=names, y=counts.values, marker_color=colors, text=counts.values, textposition="auto"))
    return base_layout(fig, title="Class distribution", height=300, ylabel="count")


def scatter_2d(df: pd.DataFrame, x: str, y: str, color_by_label=True) -> go.Figure:
    if color_by_label and config.LABEL_COL in df.columns:
        fig = px.scatter(
            df, x=x, y=y, color=df[config.LABEL_COL].map(config.CLASS_NAMES),
            color_discrete_map={"malignant": config.MAL_COLOR, "benign": config.BEN_COLOR},
            opacity=0.6,
        )
    else:
        fig = px.scatter(df, x=x, y=y, opacity=0.6)
    return base_layout(fig, title=f"{x} vs {y}", height=360, xlabel=x, ylabel=y)


def feature_histograms(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for f in config.FEATURES:
        fig.add_trace(go.Histogram(x=df[f], name=f, opacity=0.6, nbinsx=40))
    fig.update_layout(barmode="overlay")
    return base_layout(fig, title="Feature histograms", height=360, xlabel="value", ylabel="count")


def feature_dist_by_class(df: pd.DataFrame, feature: str) -> go.Figure:
    fig = go.Figure()
    for lab, name, col in [(1, "malignant", config.MAL_COLOR), (0, "benign", config.BEN_COLOR)]:
        sub = df[df[config.LABEL_COL] == lab][feature]
        fig.add_trace(go.Histogram(x=sub, name=name, opacity=0.6, nbinsx=40, marker_color=col))
    fig.update_layout(barmode="overlay")
    return base_layout(fig, title=f"{feature} distribution by class", height=360, xlabel=feature, ylabel="count")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    corr = df[config.FEATURES].corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=list(corr.columns), y=list(corr.index),
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=np.round(corr.values, 3), texttemplate="%{text}",
    ))
    return base_layout(fig, title="Feature correlation", height=360)


def covariance_heatmap(C, title="Covariance matrix", annotate=True) -> go.Figure:
    C = np.asarray(C, dtype=float)
    fig = go.Figure(go.Heatmap(
        z=C, x=config.FEATURES, y=config.FEATURES,
        colorscale="RdBu_r", zmid=0,
        text=np.round(C, 4) if annotate else None,
        texttemplate="%{text}" if annotate else None,
    ))
    return base_layout(fig, title=title, height=360)

# ---------------------------------------------------------------- evaluation
def confusion_heatmap(counts: dict, costs) -> go.Figure:
    """Confusion matrix heatmap with counts and cost contributions annotated."""
    c11, c12, c21, c22 = costs
    tp, fn = counts["caught"], counts["missed"]
    fp, tn = counts["false_alarms"], counts["cleared"]
    z = [[tn, fn], [fp, tp]]
    cz = [[c22 * tn, c21 * fn], [c12 * fp, c11 * tp]]
    text = [[f"{tn}", f"{fn}"], [f"{fp}", f"{tp}"]]
    cost_text = [[f"cost {cz[0][0]:.1f}", f"cost {cz[0][1]:.1f}"],
                 [f"cost {cz[1][0]:.1f}", f"cost {cz[1][1]:.1f}"]]
    fig = go.Figure(go.Heatmap(
        z=z, x=["benign (actual)", "malignant (actual)"],
        y=["benign (pred)", "malignant (pred)"],
        colorscale="Blues", zmin=0,
        text=text, texttemplate="%{text}", textfont=dict(size=16),
        colorbar=dict(title="count"),
    ))
    fig.update_layout(annotations=[
        dict(x=0, y=0, text=cost_text[0][0], showarrow=False, font=dict(color=TEXT)),
        dict(x=1, y=0, text=cost_text[0][1], showarrow=False, font=dict(color=TEXT)),
        dict(x=0, y=1, text=cost_text[1][0], showarrow=False, font=dict(color=TEXT)),
        dict(x=1, y=1, text=cost_text[1][1], showarrow=False, font=dict(color=TEXT)),
    ])
    return base_layout(fig, title="Confusion matrix (with cost contribution)", height=340)


def roc_curve_fig(y_true, scores) -> go.Figure:
    fpr, tpr, _ = sk_metrics.roc_curve(y_true, scores)
    auc = sk_metrics.roc_auc_score(y_true, scores)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})",
                             line=dict(color=config.ACCENT, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                             line=dict(color=MUTED, dash="dash")))
    fig.update_xaxes(range=[0, 1]); fig.update_yaxes(range=[0, 1])
    return base_layout(fig, title="ROC curve", height=360, xlabel="False-positive rate", ylabel="True-positive rate")


def prediction_distribution(y_true, y_pred) -> go.Figure:
    tp = (y_pred == 1) & (y_true == 1)
    tn = (y_pred == 0) & (y_true == 0)
    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)
    vals = [int(tp.sum()), int(tn.sum()), int(fp.sum()), int(fn.sum())]
    fig = go.Figure(go.Bar(
        x=["Caught (TP)", "Cleared (TN)", "False alarm (FP)", "Missed (FN)"],
        y=vals, text=vals, textposition="auto",
        marker_color=[config.MAL_COLOR, config.BEN_COLOR, config.MAP_COLOR, config.ML_COLOR],
    ))
    return base_layout(fig, title="Prediction outcomes", height=340, ylabel="count")


def model_comparison_bars(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["error rate"], name="error rate",
                         marker_color=config.PALETTE[: len(df)]))
    fig.add_trace(go.Bar(x=df.index, y=df["avg cost"], name="avg cost",
                         marker_color=config.RISK_COLOR, visible="legendonly"))
    fig.update_layout(barmode="group")
    return base_layout(fig, title="Error rate vs average cost", height=380, ylabel="value")


def weights_bar(features, series: dict) -> go.Figure:
    fig = go.Figure()
    for name, w in series.items():
        fig.add_trace(go.Bar(x=features, y=np.asarray(w, float), name=name))
    fig.update_layout(barmode="group")
    return base_layout(fig, title="Feature weights comparison", height=360, ylabel="weight")


def log_likelihood_hist(L, y, thresholds, d2: float) -> go.Figure:
    """Histogram of ln Lambda by class with threshold markers.

    ``thresholds`` is a list of ``(name, xi, color)`` tuples.
    """
    from scipy.stats import norm
    fig = go.Figure()
    g = np.linspace(float(np.min(L)) - 1, float(np.max(L)) + 1, 400)
    d = np.sqrt(d2)
    for lab, name, col in [(0, "benign", config.BEN_COLOR), (1, "malignant", config.MAL_COLOR)]:
        fig.add_trace(go.Histogram(x=L[y == lab], name=name, opacity=0.55, nbinsx=60,
                                   marker_color=col, histnorm="probability density"))
        mu = -d2 / 2 if lab == 0 else d2 / 2
        fig.add_trace(go.Scatter(x=g, y=norm.pdf(g, mu, d), name=f"{name} (theory)",
                                 line=dict(color=col, width=2)))
    for name, xi, col in thresholds:
        fig.add_vline(x=np.log(xi), line_color=col, line_width=2,
                      annotation_text=f"{name}: ln ξ = {np.log(xi):.2f}")
    return base_layout(fig, title="Log-likelihood ratio by class", height=400,
                       xlabel="ln Λ(x) = w·x + b_geom", ylabel="density")


def cost_vs_threshold(ts, emp_cost, theo_cost, xi) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=emp_cost, mode="lines", name="test data",
                             line=dict(color=MUTED, width=3)))
    fig.add_trace(go.Scatter(x=ts, y=theo_cost, mode="lines", name="theory",
                             line=dict(color=config.RISK_COLOR, width=2)))
    fig.add_vline(x=np.log(xi), line_color=config.RISK_COLOR, line_dash="dash",
                  annotation_text=f"ln ξ = {np.log(xi):.2f}")
    best = float(ts[int(np.argmin(theo_cost))])
    fig.add_vline(x=best, line_color=config.ML_COLOR, line_dash="dot",
                  annotation_text=f"min {best:.2f}")
    return base_layout(fig, title="Average cost vs decision threshold", height=400,
                       xlabel="threshold on ln Λ", ylabel="cost per patient")


def mistakes_per_epoch(mistakes) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(mistakes) + 1)), y=mistakes, mode="lines+markers",
        marker=dict(size=5), line=dict(color=config.ML_COLOR, width=2), name="mistakes"))
    return base_layout(fig, title="Perceptron mistakes per epoch", height=340,
                       xlabel="epoch", ylabel="training mistakes")


def pairwise_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter-matrix of the three features, coloured by class."""
    d = df.copy()
    d["class"] = d[config.LABEL_COL].map(config.CLASS_NAMES)
    fig = px.scatter_matrix(
        d, dimensions=config.FEATURES, color="class",
        color_discrete_map={"malignant": config.MAL_COLOR, "benign": config.BEN_COLOR},
        opacity=0.5,
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                      font=dict(color=TEXT), height=560)
    return fig


def decision_boundary_2d(df, x_feat, y_feat, w, b, fixed_vals=None, height=440) -> go.Figure:
    """2D projection of the 3-feature decision boundary.

    The remaining (third) feature is fixed at ``fixed_vals[feature]`` (default:
    its empirical mean), so this shows only one slice of the full boundary.
    """
    xidx = config.FEATURES.index(x_feat)
    yidx = config.FEATURES.index(y_feat)
    kidx = [i for i in range(3) if i not in (xidx, yidx)][0]
    kname = config.FEATURES[kidx]
    xk = (fixed_vals or {}).get(kname, float(df[kname].mean()))
    X = df[config.FEATURES].to_numpy()
    pad = 0.06
    xs = np.linspace(X[:, xidx].min() - pad, X[:, xidx].max() + pad, 220)
    ys = np.linspace(X[:, yidx].min() - pad, X[:, yidx].max() + pad, 220)
    XX, YY = np.meshgrid(xs, ys)
    GG = w[xidx] * XX + w[yidx] * YY + w[kidx] * xk + b

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=(GG > 0).astype(int), zmin=0, zmax=1,
        colorscale=[[0, config.BEN_COLOR], [1, config.MAL_COLOR]],
        showscale=False, opacity=0.25,
    ))
    for lab, name, col in [(0, "benign", config.BEN_COLOR), (1, "malignant", config.MAL_COLOR)]:
        sub = df[df[config.LABEL_COL] == lab]
        fig.add_trace(go.Scatter(x=sub[x_feat], y=sub[y_feat], mode="markers", name=name,
                                 marker=dict(color=col, size=5, opacity=0.55)))
    if abs(w[yidx]) > 1e-9:
        y_line = -(w[xidx] * xs + w[kidx] * xk + b) / w[yidx]
        fig.add_trace(go.Scatter(x=xs, y=y_line, mode="lines", name="boundary (g=0)",
                                 line=dict(color=config.ACCENT, width=2, dash="dash")))
    return base_layout(fig, title=f"Decision boundary · {x_feat} vs {y_feat} ({kname} = {xk:.3f})",
                       height=height, xlabel=x_feat, ylabel=y_feat)


def decision_boundary_3d(df, w, b, height=520) -> go.Figure:
    X = df[config.FEATURES].to_numpy()
    fig = go.Figure()
    for lab, name, col in [(0, "benign", config.BEN_COLOR), (1, "malignant", config.MAL_COLOR)]:
        sub = df[df[config.LABEL_COL] == lab]
        fig.add_trace(go.Scatter3d(x=sub["size"], y=sub["border"], z=sub["growth"],
                                   mode="markers", name=name,
                                   marker=dict(color=col, size=2.5, opacity=0.55)))
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 24)
    ys = np.linspace(X[:, 1].min(), X[:, 1].max(), 24)
    XX, YY = np.meshgrid(xs, ys)
    if abs(w[2]) > 1e-9:
        ZZ = -(w[0] * XX + w[1] * YY + b) / w[2]
        fig.add_trace(go.Surface(x=xs, y=ys, z=ZZ, opacity=0.4, showscale=False,
                                 colorscale=[[0, config.ACCENT], [1, config.ACCENT]]))
    fig.update_layout(scene=dict(xaxis_title="size", yaxis_title="border", zaxis_title="growth",
                                 bgcolor=DARK_BG))
    return base_layout(fig, title="3D decision boundary (plane g(x)=0)", height=height)


def decision_boundary_multi(df, x_feat, y_feat, w, boundaries, fixed_vals=None, height=460) -> go.Figure:
    """2D projection of several decision boundaries (ML/MAP/min-risk) at once.

    ``boundaries`` is a list of ``(name, b, color)``. The third feature is fixed
    at ``fixed_vals[feature]`` (default: its empirical mean).
    """
    xidx = config.FEATURES.index(x_feat)
    yidx = config.FEATURES.index(y_feat)
    kidx = [i for i in range(3) if i not in (xidx, yidx)][0]
    kname = config.FEATURES[kidx]
    xk = (fixed_vals or {}).get(kname, float(df[kname].mean()))
    X = df[config.FEATURES].to_numpy()
    pad = 0.06
    xs = np.linspace(X[:, xidx].min() - pad, X[:, xidx].max() + pad, 220)
    ys = np.linspace(X[:, yidx].min() - pad, X[:, yidx].max() + pad, 220)
    XX, YY = np.meshgrid(xs, ys)

    # Primary (min-risk) region shading using the first boundary.
    b0 = boundaries[0][1]
    GG = w[xidx] * XX + w[yidx] * YY + w[kidx] * xk + b0
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=xs, y=ys, z=(GG > 0).astype(int), zmin=0, zmax=1,
                             colorscale=[[0, config.BEN_COLOR], [1, config.MAL_COLOR]],
                             showscale=False, opacity=0.18))

    for lab, name, col in [(0, "benign", config.BEN_COLOR), (1, "malignant", config.MAL_COLOR)]:
        sub = df[df[config.LABEL_COL] == lab]
        fig.add_trace(go.Scatter(x=sub[x_feat], y=sub[y_feat], mode="markers", name=name,
                                 marker=dict(color=col, size=5, opacity=0.55)))

    if abs(w[yidx]) > 1e-9:
        for name, b, col in boundaries:
            y_line = -(w[xidx] * xs + w[kidx] * xk + b) / w[yidx]
            fig.add_trace(go.Scatter(x=xs, y=y_line, mode="lines", name=f"{name} (b={b:.3f})",
                                     line=dict(color=col, width=2, dash="dash")))
    return base_layout(fig, title=f"Decision boundaries · {x_feat} vs {y_feat} ({kname} = {xk:.3f})",
                       height=height, xlabel=x_feat, ylabel=y_feat)

