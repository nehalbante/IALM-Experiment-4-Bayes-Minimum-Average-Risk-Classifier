"""Shared Streamlit UI helpers: styling, cards, section templates."""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
/* ---- base ---- */
html, body, [data-testid="stAppViewContainer"] {
    background: #0b0f17;
}
[data-testid="stAppViewContainer"] .block-container {
    padding-top: 2.2rem;
    max-width: 1180px;
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
[data-testid="stSidebar"] {
    background: #0f1420;
    border-right: 1px solid rgba(255,255,255,0.06);
}
h1, h2, h3, h4 { color: #e6edf3; letter-spacing: -0.01em; }

/* ---- title block ---- */
.app-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: 0.04em;
    background: linear-gradient(90deg, #2bd9c7 0%, #4ea1ff 60%, #a06bff 100%);
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; margin: 0;
}
.subtitle {
    color: #9aa7b5; font-size: 1.05rem; margin: 0.15rem 0 1.4rem 0;
}

/* ---- section headings ---- */
.section-title {
    font-size: 1.35rem; font-weight: 700; color: #e6edf3;
    border-left: 4px solid #2bd9c7; padding-left: 0.7rem; margin: 1.4rem 0 0.7rem 0;
}

/* ---- cards ---- */
.card {
    background: #151c2a; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1.1rem 1.3rem; margin: 0.6rem 0;
    color: #cdd7e1;
}
.metric-card {
    background: #151c2a; border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1rem 1.1rem; text-align: center; margin-bottom: 0.5rem;
}
.metric-value { font-size: 1.7rem; font-weight: 750; color: #e6edf3; line-height: 1.15; }
.metric-label { font-size: 0.78rem; color: #8a97a6; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.3rem; }

/* ---- tags ---- */
.tag { display: inline-block; padding: 0.12rem 0.6rem; border-radius: 999px;
       font-size: 0.75rem; font-weight: 600; margin-right: 0.35rem; }
.tag-mal { background: rgba(200,71,58,0.18); color: #ef8b80; }
.tag-ben { background: rgba(31,122,112,0.18); color: #56c4b5; }
.tag-info { background: rgba(91,124,153,0.2); color: #9fbed9; }

/* ---- code / tables ---- */
[data-testid="stDataFrame"] { border-radius: 10px; }
.stDataFrame, .stMarkdown table { border-radius: 10px; overflow: hidden; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def render_title() -> None:
    st.markdown('<div class="app-title">BAYES RISK ANALYZER</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Interactive Simulation and Analysis of Bayes Minimum Average Risk Classification</div>',
        unsafe_allow_html=True,
    )


def section_header(text: str) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def metric_row(items) -> None:
    """Render a responsive row of metric cards. ``items`` = list of (label, value)."""
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )


def card(body: str) -> None:
    st.markdown(f'<div class="card">{body}</div>', unsafe_allow_html=True)


def latex_equations(eqs) -> None:
    for eq in eqs:
        st.latex(eq)


def render_symbols(symbols) -> None:
    """Render a symbol legend as a compact markdown table."""
    rows = "\n".join(f"| `{s}` | {d} |" for s, d in symbols)
    st.markdown(f"| Symbol | Meaning |\n|---|---|\n{rows}")


def render_part_block(title, aim=None, theory=None, equations=None, note=None) -> None:
    """Render a consistent Aim / Theory / Equations block for the Parts pages."""
    st.markdown(f"### {title}")
    if aim:
        st.markdown(f"**Aim.** {aim}")
    if theory:
        st.markdown(f"**Theory.** {theory}")
    if equations:
        latex_equations(equations)
    if note:
        st.info(note)


def tag(text: str, kind: str = "info") -> str:
    return f'<span class="tag tag-{kind}">{text}</span>'
