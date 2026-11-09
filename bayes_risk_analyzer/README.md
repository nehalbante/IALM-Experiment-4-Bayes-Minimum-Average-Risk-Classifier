# Bayes Risk Analyzer

**Interactive Simulation and Analysis of Bayes Minimum Average Risk Classification**

A professional, dark-themed Streamlit dashboard for **IALM Experiment 4** (M.Tech). It
implements the Bayes minimum-average-risk classifier as a single perceptron for tumor
classification (Haykin, *Neural Networks and Learning Machines*, Sec. 1.4) and reproduces
every part of the original laboratory, plus five extension experiments.

The original project files (`bayes_min_risk.py`, `make_dataset.py`, the notebooks and the
`data/` CSVs) are **not modified** — this application lives in its own folder and reads the
datasets read-only.

---

## Features

| Section | What it does |
|---|---|
| **Experiment Overview** | Slide parameters, learned model, dataset summary. |
| **Part 1 – Part 10** | Every practical part, each with Aim / Theory / equations / parameters / output / observations / result. |
| **Dataset Explorer** | Load train/test/slide or upload a CSV; dimensions, summary, histograms, class distributions, scatter, pairwise and correlation plots. |
| **Classifier Playground** | Live controls for priors, costs, weights, bias and ξ; predictions, confusion matrix, rates and cost update instantly. |
| **Model Evaluation** | Confusion heatmap, accuracy/precision/recall/F1, ROC, feature & prediction distributions, rule comparison. |
| **Decision Boundary Visualizer** | 2D / 3D decision boundary with ML/MAP/min-risk overlays and interactive priors/costs. |
| **Perceptron Laboratory** | Interactive Rosenblatt perceptron (learning rate, epochs, seed) with mistakes-per-epoch and boundaries. |
| **Extension Experiments** | The five exercises: cheaper misses, screening population, Haykin's assumptions, feature importance, correlated features. |
| **Mathematical Reference** | All 15 topics with symbols, general formula, practical values and interpretation. |

## Mathematical methodology

Two Gaussian classes (malignant = C₁, benign = C₂) share one covariance matrix **C**:

- Weights: `w = C⁻¹ (μ₁ − μ₂)`
- Geometric bias: `b_geom = ½ (μ₂ᵀC⁻¹μ₂ − μ₁ᵀC⁻¹μ₁)`
- Threshold: `ξ = (p₂ / p₁) · ((c₁₂ − c₂₂) / (c₂₁ − c₁₁))`
- Bias: `b = b_geom − ln ξ`
- Discriminant: `g(x) = wᵀx + b`, deciding **malignant** when `g(x) > 0`.

Default slide values: `p₁ = 0.2`, `p₂ = 0.8`, `μ₁ = [0.6, 0.7, 0.6]`, `μ₂ = [0.4, 0.3, 0.3]`,
`C = 0.04 I`, costs `[8, 3, 40, 1]`, hence `ξ = 0.25`, `w = [5, 10, 7.5]`, `b = −9.489`.

Reference test-set result (reproduced by the app from the data, not hardcoded):

| Metric | Test data | Theory |
|---|---|---|
| Caught / Missed / FA / Cleared | 1947 / 53 / 1618 / 6382 | — |
| Miss rate | 0.0265 | 0.0314 |
| False-alarm rate | 0.2023 | 0.2029 |
| Average cost | 2.8932 | 2.9253 |

## Project structure

```
bayes_risk_analyzer/
├── app.py                     # Streamlit entry point + all pages
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml            # dark theme + server settings
├── src/
│   ├── __init__.py
│   ├── config.py              # conventions, defaults, colors, paths
│   ├── bayes.py               # BayesMinRiskClassifier + reusable math
│   ├── metrics.py             # confusion counts, rates, cost, theory
│   ├── perceptron.py          # Rosenblatt perceptron
│   ├── lda.py                 # sklearn LDA helpers
│   ├── data_loader.py         # dataset loading + upload validation
│   ├── extensions.py          # the five extension experiments
│   ├── visualization.py       # Plotly figure builders
│   ├── theory.py              # mathematical reference content
│   └── ui.py                  # styling, cards, section templates
└── tests/
    └── test_bayes.py          # unit tests for the math
```

## Installation

```bash
# create and activate a virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

The application expects a `data/` folder containing `tumor_train.csv`,
`tumor_test.csv` and `slide_tumors.csv`. It looks **inside the app folder first**, then in
the parent project folder (where the original Colab lab keeps them).

## Launch

```bash
cd bayes_risk_analyzer
streamlit run app.py
```

Then open the printed URL (usually http://localhost:8501).

## Running the tests

```bash
cd bayes_risk_analyzer
python -m unittest discover tests -v
# or, if pytest is installed:
python -m pytest tests
```

The tests verify the threshold, bias, weights, reference test-set results, theoretical
rates, LDA relationship and all five extension experiments.

## Dataset format

| Column | Meaning |
|---|---|
| `tumor_id` | unique id (optional) |
| `size`, `border`, `growth` | normalised feature scores |
| `diagnosis` | `malignant` / `benign` (optional if `label` present) |
| `label` | `1` = malignant, `0` = benign |

Uploaded CSVs are validated for the three feature columns and a `label` (or `diagnosis`)
column. Invalid files produce a clear error and never replace the bundled datasets.

## Troubleshooting

- **`FileNotFoundError` on startup** — the app could not find `data/`. Copy or symlink the
  `data/` folder next to `app.py` (or keep it one level up, as in the original project).
- **Port already in use** — run `streamlit run app.py --server.port 8502`.
- **Charts not rendering** — ensure `plotly` is installed (`pip install plotly`).
