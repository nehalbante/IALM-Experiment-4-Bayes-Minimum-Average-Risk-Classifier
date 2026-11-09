"""Central configuration for the Bayes Risk Analyzer.

Single source of truth for feature/label conventions, default parameters,
colors and filesystem paths. Import from here everywhere so that the
application stays consistent with the original lab.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- conventions
FEATURES: list[str] = ["size", "border", "growth"]
LABEL_COL: str = "label"
DIAGNOSIS_COL: str = "diagnosis"
ID_COL: str = "tumor_id"

MALIGNANT: int = 1   # C1
BENIGN: int = 0      # C2
CLASS_NAMES: dict[int, str] = {MALIGNANT: "malignant", BENIGN: "benign"}

# ---------------------------------------------------------------- defaults (slides)
DEFAULT_COSTS: dict[str, float] = {"c11": 8.0, "c12": 3.0, "c21": 40.0, "c22": 1.0}
COST_KEYS: tuple[str, ...] = ("c11", "c12", "c21", "c22")
DEFAULT_P1: float = 0.2
DEFAULT_P2: float = 0.8
DEFAULT_XI: float = 0.25

# Slide geometry (theoretical values reproduced by the training set)
SLIDE_MU1: tuple[float, ...] = (0.6, 0.7, 0.6)   # malignant
SLIDE_MU2: tuple[float, ...] = (0.4, 0.3, 0.3)   # benign
SLIDE_SIGMA: float = 0.2                          # shared covariance = 0.04 I

# ---------------------------------------------------------------- colors (match the original notebook)
MAL_COLOR: str = "#C8473A"   # malignant / red
BEN_COLOR: str = "#1F7A70"   # benign / teal
ML_COLOR: str = "#5B7C99"    # maximum-likelihood rule
MAP_COLOR: str = "#B7791F"   # maximum-a-posteriori rule
RISK_COLOR: str = "#6A4C93"  # minimum-risk rule
ACCENT: str = "#2BD9C7"

PALETTE: list[str] = ["#5B7C99", "#B7791F", "#6A4C93", "#B8C3C5", "#C8473A", "#1F7A70"]

# ---------------------------------------------------------------- paths
def project_root() -> Path:
    """Root of the application folder (bayes_risk_analyzer/)."""
    return Path(__file__).resolve().parent.parent


def find_data_dir() -> Path:
    """Locate the data/ folder.

    Looks inside the app folder first, then in the parent project folder
    (where the original Colab lab keeps ``data/``). This keeps the original
    datasets untouched while still finding them automatically.
    """
    root = project_root()
    candidates = [
        root / "data",
        root.parent / "data",
    ]
    for candidate in candidates:
        if (candidate / "tumor_train.csv").exists():
            return candidate
    # Fall back to the original location; a clear error is raised later if absent.
    return root.parent / "data"


def costs_tuple(c11: float, c12: float, c21: float, c22: float) -> tuple[float, float, float, float]:
    """Return costs in the canonical order used across the codebase.

    Convention: ``c[i][j]`` is the cost of *deciding* class ``i`` when the true
    class is ``j`` (class 1 = malignant, class 2 = benign).
    """
    return (c11, c12, c21, c22)
