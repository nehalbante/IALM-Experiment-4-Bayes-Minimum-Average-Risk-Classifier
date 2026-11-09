"""Dataset loading and validation (keeps the original files read-only)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_train() -> pd.DataFrame:
    return pd.read_csv(config.find_data_dir() / "tumor_train.csv")


def load_test() -> pd.DataFrame:
    return pd.read_csv(config.find_data_dir() / "tumor_test.csv")


def load_slide() -> pd.DataFrame:
    return pd.read_csv(config.find_data_dir() / "slide_tumors.csv")


def split_features_labels(df: pd.DataFrame):
    """Return ``(X, y)`` numpy arrays using the canonical feature order."""
    X = df[config.FEATURES].to_numpy(dtype=float)
    y = df[config.LABEL_COL].to_numpy()
    return X, y


def dataset_summary(df: pd.DataFrame) -> dict:
    """Basic descriptive summary for the explorer page."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": int(df.isna().sum().sum()),
        "class_distribution": df[config.LABEL_COL].value_counts().to_dict() if config.LABEL_COL in df.columns else None,
    }


def validate_uploaded_dataset(df: pd.DataFrame):
    """Validate a user-uploaded CSV and normalise it for the app.

    Returns ``(normalized_df, errors)`` where ``errors`` is a list of human
    readable messages (empty means the dataset is usable).
    """
    errors: list[str] = []

    if df is None or df.empty:
        return None, ["The uploaded file is empty."]

    missing_features = [f for f in config.FEATURES if f not in df.columns]
    if missing_features:
        errors.append(f"Missing feature column(s): {', '.join(missing_features)}.")

    has_label = config.LABEL_COL in df.columns
    has_diagnosis = config.DIAGNOSIS_COL in df.columns
    if not has_label and not has_diagnosis:
        errors.append(f"Missing label column: expected '{config.LABEL_COL}' or '{config.DIAGNOSIS_COL}'.")
    if errors:
        return None, errors

    df = df.copy()

    # Build/validate the numeric label column.
    if not has_label:
        df[config.LABEL_COL] = np.where(df[config.DIAGNOSIS_COL].astype(str).str.lower().str.strip() == "malignant", 1, 0)

    df[config.LABEL_COL] = pd.to_numeric(df[config.LABEL_COL], errors="coerce")
    if df[config.LABEL_COL].isna().any():
        errors.append("The label column contains non-numeric values.")

    labels = set(df[config.LABEL_COL].dropna().unique())
    if not labels.issubset({0, 1}):
        errors.append(f"Labels must be 0 (benign) and 1 (malignant); found {sorted(labels)}.")

    for f in config.FEATURES:
        df[f] = pd.to_numeric(df[f], errors="coerce")
        if df[f].isna().any():
            errors.append(f"Feature '{f}' contains non-numeric values.")

    if errors:
        return None, errors
    return df, []
