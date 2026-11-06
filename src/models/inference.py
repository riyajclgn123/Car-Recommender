"""
src/models/inference.py
Rebuild the model's feature matrix from cleaned columns, using ONLY what
was saved at training time (feature_meta.json). Used by batch scoring and
the API, so training and serving can't drift apart.
"""
import json
import numpy as np
import polars as pl
from pathlib import Path

META_PATH = Path("data/processed/feature_meta.json")


def load_meta(path=META_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def build_feature_matrix(df: pl.DataFrame, meta: dict) -> np.ndarray:
    # 1. Encode categoricals with the TRAINING vocabulary.
    #    Unseen categories become -1, exactly like features.py did.
    for col in meta["categorical_cols"]:
        class_map = {c: i for i, c in enumerate(meta["encoders"][col])}
        df = df.with_columns(
            pl.col(col)
              .fill_null("Unknown")
              .replace_strict(class_map, default=-1, return_dtype=pl.Float64)
        )

    # 2. Booleans / flags -> 0.0 / 1.0
    for col in meta["binary_cols"]:
        df = df.with_columns(pl.col(col).cast(pl.Float64).fill_null(0))

    # 3. Scale with the TRAINING mean and std (never recompute them here)
    for col, p in meta["scaler_params"].items():
        df = df.with_columns((pl.col(col) - p["mean"]) / p["std"])

    # 4. Exact column order the model was trained on
    return df.select(meta["features"]).to_numpy().astype(np.float64)


def predict_prices(df: pl.DataFrame, model, meta: dict) -> np.ndarray:
    """Return fair-price predictions in dollars."""
    X = build_feature_matrix(df, meta)
    return np.exp(model.predict(X))