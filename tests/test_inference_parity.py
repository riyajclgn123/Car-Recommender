"""
Training-serving parity: the shared feature builder must reproduce the
exact features the model was trained/evaluated on.
Needs local data, so it's skipped automatically on machines without it.
"""
from pathlib import Path
import joblib
import numpy as np
import polars as pl
import pytest
from src.models.inference import load_meta, build_feature_matrix, predict_prices

PROC = Path("data/processed")
needs_data = pytest.mark.skipif(not (PROC / "test.parquet").exists(),
                                reason="local data only")


@needs_data
def test_features_match_training():
    meta = load_meta()
    raw = pl.read_parquet(PROC / "test.parquet")
    X_new = build_feature_matrix(raw, meta)
    X_saved = pl.read_parquet(PROC / "X_test.parquet").to_numpy().astype(np.float64)
    assert X_new.shape == X_saved.shape
    assert np.allclose(X_new, X_saved, atol=1e-6)


@needs_data
def test_test_mae_matches_day6():
    meta = load_meta()
    model = joblib.load("models/lgbm_tuned.pkl")
    raw = pl.read_parquet(PROC / "test.parquet")
    pred = predict_prices(raw, model, meta)
    mae = np.mean(np.abs(pred - raw["price"].to_numpy()))
    print(f"\nTest MAE via shared inference: ${mae:,.0f}")
    assert abs(mae - 1635) < 5