"""
api/predictor.py

Loads all models once at startup and provides
inference functions for each endpoint.
"""

import numpy as np
import joblib
import json
import logging
from pathlib import Path
from functools import lru_cache

log = logging.getLogger(__name__)

MODELS = Path("models")
PROC   = Path("data/processed")


@lru_cache(maxsize=1)
def load_all_models():
    """Load models once and cache. Called at startup."""
    log.info("Loading models into memory …")
    price_model     = joblib.load(MODELS / "lgbm_tuned.pkl")
    lifestage_model = joblib.load(MODELS / "lifestage_classifier.pkl")
    lifestage_le    = joblib.load(MODELS / "lifestage_label_encoder.pkl")

    with open(MODELS / "lifestage_meta.json") as f:
        lifestage_meta = json.load(f)
    with open(PROC / "feature_meta.json") as f:
        feature_meta = json.load(f)

    log.info("All models loaded ✅")
    return (price_model, lifestage_model, lifestage_le,
            lifestage_meta, feature_meta)


def predict_price_from_specs(specs: dict) -> float:
    """
    Predict price from raw car specs.
    Applies same encoding + scaling as features.py.
    Returns price in dollars.
    """
    price_model, _, _, _, feature_meta = load_all_models()

    feat_names    = feature_meta["features"]
    scaler_params = feature_meta["scaler_params"]
    encoders      = feature_meta["encoders"]
    cat_cols      = feature_meta["categorical_cols"]

    # Build feature vector in the correct order
    row = {}

    # Encode categoricals
    for col in cat_cols:
        val = specs.get(col, "Unknown")
        enc = encoders.get(col, [])
        class_map = {c: i for i, c in enumerate(enc)}
        row[col] = class_map.get(val, -1)

    # Numeric features
    numeric_defaults = {
        "year": 2018, "mileage": 50000, "horsepower": 200,
        "engine_displacement": 2.5, "owner_count": 1,
        "city_fuel_economy": 25, "highway_fuel_economy": 30,
        "maximum_seating": 5, "latitude": 35.0, "longitude": -90.0,
    }
    for col, default in numeric_defaults.items():
        row[col] = specs.get(col, default)

    # Binary features
    binary_defaults = {
        "is_new": 0, "has_accidents": 0, "frame_damaged": 0,
        "fleet": 0, "salvage": 0, "theft_title": 0,
        "was_missing_has_accidents": 0, "was_missing_frame_damaged": 0,
        "was_missing_fleet": 0, "was_missing_salvage": 0,
        "was_missing_theft_title": 0, "was_missing_owner_count": 0,
        "was_missing_mileage": 0, "was_missing_horsepower": 0,
        "was_missing_engine_displacement": 0,
    }
    for col, default in binary_defaults.items():
        row[col] = int(specs.get(col, default))

    # Scale numeric columns
    for col, params in scaler_params.items():
        if col in row:
            row[col] = (row[col] - params["mean"]) / params["std"]

    # Build vector in feature order
    vec = np.array([row.get(f, 0) for f in feat_names],
                   dtype=np.float32).reshape(1, -1)

    log_price = float(price_model.predict(vec)[0])
    return float(np.exp(log_price))


def get_lifestage_probs(age_group: str, income_band: str,
                        household_size: int, area_type: str) -> dict:
    """Get vehicle class probabilities for a user profile."""
    _, lifestage_model, lifestage_le, lifestage_meta, _ = load_all_models()

    age_map    = lifestage_meta["age_map"]
    income_map = lifestage_meta["income_map"]
    area_map   = lifestage_meta["area_map"]

    age    = age_map.get(age_group, 1)
    income = income_map.get(income_band, 1)
    hh     = min(household_size, 4)
    area   = area_map.get(area_type, 1)

    features = np.array([[age, income, hh, area,
                          age * income, hh * area]])
    probs    = lifestage_model.predict_proba(features)[0]
    classes  = lifestage_meta["classes"]
    return dict(zip(classes, probs))