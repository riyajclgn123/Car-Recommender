"""
src/data/features.py

Feature engineering: encode categoricals, scale numerics.
Fit ONLY on train; transform train, val, test.

Run:
    python -m src.data.features

Outputs:
    data/processed/X_train.parquet, y_train.parquet
    data/processed/X_val.parquet,   y_val.parquet
    data/processed/X_test.parquet,  y_test.parquet
    data/processed/feature_meta.json  (column names + scaler params)
"""

import polars as pl
import numpy as np
import json
import logging
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC = Path("data/processed")

# ── Feature groups ────────────────────────────────────────────────────────────
# Target: log_price (NOT price — we never touch raw price as target)
TARGET = "log_price"

# Numeric features — will be StandardScaled
NUMERIC_COLS = [
    "year",
    "mileage",
    "horsepower",
    "engine_displacement",
    "owner_count",
    "city_fuel_economy",
    "highway_fuel_economy",
    "maximum_seating",
    "latitude",
    "longitude",
    # Encoded categoricals — integers after encoding, must scale for linear models
    "make_name", "model_name", "trim_name", "body_type",
    "fuel_type", "engine_type", "engine_cylinders",
    "transmission", "wheel_system",
]


# Categorical features — will be LabelEncoded
CATEGORICAL_COLS = [
    "make_name",
    "model_name",
    "trim_name",
    "body_type",
    "fuel_type",
    "engine_type",
    "engine_cylinders",
    "transmission",
    "wheel_system",
]

# Binary / flag features — keep as 0/1 integers, no scaling needed
BINARY_COLS = [
    "is_new",
    "has_accidents",
    "frame_damaged",
    "fleet",
    "salvage",
    "theft_title",
    "was_missing_has_accidents",
    "was_missing_frame_damaged",
    "was_missing_fleet",
    "was_missing_salvage",
    "was_missing_theft_title",
    "was_missing_owner_count",
    "was_missing_mileage",
    "was_missing_horsepower",
    "was_missing_engine_displacement",
]

# Columns to drop before modeling
DROP_COLS = [
    "vin",          # identifier
    "price",        # raw price — target is log_price
    "dealer_zip",   # too many levels; lat/lon already capture location
    "listed_date",  # already dropped in clean.py but guard anyway
]


def load_split(name: str) -> pl.DataFrame:
    path = PROC / f"{name}.parquet"
    df = pl.read_parquet(path)
    log.info(f"Loaded {name}: {df.shape}")
    return df


def encode_categoricals(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """
    LabelEncode each categorical column.
    Fit encoder on TRAIN only; transform val and test.
    Unknown categories in val/test get code = -1.
    """
    encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in train.columns:
            log.warning(f"  Column {col} not found, skipping")
            continue

        le = LabelEncoder()
        # Fit on train
        train_vals = train[col].fill_null("Unknown").to_list()
        le.fit(train_vals)
        encoders[col] = list(le.classes_)  # save for feature_meta

        def encode_col(series: pl.Series, le=le) -> pl.Series:
            vals = series.fill_null("Unknown").to_list()
            # Map unseen categories to -1
            class_map = {c: i for i, c in enumerate(le.classes_)}
            encoded = [class_map.get(v, -1) for v in vals]
            return pl.Series(name=series.name, values=encoded, dtype=pl.Int32)

        train = train.with_columns(encode_col(train[col]).alias(col))
        val   = val.with_columns(encode_col(val[col]).alias(col))
        test  = test.with_columns(encode_col(test[col]).alias(col))

    return train, val, test, encoders


def scale_numerics(
    train: pl.DataFrame,
    val: pl.DataFrame,
    test: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """
    StandardScale numeric columns.
    Fit scaler on TRAIN only; transform val and test.
    """
    scaler_params = {}

    for col in NUMERIC_COLS:
        if col not in train.columns:
            log.warning(f"  Column {col} not found, skipping")
            continue

        vals_train = train[col].to_numpy().astype(float)
        mean = float(np.nanmean(vals_train))
        std  = float(np.nanstd(vals_train))
        std  = std if std > 0 else 1.0  # guard against zero std

        scaler_params[col] = {"mean": mean, "std": std}

        for df_name, df in [("train", train), ("val", val), ("test", test)]:
            vals = df[col].to_numpy().astype(float)
            scaled = (vals - mean) / std
            if df_name == "train":
                train = train.with_columns(
                    pl.Series(col, scaled, dtype=pl.Float64))
            elif df_name == "val":
                val = val.with_columns(
                    pl.Series(col, scaled, dtype=pl.Float64))
            else:
                test = test.with_columns(
                    pl.Series(col, scaled, dtype=pl.Float64))

    return train, val, test, scaler_params


def build_feature_matrix(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.Series]:
    """Split into X (features) and y (target)."""
    y = df[TARGET]

    # Drop non-feature columns
    drop = [c for c in DROP_COLS + [TARGET] if c in df.columns]
    X = df.drop(drop)

    return X, y


def leakage_audit(X: pl.DataFrame) -> None:
    """
    Check for columns that could leak the target.
    Anything with 'price' or 'saving' in the name is suspicious.
    """
    log.info("  Leakage audit:")
    suspicious = [c for c in X.columns
                  if any(kw in c.lower()
                         for kw in ["price", "saving", "mmr",
                                    "market", "deal", "discount"])]
    if suspicious:
        log.warning(f"  🚨 SUSPICIOUS COLUMNS: {suspicious}")
    else:
        log.info("  ✅ No suspicious columns found")
    log.info(f"  Total features: {len(X.columns)}")
    log.info(f"  Feature names: {X.columns}")


def run() -> None:
    log.info("=" * 55)
    log.info("FEATURE ENGINEERING START")
    log.info("=" * 55)

    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test")

    # Encode categoricals (fit on train only)
    log.info("Encoding categoricals …")
    train, val, test, encoders = encode_categoricals(train, val, test)

    # Scale numerics (fit on train only)
    log.info("Scaling numerics …")
    train, val, test, scaler_params = scale_numerics(train, val, test)

    # Build X, y
    X_train, y_train = build_feature_matrix(train)
    X_val,   y_val   = build_feature_matrix(val)
    X_test,  y_test  = build_feature_matrix(test)

    # Leakage audit on final feature matrix
    log.info("Running leakage audit …")
    leakage_audit(X_train)

    # Save
    for name, X, y in [
        ("train", X_train, y_train),
        ("val",   X_val,   y_val),
        ("test",  X_test,  y_test),
    ]:
        X.write_parquet(PROC / f"X_{name}.parquet")
        y.to_frame().write_parquet(PROC / f"y_{name}.parquet")
        log.info(f"  Saved X_{name}: {X.shape}")

    # Save metadata (column names, scaler params, encoder classes)
    meta = {
        "features":       X_train.columns,
        "target":         TARGET,
        "numeric_cols":   NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "binary_cols":    BINARY_COLS,
        "scaler_params":  scaler_params,
        "encoders":       encoders,
    }
    with open(PROC / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"  Saved feature_meta.json")

    log.info("\nFEATURE ENGINEERING COMPLETE")
    log.info(f"  X_train shape: {X_train.shape}")
    log.info(f"  Features: {len(X_train.columns)}")


if __name__ == "__main__":
    run()