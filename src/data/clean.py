"""
src/data/clean.py

Cleaning pipeline for the CarGurus sample.
Reads: data/processed/sample_500k.parquet
Writes: data/processed/cleaned.parquet
        data/processed/train.parquet
        data/processed/val.parquet
        data/processed/test.parquet

Run:
    python -m src.data.clean

Rules:
  - Never touch raw files
  - Log row count before and after EVERY step
  - No leakage columns in output
  - Split by VIN (same car never in two splits)
  - Fit nothing on val/test (scalers/encoders come later in features.py)
"""

import polars as pl
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_PARQUET  = Path("data/processed/sample_500k.parquet")
OUT_CLEAN   = Path("data/processed/cleaned.parquet")
OUT_TRAIN   = Path("data/processed/train.parquet")
OUT_VAL     = Path("data/processed/val.parquet")
OUT_TEST    = Path("data/processed/test.parquet")

# ── Leakage columns — never allowed in output ─────────────────────────────────
LEAKAGE_COLS = [
    "savings_amount",    # derived from price
    "daysonmarket",      # unknown at prediction time
    "listed_date",       # unknown at prediction time
    "listing_id",        # identifier, not a feature
]

# ── Split fractions ───────────────────────────────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
RANDOM_SEED = 42


def log_step(df: pl.DataFrame, step: str, before: int) -> None:
    """Log how many rows a step removed."""
    after = len(df)
    removed = before - after
    pct = removed / before * 100 if before > 0 else 0
    log.info(f"  [{step}] {before:,} → {after:,}  "
             f"(removed {removed:,} = {pct:.2f}%)")


def step1_fix_types(df: pl.DataFrame) -> pl.DataFrame:
    """Cast columns to correct types; strings like '--' become null."""
    df = df.with_columns([
        # engine_cylinders comes in as string e.g. "4 cylinders" or "6"
        pl.col("engine_cylinders")
          .str.extract(r"(\d+)", 1)
          .cast(pl.Float64)
          .alias("engine_cylinders"),

        # maximum_seating comes in as "5 seats" or "5"
        pl.col("maximum_seating")
          .str.extract(r"(\d+)", 1)
          .cast(pl.Float64)
          .alias("maximum_seating"),

        # listed_date string → proper date
        pl.col("listed_date")
          .str.to_date(format="%Y-%m-%d", strict=False)
          .alias("listed_date"),

        # ensure price is Float64
        pl.col("price").cast(pl.Float64),

        # year as Int32 is enough
        pl.col("year").cast(pl.Int32),
    ])
    return df


def step2_drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    """Remove duplicate VINs — keep first occurrence."""
    return df.unique(subset=["vin"], keep="first", maintain_order=True)


def step3_domain_rules(df: pl.DataFrame) -> pl.DataFrame:
    """Drop rows that violate basic real-world constraints."""
    df = df.filter(
        (pl.col("price") >= 500) &          # no junk listings
        (pl.col("price") <= 300_000) &      # no exotic outliers (handle separately)
        (pl.col("year") >= 1990) &          # pre-1990 = noise
        (pl.col("year") <= 2021) &          # no future years
        (
            pl.col("mileage").is_null() |   # keep nulls (impute later)
            (pl.col("mileage") <= 400_000)  # drop extreme mileage
        )
    )
    return df


def step4_missingness(df: pl.DataFrame) -> pl.DataFrame:
    """Add was_missing flags, then impute within make-model-year groups."""

    # ── Columns that matter despite high missingness ──────────────────────────
    KEEP_DESPITE_NULLS = [
        "has_accidents", "frame_damaged", "fleet",
        "salvage", "theft_title", "owner_count", "mileage",
        "horsepower", "engine_displacement",
    ]

    # ── Add was_missing flag for each important column ────────────────────────
    flag_cols = [
        pl.col(c).is_null().cast(pl.Int8).alias(f"was_missing_{c}")
        for c in KEEP_DESPITE_NULLS
        if c in df.columns
    ]
    df = df.with_columns(flag_cols)

    # ── Fill boolean damage flags with False (missing = not reported) ─────────
    bool_fill = ["has_accidents", "frame_damaged", "fleet", "salvage", "theft_title"]
    df = df.with_columns([
        pl.col(c).fill_null(False) for c in bool_fill if c in df.columns
    ])

    # ── Impute numeric columns with median within make-model-year groups ───────
    for col in ["mileage", "horsepower", "engine_displacement",
                "city_fuel_economy", "highway_fuel_economy",
                "owner_count", "maximum_seating"]:
        if col not in df.columns:
            continue
        df = df.with_columns(
            pl.col(col)
              .fill_null(pl.col(col).median().over(["make_name", "model_name", "year"]))
              .alias(col)
        )
        # If group median is still null (rare make-model-year), use global median
        global_median = df[col].median()
        df = df.with_columns(
            pl.col(col).fill_null(global_median).alias(col)
        )

    # ── Drop columns >50% null that don't matter ──────────────────────────────
    threshold = 0.50
    for col in df.columns:
        null_frac = df[col].is_null().mean()
        if null_frac > threshold and col not in KEEP_DESPITE_NULLS:
            log.info(f"    Dropping {col} ({null_frac:.1%} null)")
            df = df.drop(col)

        # ── Fill categorical nulls with "Unknown" ─────────────────────────────────
    cat_cols = ["body_type", "fuel_type", "engine_type", "engine_cylinders",
                "transmission", "wheel_system", "trim_name"]
    df = df.with_columns([
        pl.col(c).fill_null("Unknown")
        for c in cat_cols if c in df.columns
    ])

    # dealer_zip: fill with 0 (will be treated as missing region)
    df = df.with_columns(
        pl.col("dealer_zip").fill_null(0)
    )

    return df


def step5_outliers(df: pl.DataFrame) -> pl.DataFrame:
    """
    IQR on log(price) within make-model-year groups.
    Drops rows outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    A $90k Porsche is fine; a $90k 2008 Corolla is not.
    """
    df = df.with_columns(
        pl.col("price").log().alias("_log_price_tmp")
    )

    df = df.with_columns([
        pl.col("_log_price_tmp")
          .quantile(0.25).over(["make_name", "model_name", "year"])
          .alias("_q1"),
        pl.col("_log_price_tmp")
          .quantile(0.75).over(["make_name", "model_name", "year"])
          .alias("_q3"),
    ])

    df = df.with_columns(
        ((pl.col("_q3") - pl.col("_q1")) * 1.5).alias("_iqr_fence")
    )

    before = len(df)
    df = df.filter(
        (pl.col("_log_price_tmp") >= pl.col("_q1") - pl.col("_iqr_fence")) &
        (pl.col("_log_price_tmp") <= pl.col("_q3") + pl.col("_iqr_fence"))
    )
    log.info(f"    IQR removed {before - len(df):,} outlier rows")

    # Drop temp columns
    df = df.drop(["_log_price_tmp", "_q1", "_q3", "_iqr_fence"])
    return df


def step6_add_log_price(df: pl.DataFrame) -> pl.DataFrame:
    """Add log_price column — this is the model target."""
    return df.with_columns(
        pl.col("price").log().alias("log_price")
    )


def step7_drop_leakage(df: pl.DataFrame) -> pl.DataFrame:
    """Drop any column that would cause data leakage."""
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    if cols_to_drop:
        log.info(f"  Dropping leakage columns: {cols_to_drop}")
    return df.drop(cols_to_drop)


def step8_vin_split(df: pl.DataFrame) -> tuple:
    """
    Split by VIN so the same car never appears in two sets.
    Strategy: shuffle unique VINs, assign 70/15/15.
    """
    # Get unique VINs and shuffle them
    vins = df["vin"].unique().to_list()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(vins)

    n = len(vins)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_vins = set(vins[:n_train])
    val_vins   = set(vins[n_train:n_train + n_val])
    test_vins  = set(vins[n_train + n_val:])

    train = df.filter(pl.col("vin").is_in(train_vins))
    val   = df.filter(pl.col("vin").is_in(val_vins))
    test  = df.filter(pl.col("vin").is_in(test_vins))

    return train, val, test


def run_pipeline() -> None:
    log.info("=" * 55)
    log.info("CLEANING PIPELINE START")
    log.info("=" * 55)

    df = pl.read_parquet(IN_PARQUET)
    log.info(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

    steps = [
        ("fix_types",      step1_fix_types),
        ("drop_dupes",     step2_drop_duplicates),
        ("domain_rules",   step3_domain_rules),
        ("missingness",    step4_missingness),
        ("outliers",       step5_outliers),
        ("log_price",      step6_add_log_price),
        ("drop_leakage",   step7_drop_leakage),
    ]

    for name, fn in steps:
        before = len(df)
        df = fn(df)
        log_step(df, name, before)

    log.info(f"\nFinal clean shape: {df.shape}")
    df.write_parquet(OUT_CLEAN, compression="zstd")
    log.info(f"Saved → {OUT_CLEAN}")

    # Split
    train, val, test = step8_vin_split(df)
    train.write_parquet(OUT_TRAIN, compression="zstd")
    val.write_parquet(OUT_VAL,   compression="zstd")
    test.write_parquet(OUT_TEST, compression="zstd")

    log.info(f"\nSplits (by VIN):")
    log.info(f"  Train: {len(train):,} rows")
    log.info(f"  Val:   {len(val):,} rows")
    log.info(f"  Test:  {len(test):,} rows")
    log.info("PIPELINE COMPLETE")


if __name__ == "__main__":
    run_pipeline()