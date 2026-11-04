"""
src/data/load.py
Load a sampled subset of the CarGurus raw CSV into a Parquet file.

Run:
    python -m src.data.load
"""

import polars as pl
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR   = Path("data/raw")
PROC_DIR  = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV    = RAW_DIR / "used_cars_data.csv"
OUT_PARQUET = PROC_DIR / "sample_500k.parquet"

# ── Columns we actually need (drop the rest to save RAM) ─────────────────────
# We'll audit this list during EDA tomorrow and add/remove columns.
KEEP_COLS = [
    "vin",                    # for dedup and VIN-grouped splits
    "price",                  # target (will become log_price)
    "year",                   # model year
    "make_name",
    "model_name",
    "trim_name",
    "body_type",
    "mileage",
    "fuel_type",
    "engine_type",
    "engine_cylinders",
    "engine_displacement",
    "transmission",
    "wheel_system",           # FWD / AWD / 4WD
    "horsepower",
    "maximum_seating",
    "city_fuel_economy",
    "highway_fuel_economy",
    "has_accidents",          # accident history flag
    "frame_damaged",
    "fleet",
    "salvage",
    "theft_title",
    "owner_count",
    "dealer_zip",
    "is_new",                 # new vs used flag
    "listed_date",            # for drift analysis later; drop before training
    "latitude",               # may help with regional pricing
    "longitude",
]

SAMPLE_SIZE = 500_000
RANDOM_SEED = 42


def load_sample() -> pl.DataFrame:
    """
    Read the raw CarGurus CSV with Polars lazy API,
    select only the columns we need, sample SAMPLE_SIZE rows,
    and return a DataFrame.
    """
    log.info(f"Reading {RAW_CSV} lazily …")

    # scan_csv builds a query plan — nothing is loaded yet
    lf = pl.scan_csv(
        RAW_CSV,
        infer_schema_length=10_000,   # scan first 10k rows to guess types
        ignore_errors=True,            # some rows have encoding issues
        try_parse_dates=False,         # we'll parse dates ourselves
    )

    # Keep only the columns that exist in this file
    # (guards against typos in KEEP_COLS)
    available = lf.collect_schema().names()
    cols_to_keep = [c for c in KEEP_COLS if c in available]
    missing = set(KEEP_COLS) - set(cols_to_keep)
    if missing:
        log.warning(f"Columns not found in CSV, skipping: {missing}")

    lf = lf.select(cols_to_keep)

    log.info("Collecting sample …")
    # .collect() executes the lazy plan — this is where RAM is used
    df = lf.collect()

    log.info(f"Full filtered shape: {df.shape}")

    # Sample without replacement
    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, seed=RANDOM_SEED, shuffle=True)
        log.info(f"Sampled down to {len(df):,} rows")

    return df


def save_parquet(df: pl.DataFrame, path: Path) -> None:
    df.write_parquet(path, compression="zstd")
    size_mb = path.stat().st_size / 1e6
    log.info(f"Saved → {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    df = load_sample()

    # ── Quick sanity check ───────────────────────────────────────────────────
    log.info("\n--- Schema ---")
    print(df.schema)

    log.info("\n--- First 5 rows ---")
    print(df.head())

    log.info("\n--- Null counts ---")
    print(df.null_count())

    save_parquet(df, OUT_PARQUET)
    log.info("Done. Load tomorrow with: pl.read_parquet('data/processed/sample_500k.parquet')")