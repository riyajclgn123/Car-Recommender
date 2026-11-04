"""
src/models/baseline.py

Baseline model: median log_price by make-model-year group.
This is the floor every real model must beat.

Run:
    python -m src.models.baseline
"""

import polars as pl
import numpy as np
import mlflow
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

PROC = Path("data/processed")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    label: str = "") -> dict:
    """
    All metrics in DOLLARS (convert from log space).
    y_true and y_pred are in log(price) space.
    """
    # Convert back to dollars
    true_dollars = np.exp(y_true)
    pred_dollars = np.exp(y_pred)

    mae  = np.mean(np.abs(true_dollars - pred_dollars))
    mape = np.mean(np.abs((true_dollars - pred_dollars) / true_dollars)) * 100
    rmse = np.sqrt(np.mean((true_dollars - pred_dollars) ** 2))

    # R² on log scale (standard practice)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    if label:
        log.info(f"\n  {label}:")
    log.info(f"    MAE:  ${mae:>10,.0f}")
    log.info(f"    MAPE: {mape:>9.2f}%")
    log.info(f"    RMSE: ${rmse:>10,.0f}")
    log.info(f"    R²:   {r2:>10.4f}")

    return {"mae": mae, "mape": mape, "rmse": rmse, "r2": r2}


def run():
    log.info("=" * 55)
    log.info("BASELINE MODEL")
    log.info("=" * 55)

    # Load train (to build lookup table) and val (to evaluate)
    train = pl.read_parquet(PROC / "train.parquet")
    val   = pl.read_parquet(PROC / "val.parquet")
    test  = pl.read_parquet(PROC / "test.parquet")

    log.info(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")

    # ── Build lookup table: median log_price by make-model-year ───────────────
    lookup = (
        train.group_by(["make_name", "model_name", "year"])
        .agg(pl.col("log_price").median().alias("pred_log_price"))
    )
    log.info(f"Lookup table: {len(lookup):,} make-model-year groups")

    # Global fallback: overall median (for unseen groups in val/test)
    global_median = float(train["log_price"].median())
    log.info(f"Global fallback median log_price: {global_median:.4f} "
             f"(= ${np.exp(global_median):,.0f})")

    def predict(df: pl.DataFrame) -> np.ndarray:
        """Join lookup table; fall back to global median for unseen groups."""
        df = df.join(lookup, on=["make_name", "model_name", "year"], how="left")
        df = df.with_columns(
            pl.col("pred_log_price")
              .fill_null(global_median)
              .alias("pred_log_price")
        )
        return df["pred_log_price"].to_numpy()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="baseline_median_make_model_year"):
        mlflow.set_tag("model_type", "baseline")
        mlflow.set_tag("description", "Median log_price by make-model-year")

        for split_name, df in [("val", val), ("test", test)]:
            y_true = df["log_price"].to_numpy()
            y_pred = predict(df)

            metrics = compute_metrics(y_true, y_pred, label=split_name.upper())

            # Log to MLflow
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"{split_name}_{metric_name}", value)

        # How many val rows needed the global fallback?
        val_joined = val.join(lookup, on=["make_name","model_name","year"],
                              how="left")
        fallback_count = val_joined["pred_log_price"].is_null().sum()
        log.info(f"\n  Val rows using global fallback: {fallback_count:,} "
                 f"({fallback_count/len(val)*100:.1f}%)")
        mlflow.log_metric("val_fallback_rows", fallback_count)

    log.info("\nRun logged to MLflow — check http://localhost:5000")
    log.info("BASELINE COMPLETE")


if __name__ == "__main__":
    run()