"""
src/data/make_deploy_inventory.py
Builds the slim inventory the live API serves, with REAL fair-price
predictions computed once, offline (batch scoring).

- CarGurus rows: only cars from the val/test splits, which the model never
  trained on, so "deal" scores are honest out-of-sample estimates.
- Carvana rows: specs are placeholders, so predicted_price stays null
  (shown as "no estimate yet") rather than a made-up number.

Run:
    python -m src.data.make_deploy_inventory
"""
import joblib
import numpy as np
import polars as pl
from src.models.inference import load_meta, predict_prices

PROC = "data/processed"

KEEP = [
    "vin", "price", "predicted_price", "year", "make_name", "model_name",
    "trim_name", "body_type", "wheel_system", "maximum_seating", "is_new",
    "mileage", "city_fuel_economy", "highway_fuel_economy",
]

inv = pl.read_parquet(f"{PROC}/inventory.parquet")

# Out-of-sample CarGurus cars only
holdout_vins = pl.concat([
    pl.read_parquet(f"{PROC}/val.parquet").select("vin"),
    pl.read_parquet(f"{PROC}/test.parquet").select("vin"),
])
cargurus = inv.join(holdout_vins, on="vin", how="semi")
cargurus = cargurus.sample(n=min(80_000, len(cargurus)), seed=42)

# Carvana cars with real borrowed specs + drift-adjusted fair prices
# (unmatched Carvana names are left out: no specs, no honest estimate)
carvana = pl.read_parquet(f"{PROC}/carvana_enriched.parquet")

# Batch scoring with the shared, parity-tested inference code
model = joblib.load("models/lgbm_tuned.pkl")
meta = load_meta()
preds = predict_prices(cargurus, model, meta)
cargurus = cargurus.with_columns(pl.Series("predicted_price", preds))


deploy = pl.concat([cargurus.select(KEEP), carvana.select(KEEP)], how="vertical_relaxed")
deploy.write_parquet(f"{PROC}/inventory_deploy.parquet", compression="zstd")

# Sanity report
actual = cargurus["price"].to_numpy()
mae = np.mean(np.abs(preds - actual))
below = np.mean(actual < preds * 0.95) * 100
above = np.mean(actual > preds * 1.05) * 100
print(f"CarGurus (out-of-sample): {len(cargurus):,}  MAE ${mae:,.0f}")
print(f"Listed >5% below fair price: {below:.1f}%   >5% above: {above:.1f}%")
print(f"Carvana (enriched specs, drift-adjusted fair price): {len(carvana):,}")
print(f"Total: {len(deploy):,} → {PROC}/inventory_deploy.parquet")