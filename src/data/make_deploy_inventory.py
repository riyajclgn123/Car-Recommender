"""
src/data/make_deploy_inventory.py
Small inventory for the live server: only the columns /recommend uses,
all Carvana rows (newer cars) + a sample of CarGurus rows.

Run:
    python -m src.data.make_deploy_inventory
"""
import polars as pl

COLS = [
    "vin", "price", "year", "make_name", "model_name", "trim_name",
    "body_type", "wheel_system", "maximum_seating", "is_new", "mileage",
    "city_fuel_economy", "highway_fuel_economy",
]

df = pl.read_parquet("data/processed/inventory.parquet").select(COLS)

carvana  = df.filter(pl.col("vin").str.starts_with("CVN_"))
cargurus = df.filter(~pl.col("vin").str.starts_with("CVN_"))
cargurus = cargurus.sample(n=min(80_000, len(cargurus)), seed=42)

deploy = pl.concat([carvana, cargurus])
deploy.write_parquet("data/processed/inventory_deploy.parquet", compression="zstd")

print(f"Carvana: {len(carvana):,}  CarGurus sample: {len(cargurus):,}  "
      f"Total: {len(deploy):,}")