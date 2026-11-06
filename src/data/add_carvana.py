"""
src/data/add_carvana.py
Builds the recommender inventory: cleaned CarGurus + Carvana listings.
Writes a SEPARATE file so training data (cleaned.parquet) stays untouched.

Run:
    python -m src.data.add_carvana
"""
import math
import pandas as pd
import polars as pl

cv = pd.read_csv("data/raw/carvana.csv")
print(f"Carvana raw: {len(cv):,}")

for c in ["Price", "Miles", "Year"]:
    cv[c] = pd.to_numeric(cv[c], errors="coerce")
cv = cv.dropna(subset=["Price", "Miles", "Year"])
cv = cv[(cv["Price"] > 500) & (cv["Year"].between(2010, 2024))
        & (cv["Miles"] <= 400_000)]

split = cv["Name"].str.strip().str.split(" ", n=1, expand=True)
cv["make_name"]  = split[0]
cv["model_name"] = split[1].fillna("Unknown")
print(f"Carvana clean: {len(cv):,}  years {cv['Year'].min():.0f}-{cv['Year'].max():.0f}")

existing = pl.read_parquet("data/processed/cleaned.parquet")
n = len(cv)
rows = {
    "vin": [f"CVN_{i}" for i in range(n)],
    "price": cv["Price"].astype(float).tolist(),
    "year": cv["Year"].astype(int).tolist(),
    "make_name": cv["make_name"].tolist(),
    "model_name": cv["model_name"].tolist(),
    "mileage": cv["Miles"].astype(float).tolist(),
    "log_price": [math.log(p) for p in cv["Price"]],
}
defaults = {
    "trim_name": "Unknown", "body_type": "Unknown", "fuel_type": "Gasoline",
    "engine_type": "Unknown", "engine_cylinders": "Unknown",
    "transmission": "Automatic", "wheel_system": "FWD",
    "engine_displacement": 2.5, "horsepower": 180.0, "maximum_seating": 5.0,
    "city_fuel_economy": 25.0, "highway_fuel_economy": 30.0,
    "owner_count": 1.0, "dealer_zip": 0, "latitude": 35.0, "longitude": -90.0,
    "has_accidents": False, "frame_damaged": False, "fleet": False,
    "salvage": False, "theft_title": False, "is_new": False,
}
cv_df = pl.DataFrame(rows).with_columns(
    [pl.lit(v).alias(k) for k, v in defaults.items()]
)
cv_df = cv_df.with_columns(
    [pl.lit(0).cast(pl.Int8).alias(c) for c in existing.columns
     if c.startswith("was_missing_")]
)

# Match schema exactly, then stack
cv_df = cv_df.select(existing.columns).cast(existing.schema)
inventory = pl.concat([existing, cv_df])
inventory.write_parquet("data/processed/inventory.parquet", compression="zstd")
print(f"Inventory: {len(inventory):,} rows → data/processed/inventory.parquet")