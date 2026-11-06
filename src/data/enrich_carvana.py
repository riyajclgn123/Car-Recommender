"""
src/data/enrich_carvana.py
Give Carvana listings real specs, score them with the price model, and
measure market drift.

Carvana only gives a name ("Kia Sorento Hybrid"), year, miles and price.
CarGurus (cleaned.parquet) has full specs. For each Carvana car we find the
CarGurus group with the same make + model + (nearest) year and borrow its
TYPICAL specs: median numbers, most common categories. Names we can't match
safely stay unmatched and get no price estimate (no guessing).

Drift check: Carvana prices come from a later market than the 2019-2020
training data, and the market moved differently by model year (older cars
rose most). Per-model-year factors (median of price / prediction) are
learned on half the cars and validated on the other half, then re-learned
on all cars. Deal scores therefore mean "vs. similar Carvana cars of the
same model year", centred by construction: be upfront about it.

Run:
    python -m src.data.enrich_carvana
Output:
    data/processed/carvana_enriched.parquet
"""
import joblib
import numpy as np
import pandas as pd
import polars as pl
from src.models.inference import load_meta, predict_prices

PROC = "data/processed"
RAW = "data/raw"

NUMERIC_SPECS = ["horsepower", "engine_displacement", "city_fuel_economy",
                 "highway_fuel_economy", "maximum_seating"]
CATEGORICAL_SPECS = ["trim_name", "body_type", "fuel_type", "engine_type",
                     "engine_cylinders", "transmission", "wheel_system"]
HISTORY_FLAGS = ["has_accidents", "frame_damaged", "fleet", "salvage", "theft_title"]
MIN_REVERSE_PREFIX = 2   # "EX" may match EX35; "X" is too ambiguous
MAX_YEAR_GAP = 2         # bigger gaps risk borrowing a different generation's specs
MIN_CARS_PER_YEAR = 30   # model years with fewer cars use the global drift factor
MIN_DONORS = 5           # a donor group needs at least this many CarGurus cars


def load_carvana() -> pd.DataFrame:
    cv = pd.read_csv(f"{RAW}/carvana.csv")
    for c in ["Price", "Miles", "Year"]:
        cv[c] = pd.to_numeric(cv[c], errors="coerce")
    cv = cv.dropna(subset=["Price", "Miles", "Year"])
    cv = cv[(cv["Price"] > 500) & cv["Year"].between(2010, 2024)
            & (cv["Miles"] <= 400_000)].copy()
    cv["Name"] = cv["Name"].str.strip()
    cv["vin"] = [f"CVN_{i}" for i in range(len(cv))]
    return cv.reset_index(drop=True)


def build_donors(cg: pl.DataFrame) -> pl.DataFrame:
    """Typical specs per CarGurus make + model + year."""
    return cg.group_by(["make_name", "model_name", "year"]).agg(
        [pl.col(c).median() for c in NUMERIC_SPECS]
        + [pl.col(c).drop_nulls().mode().first() for c in CATEGORICAL_SPECS]
        + [pl.len().alias("donor_n")]
    )


def match_candidates(name, makes_by_lower, models_by_make):
    """Return (make, [(model, method), ...]) in order of preference.

    All plausible models are returned, so if the best name match has no
    donor near the right year (e.g. a rare old listing called just "RX"),
    the next candidate ("RX 350") still gets a chance.
    """
    words = name.split()
    make = None
    for k in range(min(3, len(words) - 1), 0, -1):       # longest make first
        cand = " ".join(words[:k]).lower()
        if cand in makes_by_lower:
            make, rest = makes_by_lower[cand], " ".join(words[k:])
            break
    if make is None:
        return None, []

    models = models_by_make[make]                          # {model: count}
    rest_l = rest.lower()
    out = [(m, "exact") for m in models if m.lower() == rest_l]
    fwd = sorted((m for m in models if rest_l.startswith(m.lower() + " ")),
                 key=len, reverse=True)                    # "Titan Crew Cab" -> "Titan"
    out += [(m, "prefix") for m in fwd]
    if len(rest) >= MIN_REVERSE_PREFIX:                    # "RX" -> "RX 350", "EX" -> "EX35"
        rev = sorted((m for m in models
                      if m.lower().startswith(rest_l) and m.lower() != rest_l),
                     key=lambda m: -models[m])             # most common first
        out += [(m, "truncated") for m in rev]
    return make, out


def drift_factors(years, ratio):
    """Median real/predicted per model year; small years fall back to global."""
    glob = float(np.median(ratio))
    stats = (pd.DataFrame({"year": years, "r": ratio})
             .groupby("year")["r"].agg(["median", "size"]))
    per_year = {int(y): (float(st["median"]) if st["size"] >= MIN_CARS_PER_YEAR else glob)
                for y, st in stats.iterrows()}
    return glob, per_year


def run():
    cv = load_carvana()
    cg = pl.read_parquet(f"{PROC}/cleaned.parquet")
    donors = build_donors(cg)

    counts = cg.group_by(["make_name", "model_name"]).len()
    models_by_make = {}
    for mk, md, n in counts.iter_rows():
        models_by_make.setdefault(mk, {})[md] = n
    makes_by_lower = {mk.lower(): mk for mk in models_by_make}

    # Donor years available for each make/model (only groups with enough cars,
    # so one odd listing can't set the specs for hundreds of Carvana cars)
    years = {}
    for mk, md, yr, n in donors.select(
            ["make_name", "model_name", "year", "donor_n"]).iter_rows():
        if n >= MIN_DONORS:
            years.setdefault((mk, md), []).append(yr)

    # Candidate models for each distinct name (342 names, not 19k rows)
    cands = {n: match_candidates(n, makes_by_lower, models_by_make)
             for n in cv["Name"].unique()}

    def choose(row):
        """First candidate with a donor within MAX_YEAR_GAP years."""
        make, options = cands[row["Name"]]
        if make is None:
            return pd.Series([None, None, "unknown_make", np.nan])
        if not options:
            return pd.Series([make, None, "unknown_model", np.nan])
        for model, method in options:
            ys = years.get((make, model))
            if ys:
                best = min(ys, key=lambda y: (abs(y - row["Year"]), -y))
                if abs(best - row["Year"]) <= MAX_YEAR_GAP:
                    return pd.Series([make, model, method, float(best)])
        return pd.Series([make, None, "year_gap_too_big", np.nan])

    cv[["make_name", "model_name", "match_method", "donor_year"]] = cv.apply(choose, axis=1)
    cv["donor_year"] = pd.to_numeric(cv["donor_year"])
    cv["year_gap"] = (cv["Year"] - cv["donor_year"]).abs()

    # Report the match quality
    print("Match methods (rows):", cv["match_method"].value_counts().to_dict())
    ok = cv["donor_year"].notna()
    print(f"Matched: {ok.sum():,} / {len(cv):,} rows ({ok.mean():.1%})")
    print("Year gap for matched rows:",
          cv.loc[ok, "year_gap"].value_counts().sort_index().to_dict())
    bad = cv.loc[~ok, "Name"].value_counts().head(15)
    print("Top unmatched names:", bad.to_dict())

    # Join borrowed specs
    m = pl.from_pandas(cv[ok].rename(columns={"donor_year": "donor_year_f"}))
    m = m.with_columns(pl.col("donor_year_f").cast(pl.Int32).alias("donor_year"))
    m = m.join(donors.rename({"year": "donor_year"}),
               on=["make_name", "model_name", "donor_year"], how="left")

    # Fields Carvana doesn't report -> same treatment as missing in training
    medians = cg.select(["owner_count", "latitude", "longitude"]).median().row(0)
    m = m.with_columns(
        [pl.col("Year").cast(pl.Int32).alias("year"),
         pl.col("Miles").cast(pl.Float64).alias("mileage"),
         pl.col("Price").cast(pl.Float64).alias("price"),
         pl.lit(False).alias("is_new"),
         pl.lit(medians[0]).alias("owner_count"),
         pl.lit(medians[1]).alias("latitude"),
         pl.lit(medians[2]).alias("longitude"),
         pl.lit(1, dtype=pl.Int8).alias("was_missing_owner_count"),
         pl.lit(0, dtype=pl.Int8).alias("was_missing_mileage"),
         pl.lit(0, dtype=pl.Int8).alias("was_missing_horsepower"),
         pl.lit(0, dtype=pl.Int8).alias("was_missing_engine_displacement")]
        + [pl.lit(False).alias(f) for f in HISTORY_FLAGS]
        + [pl.lit(1, dtype=pl.Int8).alias(f"was_missing_{f}") for f in HISTORY_FLAGS]
    )

    # Score with the shared, parity-tested inference code
    model = joblib.load("models/lgbm_tuned.pkl")
    meta = load_meta()
    pred = predict_prices(m, model, meta)
    actual = m["price"].to_numpy()

    # Drift check, done honestly: learn factors on half the cars,
    # measure error on the other half.
    years_arr = m["year"].to_numpy()
    ratio = actual / pred
    fit = np.random.default_rng(42).random(len(m)) < 0.5
    ev = ~fit

    g_fit, py_fit = drift_factors(years_arr[fit], ratio[fit])
    per_year_ev = np.array([py_fit.get(int(y), g_fit) for y in years_arr[ev]])

    def mae(f):
        return np.mean(np.abs(actual[ev] - pred[ev] * f))

    mape_py = np.mean(np.abs(actual[ev] - pred[ev] * per_year_ev) / actual[ev]) * 100
    print(f"\nDrift check (factors learned on {fit.sum():,} cars, "
          f"evaluated on {ev.sum():,} held-out cars):")
    print(f"  No adjustment:        MAE ${mae(1.0):,.0f}")
    print(f"  One global factor:    MAE ${mae(g_fit):,.0f}   (factor {g_fit:.3f})")
    print(f"  Per-model-year:       MAE ${mae(per_year_ev):,.0f}   (MAPE {mape_py:.1f}%)")

    # Final factors from ALL matched cars
    g_all, py_all = drift_factors(years_arr, ratio)
    factors = np.array([py_all.get(int(y), g_all) for y in years_arr])
    print("  Final per-year factors:",
          {y: round(f, 2) for y, f in sorted(py_all.items())})

    out = m.with_columns(
        pl.Series("predicted_price_2020", pred),
        pl.Series("predicted_price", pred * factors),
        pl.Series("drift_factor", factors),
    ).select(["vin", "price", "predicted_price", "predicted_price_2020",
              "drift_factor", "year", "make_name", "model_name", "trim_name",
              "body_type", "wheel_system", "maximum_seating", "is_new", "mileage",
              "city_fuel_economy", "highway_fuel_economy",
              "match_method", "year_gap"])
    out.write_parquet(f"{PROC}/carvana_enriched.parquet", compression="zstd")
    print(f"\nSaved {len(out):,} enriched Carvana cars → "
          f"{PROC}/carvana_enriched.parquet")


if __name__ == "__main__":
    run()
    