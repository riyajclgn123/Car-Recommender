"""
src/recommender/engine.py

Main recommender engine — wires filters, scoring, and ranking.

Run:
    python -m src.recommender.engine
"""

import polars as pl
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from dataclasses import dataclass

from src.recommender.filters import compute_budget, apply_hard_filters
from src.recommender.score   import compute_scores
from src.recommender.rank    import diversity_rank, format_results
from src.cost.tco            import compute_tco, get_maintenance_5yr

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

MODELS = Path("models")
PROC   = Path("data/processed")


@dataclass
class UserProfile:
    monthly_income:          float
    max_maintenance_monthly: float
    household_size:          int
    age_group:               str    # "18-34", "35-54", "55-64", "65+"
    income_band:             str    # "low", "middle", "high"
    area_type:               str    # "urban", "suburban", "rural"
    state:                   str    # 2-letter state code
    prefer_new:              bool | None = None  # True/False/None


def load_models():
    """Load all three models."""
    log.info("Loading models …")
    price_model     = joblib.load(MODELS / "lgbm_tuned.pkl")
    lifestage_model = joblib.load(MODELS / "lifestage_classifier.pkl")
    lifestage_le    = joblib.load(MODELS / "lifestage_label_encoder.pkl")

    with open(MODELS / "lifestage_meta.json") as f:
        lifestage_meta = json.load(f)

    with open(PROC / "feature_meta.json") as f:
        feature_meta = json.load(f)

    return price_model, lifestage_model, lifestage_le, \
           lifestage_meta, feature_meta


def get_lifestage_probs(profile: UserProfile,
                        lifestage_model,
                        lifestage_le,
                        lifestage_meta: dict) -> dict:
    """Get vehicle class probabilities for the user profile."""
    age_map    = lifestage_meta["age_map"]
    income_map = lifestage_meta["income_map"]
    area_map   = lifestage_meta["area_map"]

    age    = age_map.get(profile.age_group, 1)
    income = income_map.get(profile.income_band, 1)
    hh     = min(profile.household_size, 4)
    area   = area_map.get(profile.area_type, 1)

    features = np.array([[
        age, income, hh, area,
        age * income,   # interaction
        hh  * area,     # interaction
    ]])

    probs   = lifestage_model.predict_proba(features)[0]
    classes = lifestage_meta["classes"]

    return dict(zip(classes, probs))


def add_predictions_and_tco(df: pl.DataFrame,
                             price_model,
                             feature_meta: dict,
                             profile: UserProfile) -> pl.DataFrame:
    """
    Add predicted_price, tco_5yr, and est_maintenance_annual
    to the dataframe using the price model and TCO formula.
    """
    # Use price column as predicted_price proxy
    # (in production, run actual model inference per row)
    # For now: use listing price ± small noise as stand-in
    # Real implementation wires through the full feature pipeline

    prices    = df["price"].to_numpy()
    make_col  = df["make_name"].to_list()
    year_col  = df["year"].to_list()
    city_mpg  = df["city_fuel_economy"].fill_null(25).to_numpy()
    hwy_mpg   = df["highway_fuel_economy"].fill_null(30).to_numpy()

    predicted_prices   = []
    tco_5yr_vals       = []
    maint_annual_vals  = []

    for i in range(len(df)):
        # Simulate price model predictions with small variation
        # Real implementation: run lgbm model on feature row
        import random
        variation = random.uniform(0.90, 1.10)
        pred_price = float(prices[i]) * variation
        predicted_prices.append(pred_price)

        tco = compute_tco(
            predicted_price = pred_price,
            make_name       = make_col[i],
            year            = int(year_col[i]),
            city_mpg        = float(city_mpg[i]),
            highway_mpg     = float(hwy_mpg[i]),
            state           = profile.state,
            annual_miles    = 12_000,
        )
        tco_5yr_vals.append(tco.total_tco)

        age = max(0, 2024 - int(year_col[i]))
        maint = get_maintenance_5yr(make_col[i], age) / 5
        maint_annual_vals.append(maint)

    df = df.with_columns([
        pl.Series("predicted_price",      predicted_prices),
        pl.Series("tco_5yr",              tco_5yr_vals),
        pl.Series("est_maintenance_annual", maint_annual_vals),
    ])

    return df


def recommend(profile: UserProfile,
              n_candidates: int = 2000) -> pl.DataFrame:
    """
    Main recommender function.
    Returns top 5 cars for the user profile.
    """
    # Load models
    (price_model, lifestage_model, lifestage_le,
     lifestage_meta, feature_meta) = load_models()

    # Compute budget
    budget = compute_budget(profile.monthly_income)
    log.info(f"\nBudget: max price=${budget['max_price']:,}  "
             f"max monthly=${budget['max_monthly_cost']:,}")

    # Get life-stage probabilities
    lifestage_probs = get_lifestage_probs(
        profile, lifestage_model, lifestage_le, lifestage_meta)
    log.info(f"Life-stage probs: "
             + " ".join(f"{k}={v:.2f}" for k,v in lifestage_probs.items()))

    # Load candidate cars from cleaned data
    log.info(f"\nLoading {n_candidates:,} candidate cars …")
    df = pl.read_parquet(PROC / "cleaned.parquet")

    # Sample candidates (in production: query a database)
    if len(df) > n_candidates:
        df = df.sample(n=n_candidates, seed=42)

    # Add predictions and TCO
    log.info("Computing predictions and TCO …")
    df = add_predictions_and_tco(df, price_model, feature_meta, profile)

    # Hard filters
    log.info("\nApplying hard filters …")
    df = apply_hard_filters(
        df,
        budget                  = budget,
        max_maintenance_monthly = profile.max_maintenance_monthly,
        min_seats               = profile.household_size,
        prefer_new              = profile.prefer_new,
    )

    if len(df) == 0:
        log.warning("No cars passed filters! "
                    "Try relaxing budget or maintenance constraints.")
        return pl.DataFrame()

    # Score
    log.info("\nScoring candidates …")
    df = compute_scores(df, budget, lifestage_probs, profile.state)

    # Rank with diversity
    log.info("\nRanking with diversity pass …")
    top5 = diversity_rank(df, top_n=5, max_per_model=2)

    return top5


def run_persona_tests():
    """
    Test the recommender with the 3 project personas.
    All results must be under budget.
    """
    personas = [
        UserProfile(
            monthly_income          = 25_000 / 12,
            max_maintenance_monthly = 150,
            household_size          = 1,
            age_group               = "18-34",
            income_band             = "low",
            area_type               = "urban",
            state                   = "LA",
            prefer_new              = False,
        ),
        UserProfile(
            monthly_income          = 95_000 / 12,
            max_maintenance_monthly = 300,
            household_size          = 4,
            age_group               = "35-54",
            income_band             = "high",
            area_type               = "suburban",
            state                   = "TX",
            prefer_new              = None,
        ),
        UserProfile(
            monthly_income          = 60_000 / 12,
            max_maintenance_monthly = 200,
            household_size          = 2,
            age_group               = "65+",
            income_band             = "middle",
            area_type               = "suburban",
            state                   = "FL",
            prefer_new              = False,
        ),
    ]

    names = [
        "21-yr student ($25k/yr, LA)",
        "Family of 4 ($95k/yr, TX)",
        "68-yr retiree ($60k/yr, FL)",
    ]

    all_pass = True

    for name, profile in zip(names, personas):
        log.info("\n" + "="*65)
        log.info(f"PERSONA: {name}")
        log.info("="*65)

        budget = compute_budget(profile.monthly_income)
        top5   = recommend(profile)

        if len(top5) == 0:
            log.warning(f"  ❌ No results for {name}")
            all_pass = False
            continue

        format_results(top5)

        # Hard check: no result over budget
        over_budget = top5.filter(
            pl.col("predicted_price") > budget["max_price"])
        if len(over_budget) > 0:
            log.error(f"  🚨 BUDGET VIOLATION: "
                      f"{len(over_budget)} results over "
                      f"${budget['max_price']:,}")
            all_pass = False
        else:
            log.info(f"\n  ✅ All results within budget "
                     f"(${budget['max_price']:,})")

    log.info("\n" + "="*65)
    if all_pass:
        log.info("✅ ALL PERSONA TESTS PASSED")
    else:
        log.info("❌ SOME PERSONA TESTS FAILED — review above")
    log.info("="*65)


if __name__ == "__main__":
    run_persona_tests()