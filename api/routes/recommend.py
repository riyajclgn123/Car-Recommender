"""
api/routes/recommend.py
Top-5 recommendations from the pre-scored inventory.
No model inference per request: fair prices were computed offline
(src/data/make_deploy_inventory.py). No random numbers anywhere.
"""
import os
from functools import lru_cache
from pathlib import Path

import polars as pl
from fastapi import APIRouter, HTTPException

from api.schemas import RecommendRequest, RecommendResponse, CarRecommendation
from api.predictor import get_lifestage_probs
from src.recommender.filters import compute_budget, apply_hard_filters
from src.recommender.score import compute_scores
from src.recommender.rank import diversity_rank
from src.cost.tco import compute_tco, get_maintenance_5yr

router = APIRouter()

INVENTORY_PATH = Path(os.getenv("INVENTORY_PATH",
                                "data/processed/inventory_deploy.parquet"))

BODY_TO_CLASS = {
    "Sedan": "sedan", "Coupe": "sedan", "Hatchback": "sedan", "Wagon": "sedan",
    "SUV / Crossover": "suv_crossover", "Pickup Truck": "pickup",
    "Minivan": "minivan_van", "Van": "minivan_van", "Convertible": "other",
}


@lru_cache(maxsize=1)
def load_inventory() -> pl.DataFrame:
    """Read the pre-scored inventory once and keep it in memory."""
    df = pl.read_parquet(INVENTORY_PATH)
    if "predicted_price" not in df.columns:
        raise RuntimeError(f"{INVENTORY_PATH} has no predicted_price column. "
                           "Run: python -m src.data.make_deploy_inventory")
    return df


def build_why(row: dict, lifestage_probs: dict) -> str:
    """Plain-language reason, built only from real numbers."""
    parts = [f"The {row['year']} {row['make_name']} {row['model_name']} "
             f"is listed at ${row['price']:,.0f}, within your budget."]

    if row.get("s_tco_fit", 0) > 0.80:
        parts.append(f"Its 5-year cost of ${row['tco_5yr']:,.0f} fits your plan well.")

    if row.get("s_reliability", 0) >= 0.90:
        parts.append(f"{row['make_name']} ranks among the most reliable brands.")

    cls = BODY_TO_CLASS.get(row.get("body_type", ""), "other")
    if lifestage_probs.get(cls, 0) > 0.30:
        parts.append(f"A {row['body_type']} matches what people in your "
                     f"life stage typically drive.")

    if row.get("has_fair_price"):
        diff = row["fair_price"] - row["price"]
        if diff > 500:
            parts.append(f"It's listed ${diff:,.0f} below our fair-price estimate.")

    return " ".join(parts)


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    budget = compute_budget(req.monthly_income)
    lifestage_probs = get_lifestage_probs(
        req.age_group, req.income_band, req.household_size, req.area_type)

    # Budget is about what the user PAYS, so filter on listing price first
    df = load_inventory().filter(pl.col("price") <= budget["max_price"])
    if len(df) == 0:
        raise HTTPException(404, "No cars found within your budget. "
                                 "Try a higher income or maintenance budget.")
    if len(df) > req.n_candidates:
        df = df.sample(n=req.n_candidates, seed=42)

    # fair_price = model estimate (may be null); predicted_price is filled
    # with the listing price for cars with no estimate, so the deal score is neutral
    df = df.with_columns([
        pl.col("predicted_price").is_not_null().alias("has_fair_price"),
        pl.col("predicted_price").alias("fair_price"),
        pl.col("predicted_price").fill_null(pl.col("price")).alias("predicted_price"),
    ])

    # 5-year cost and maintenance, based on the price the user actually pays
    tco_vals, maint_vals = [], []
    for price, make, year, city, hwy in df.select(
            ["price", "make_name", "year",
             "city_fuel_economy", "highway_fuel_economy"]).iter_rows():
        tco = compute_tco(float(price), make, int(year),
                          float(city or 25), float(hwy or 30), req.state)
        tco_vals.append(tco.total_tco)
        maint_vals.append(get_maintenance_5yr(make, max(0, 2024 - int(year))) / 5)

    df = df.with_columns([
        pl.Series("tco_5yr", tco_vals),
        pl.Series("est_maintenance_annual", maint_vals),
    ])

    df = apply_hard_filters(df, budget, req.max_maintenance_monthly,
                            req.household_size, req.prefer_new)
    if len(df) == 0:
        raise HTTPException(404, "No cars matched all your constraints. "
                                 "Try increasing your maintenance budget.")

    df = compute_scores(df, budget, lifestage_probs, req.state)
    top5 = diversity_rank(df, top_n=5, max_per_model=2)

    recs = []
    for i, row in enumerate(top5.iter_rows(named=True), 1):
        recs.append(CarRecommendation(
            rank=i,
            make_name=row["make_name"],
            model_name=row["model_name"],
            year=row["year"],
            listing_price=round(row["price"], 2),
            predicted_price=round(row["predicted_price"], 2),
            has_fair_price=bool(row["has_fair_price"]),
            tco_5yr=round(row["tco_5yr"], 2),
            monthly_cost=round(row["tco_5yr"] / 60, 2),
            total_score=round(row["total_score"], 4),
            score_breakdown={
                "tco_fit": round(row["s_tco_fit"], 3),
                "deal": round(row["s_deal"], 3),
                "reliability": round(row["s_reliability"], 3),
                "lifestage": round(row["s_lifestage"], 3),
                "climate": round(row["s_climate"], 3),
            },
            why=build_why(row, lifestage_probs),
        ))

    return RecommendResponse(
        budget_max_price=budget["max_price"],
        budget_monthly=budget["max_monthly_cost"],
        recommendations=recs,
    )