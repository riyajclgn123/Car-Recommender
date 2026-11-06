import polars as pl
from fastapi import APIRouter, HTTPException
from api.schemas import RecommendRequest, RecommendResponse, CarRecommendation
from api.predictor import get_lifestage_probs
from src.recommender.filters import compute_budget, apply_hard_filters
from src.recommender.score   import compute_scores
from src.recommender.rank    import diversity_rank
from src.cost.tco import compute_tco, get_maintenance_5yr
import numpy as np

router = APIRouter()
PROC   = __import__("pathlib").Path("data/processed")

BODY_TO_CLASS = {
    "Sedan": "sedan", "Coupe": "sedan",
    "Hatchback": "sedan", "Wagon": "sedan",
    "SUV / Crossover": "suv_crossover",
    "Pickup Truck": "pickup",
    "Minivan": "minivan_van", "Van": "minivan_van",
    "Convertible": "other",
}


def build_why(row: dict, lifestage_probs: dict) -> str:
    """Generate a plain-language reason for the recommendation."""
    parts = []
    make  = row.get("make_name", "")
    model = row.get("model_name", "")
    year  = row.get("year", "")
    price = row.get("predicted_price", 0)
    tco   = row.get("tco_5yr", 0)
    body  = row.get("body_type", "")

    parts.append(f"The {year} {make} {model} fits your budget at "
                 f"${price:,.0f}.")

    if row.get("s_tco_fit", 0) > 0.80:
        parts.append(f"Its 5-year cost of ${tco:,.0f} is well "
                     f"within your financial plan.")

    if row.get("s_reliability", 0) >= 0.90:
        parts.append(f"{make} ranks among the most reliable brands.")

    cls = BODY_TO_CLASS.get(body, "other")
    if lifestage_probs.get(cls, 0) > 0.30:
        parts.append(f"A {body} matches what people in your "
                     f"life stage typically drive.")

    if row.get("s_deal", 0) > 0.60:
        diff = price - row.get("price", price)
        if diff > 0:
            parts.append(f"It's listed ${diff:,.0f} below fair market value.")

    return " ".join(parts)


@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    budget = compute_budget(req.monthly_income)

    lifestage_probs = get_lifestage_probs(
        req.age_group, req.income_band,
        req.household_size, req.area_type)

    # Load candidates
    df = pl.read_parquet(PROC / "inventory.parquet")
    if len(df) > req.n_candidates:
        df = df.sample(n=req.n_candidates, seed=42)

    # Add TCO estimates
    prices    = df["price"].to_numpy()
    make_col  = df["make_name"].to_list()
    year_col  = df["year"].to_list()
    city_mpg  = df["city_fuel_economy"].fill_null(25).to_numpy()
    hwy_mpg   = df["highway_fuel_economy"].fill_null(30).to_numpy()

    pred_prices, tco_vals, maint_vals = [], [], []
    rng = np.random.default_rng(42)

    for i in range(len(df)):
        p = float(prices[i]) * rng.uniform(0.92, 1.08)
        pred_prices.append(p)
        tco = compute_tco(p, make_col[i], int(year_col[i]),
                          float(city_mpg[i]), float(hwy_mpg[i]),
                          req.state)
        tco_vals.append(tco.total_tco)
        age = max(0, 2024 - int(year_col[i]))
        maint_vals.append(get_maintenance_5yr(make_col[i], age) / 5)

    df = df.with_columns([
        pl.Series("predicted_price", pred_prices),
        pl.Series("tco_5yr", tco_vals),
        pl.Series("est_maintenance_annual", maint_vals),
    ])

    # Filters
    df = apply_hard_filters(df, budget,
                            req.max_maintenance_monthly,
                            req.household_size, req.prefer_new)

    if len(df) == 0:
        raise HTTPException(
            status_code=404,
            detail="No cars found within your budget and constraints. "
                   "Try increasing your maintenance budget.")

    # Score + rank
    df   = compute_scores(df, budget, lifestage_probs, req.state)
    top5 = diversity_rank(df, top_n=5, max_per_model=2)

    # Build response
    recs = []
    for i, row in enumerate(top5.iter_rows(named=True), 1):
        monthly = row.get("tco_5yr", 0) / 60
        recs.append(CarRecommendation(
            rank            = i,
            make_name       = row.get("make_name", ""),
            model_name      = row.get("model_name", ""),
            year            = row.get("year", 0),
            listing_price   = round(row.get("price", 0), 2),
            predicted_price = round(row.get("predicted_price", 0), 2),
            tco_5yr         = round(row.get("tco_5yr", 0), 2),
            monthly_cost    = round(monthly, 2),
            total_score     = round(row.get("total_score", 0), 4),
            score_breakdown = {
                "tco_fit":    round(row.get("s_tco_fit", 0), 3),
                "deal":       round(row.get("s_deal", 0), 3),
                "reliability":round(row.get("s_reliability", 0), 3),
                "lifestage":  round(row.get("s_lifestage", 0), 3),
                "climate":    round(row.get("s_climate", 0), 3),
            },
            why = build_why(row, lifestage_probs),
        ))

    return RecommendResponse(
        budget_max_price  = budget["max_price"],
        budget_monthly    = budget["max_monthly_cost"],
        recommendations   = recs,
    )