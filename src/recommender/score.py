"""
src/recommender/score.py

Weighted scoring for each car after hard filters.

Weights (starting values, tune against persona tests):
  TCO fit        0.30  — 5-year cost vs user's budget
  Deal score     0.25  — predicted fair price vs listing price
  Reliability    0.20  — NHTSA complaints (proxied by brand tier)
  Life-stage fit 0.15  — probability of this vehicle class for user
  Climate fit    0.10  — AWD in cold states, etc.
"""

import polars as pl
import numpy as np
import logging

log = logging.getLogger(__name__)

# Scoring weights — must sum to 1.0
WEIGHTS = {
    "tco_fit":      0.30,
    "deal_score":   0.25,
    "reliability":  0.20,
    "lifestage_fit": 0.15,
    "climate_fit":  0.10,
}

# Brand reliability tiers (proxy for NHTSA complaints)
# 1.0 = most reliable, 0.0 = least reliable
RELIABILITY = {
    "Toyota": 0.95, "Honda": 0.93, "Mazda": 0.90, "Lexus": 0.95,
    "Acura": 0.88,  "Subaru": 0.85, "Hyundai": 0.80, "Kia": 0.78,
    "Buick": 0.75,  "Nissan": 0.72, "Ford": 0.68, "Chevrolet": 0.65,
    "GMC": 0.65,    "Jeep": 0.55,  "Dodge": 0.52, "Chrysler": 0.50,
    "BMW": 0.60,    "Mercedes-Benz": 0.62, "Audi": 0.58,
    "Volkswagen": 0.65, "Volvo": 0.70, "Cadillac": 0.60,
    "Land Rover": 0.35, "Jaguar": 0.40, "Maserati": 0.30,
}

# States that benefit from AWD/4WD
AWD_STATES = {
    "AK", "CO", "ID", "ME", "MI", "MN", "MT", "NH",
    "ND", "OR", "SD", "UT", "VT", "WA", "WI", "WY"
}


def score_tco_fit(tco_5yr: float, budget: dict) -> float:
    """
    How well does the 5-year TCO fit the user's budget?
    Score = 1 - (TCO / (max_monthly_cost * 60))
    Capped at [0, 1].
    """
    max_tco = budget["max_monthly_cost"] * 60
    ratio   = tco_5yr / max_tco if max_tco > 0 else 1.0
    return float(np.clip(1 - ratio + 0.5, 0, 1))


def score_deal(predicted_price: float, listing_price: float) -> float:
    """
    Deal score: how much below fair value is this listing?
    Positive = good deal, negative = overpriced.
    Score normalized to [0, 1].
    """
    if listing_price <= 0:
        return 0.5
    pct_diff = (predicted_price - listing_price) / predicted_price
    # +20% below fair = score 1.0; +20% above fair = score 0.0
    return float(np.clip(pct_diff / 0.20 * 0.5 + 0.5, 0, 1))


def score_reliability(make_name: str) -> float:
    """Brand reliability from lookup table."""
    return RELIABILITY.get(make_name, 0.60)


def score_lifestage(body_type: str,
                    lifestage_probs: dict) -> float:
    """
    Probability that this vehicle class fits the user's life stage.
    Maps body_type to vehicle class and returns the probability.
    """
    body_to_class = {
        "Sedan":          "sedan",
        "Coupe":          "sedan",
        "Hatchback":      "sedan",
        "Wagon":          "sedan",
        "SUV / Crossover": "suv_crossover",
        "Pickup Truck":   "pickup",
        "Minivan":        "minivan_van",
        "Van":            "minivan_van",
        "Convertible":    "other",
    }
    vehicle_class = body_to_class.get(body_type, "other")
    return lifestage_probs.get(vehicle_class, 0.10)


def score_climate(wheel_system: str, state: str) -> float:
    """
    AWD/4WD bonus in cold/snowy states.
    FWD penalty in those states.
    """
    needs_awd = state.upper() in AWD_STATES
    if not needs_awd:
        return 0.70   # neutral score for non-snowy states

    if wheel_system in ("AWD", "4WD", "4X4"):
        return 1.0
    elif wheel_system == "RWD":
        return 0.30
    else:
        return 0.55   # FWD — acceptable but not ideal


def compute_scores(
    df: pl.DataFrame,
    budget: dict,
    lifestage_probs: dict,
    state: str,
) -> pl.DataFrame:
    """
    Compute weighted score for each car.
    Returns df with score columns added.
    """
    scores = {
        "s_tco_fit":    [],
        "s_deal":       [],
        "s_reliability":[],
        "s_lifestage":  [],
        "s_climate":    [],
        "total_score":  [],
    }

    for row in df.iter_rows(named=True):
        tco       = row.get("tco_5yr", budget["max_monthly_cost"] * 60)
        pred_p    = row.get("predicted_price", row.get("price", 0))
        list_p    = row.get("price", pred_p)
        make      = row.get("make_name", "")
        body      = row.get("body_type", "")
        wheel     = row.get("wheel_system", "FWD")

        s_tco  = score_tco_fit(tco, budget)
        s_deal = score_deal(pred_p, list_p)
        s_rel  = score_reliability(make)
        s_life = score_lifestage(body, lifestage_probs)
        s_clim = score_climate(wheel, state)

        total = (
            WEIGHTS["tco_fit"]       * s_tco  +
            WEIGHTS["deal_score"]    * s_deal +
            WEIGHTS["reliability"]   * s_rel  +
            WEIGHTS["lifestage_fit"] * s_life +
            WEIGHTS["climate_fit"]   * s_clim
        )

        scores["s_tco_fit"].append(s_tco)
        scores["s_deal"].append(s_deal)
        scores["s_reliability"].append(s_rel)
        scores["s_lifestage"].append(s_life)
        scores["s_climate"].append(s_clim)
        scores["total_score"].append(total)

    for col, vals in scores.items():
        df = df.with_columns(pl.Series(col, vals))

    return df