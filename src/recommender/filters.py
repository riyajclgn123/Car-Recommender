"""
src/recommender/filters.py

Hard filters — applied before scoring.
A car that fails any filter never appears in results.
"""

import polars as pl
import numpy as np
import logging

log = logging.getLogger(__name__)


def compute_budget(monthly_income: float) -> dict:
    """
    20/4/10 guideline:
    - 20% down payment
    - Loan of at most 4 years
    - Total car costs ≤ 10% of gross monthly income
    """
    max_monthly_cost = monthly_income * 0.10
    # Rough max price: monthly cost × 48 months + 20% down back
    # More precisely: solve for P given monthly payment = max_monthly_cost * 0.6
    # (leaving 40% for insurance + fuel + maintenance)
    loan_payment_budget = max_monthly_cost * 0.60
    # At 7.2% APR, 48 months: payment = P*0.8 * rate_factor
    rate_factor = 0.072 / 12 * (1 + 0.072/12)**48 / ((1 + 0.072/12)**48 - 1)
    max_loan    = loan_payment_budget / rate_factor
    max_price   = max_loan / 0.80   # loan is 80% of price (20% down)

    return {
        "max_price":        round(max_price),
        "max_monthly_cost": round(max_monthly_cost),
        "down_payment":     round(max_price * 0.20),
        "monthly_income":   monthly_income,
    }


def apply_hard_filters(
    df: pl.DataFrame,
    budget: dict,
    max_maintenance_monthly: float,
    min_seats: int = 1,
    prefer_new: bool | None = None,
) -> pl.DataFrame:
    """
    Apply hard filters to the car inventory.

    Args:
        df:                      Car listings with predicted_price column
        budget:                  Output of compute_budget()
        max_maintenance_monthly: User's stated maintenance budget per month
        min_seats:               Minimum seats needed (household_size)
        prefer_new:              True=new only, False=used only, None=either
    """
    before = len(df)

    # 1. Price filter — never exceed budget
    df = df.filter(pl.col("predicted_price") <= budget["max_price"])
    log.info(f"  After price filter (≤${budget['max_price']:,}): {len(df):,} cars")

    # 2. Maintenance filter
    max_maint_annual = max_maintenance_monthly * 12
    if "est_maintenance_annual" in df.columns:
        df = df.filter(
            pl.col("est_maintenance_annual") <= max_maint_annual)
        log.info(f"  After maintenance filter: {len(df):,} cars")

    # 3. Seats filter
    if "maximum_seating" in df.columns and min_seats > 1:
        df = df.filter(pl.col("maximum_seating") >= min_seats)
        log.info(f"  After seats filter (≥{min_seats}): {len(df):,} cars")

    # 4. New/used preference
    if prefer_new is True:
        df = df.filter(pl.col("is_new") == True)
    elif prefer_new is False:
        df = df.filter(pl.col("is_new") == False)

    log.info(f"  Total removed by filters: {before - len(df):,} "
             f"({(before - len(df))/before*100:.1f}%)")
    return df