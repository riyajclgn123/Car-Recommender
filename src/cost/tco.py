"""
src/cost/tco.py

5-year Total Cost of Ownership formula over lookup tables.
Not a trained model — deterministic math.

Usage:
    from src.cost.tco import compute_tco

    result = compute_tco(
        predicted_price   = 24000,
        make_name         = "Toyota",
        model_name        = "Camry",
        year              = 2020,
        city_mpg          = 29,
        highway_mpg       = 38,
        state             = "LA",
        annual_miles      = 12000,
        is_new            = False,
        price_model       = lgbm_model,   # for depreciation
        feature_row       = X_row,        # feature vector for this car
    )

Run standalone test:
    python -m src.cost.tco
"""

import json
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

LOOKUP = Path("src/cost/lookup_tables")

# ── Load lookup tables once at import time ────────────────────────────────────
with open(LOOKUP / "gas_prices.json")  as f: GAS    = json.load(f)
with open(LOOKUP / "insurance.json")   as f: INS    = json.load(f)
with open(LOOKUP / "maintenance.json") as f: MAINT  = json.load(f)

# National average auto loan rate (source: Federal Reserve, recent)
LOAN_RATE = 0.072   # 7.2% APR
DOWN_PAYMENT_PCT = 0.20
LOAN_YEARS = 4


@dataclass
class TCOResult:
    """Full 5-year cost breakdown."""
    predicted_price:  float
    depreciation:     float
    fuel_5yr:         float
    insurance_5yr:    float
    maintenance_5yr:  float
    loan_interest:    float
    total_tco:        float
    monthly_cost:     float

    def display(self):
        log.info("\n  5-Year Cost of Ownership:")
        log.info(f"    Predicted price:   ${self.predicted_price:>10,.0f}")
        log.info(f"    Depreciation:      ${self.depreciation:>10,.0f}")
        log.info(f"    Fuel (5yr):        ${self.fuel_5yr:>10,.0f}")
        log.info(f"    Insurance (5yr):   ${self.insurance_5yr:>10,.0f}")
        log.info(f"    Maintenance (5yr): ${self.maintenance_5yr:>10,.0f}")
        log.info(f"    Loan interest:     ${self.loan_interest:>10,.0f}")
        log.info(f"    ─────────────────────────────────")
        log.info(f"    TOTAL TCO (5yr):   ${self.total_tco:>10,.0f}")
        log.info(f"    Monthly cost:      ${self.monthly_cost:>10,.0f}")


# ── Component functions ───────────────────────────────────────────────────────

def get_fuel_cost_5yr(city_mpg: float, highway_mpg: float,
                      state: str, annual_miles: float) -> float:
    """
    Fuel cost over 5 years.
    Uses combined MPG (55% city, 45% highway) and regional gas price.
    """
    combined_mpg = 1 / (0.55 / city_mpg + 0.45 / highway_mpg)
    region       = GAS["state_to_region"].get(state.upper(), "default")
    gas_price    = GAS.get(region, GAS["default"])
    annual_cost  = (annual_miles / combined_mpg) * gas_price
    return annual_cost * 5


def get_insurance_5yr(state: str) -> float:
    """Annual premium from state lookup × 5 years."""
    annual = INS.get(state.upper(), INS["default"])
    return annual * 5


def get_maintenance_5yr(make_name: str, vehicle_age: int) -> float:
    """
    Maintenance cost over 5 years.
    Costs increase as the car ages — accounts for warranty end at ~3 years.
    """
    # Find brand tier
    tier = "average"  # default
    for t, brands in MAINT["brand_tier"].items():
        if any(make_name.lower() == b.lower() for b in brands):
            tier = t
            break

    total = 0
    for year_offset in range(5):
        age = vehicle_age + year_offset
        # Find age bracket
        bracket = "11+yr"
        for b in MAINT["age_brackets"]:
            if b["min_age"] <= age <= b["max_age"]:
                bracket = b["label"]
                break
        total += MAINT["costs"][tier][bracket]

    return total


def get_depreciation(predicted_price: float,
                     price_model,
                     feature_row: np.ndarray,
                     mileage_idx: int,
                     year_idx: int,
                     annual_miles: float = 12_000) -> float:
    """
    Depreciation = today's predicted price − predicted price in 5 years.
    Uses the price model to predict future value by adjusting
    mileage (+60k) and year (-5) in the feature vector.
    """
    if price_model is None:
        # Fallback: use standard depreciation curve (15% per year)
        future_value = predicted_price * (0.85 ** 5)
        return predicted_price - future_value

    future_row = feature_row.copy()

    # Adjust mileage and year for 5 years from now
    # Features are scaled — approximate adjustment
    future_row[mileage_idx] += (60_000 / 50_000)  # rough scale factor
    future_row[year_idx]    -= (5 / 10)            # rough scale factor

    future_log_price = price_model.predict(future_row.reshape(1, -1))[0]
    future_price     = np.exp(future_log_price)

    return max(predicted_price - future_price, 0)


def get_loan_interest(price: float) -> float:
    """
    Interest paid on a 4-year loan with 20% down.
    Uses standard amortization formula.
    """
    principal    = price * (1 - DOWN_PAYMENT_PCT)
    monthly_rate = LOAN_RATE / 12
    n_payments   = LOAN_YEARS * 12

    # Monthly payment formula
    monthly_pmt  = (principal * monthly_rate *
                    (1 + monthly_rate) ** n_payments /
                    ((1 + monthly_rate) ** n_payments - 1))

    total_paid   = monthly_pmt * n_payments
    return total_paid - principal


def compute_tco(
    predicted_price: float,
    make_name:       str,
    year:            int,
    city_mpg:        float,
    highway_mpg:     float,
    state:           str,
    annual_miles:    float = 12_000,
    price_model      = None,
    feature_row:     np.ndarray = None,
    mileage_idx:     int = 5,
    year_idx:        int = 0,
) -> TCOResult:
    """
    Compute full 5-year TCO for a car.
    Returns a TCOResult dataclass with all components.
    """
    vehicle_age = 2024 - year  # approximate current year

    depreciation   = get_depreciation(
        predicted_price, price_model,
        feature_row, mileage_idx, year_idx, annual_miles
    ) if feature_row is not None else predicted_price * (1 - 0.85**5)

    fuel_5yr       = get_fuel_cost_5yr(
        city_mpg, highway_mpg, state, annual_miles)
    insurance_5yr  = get_insurance_5yr(state)
    maintenance_5yr = get_maintenance_5yr(make_name, vehicle_age)
    loan_interest  = get_loan_interest(predicted_price)

    total_tco    = (depreciation + fuel_5yr +
                    insurance_5yr + maintenance_5yr + loan_interest)
    monthly_cost = total_tco / 60

    return TCOResult(
        predicted_price  = predicted_price,
        depreciation     = depreciation,
        fuel_5yr         = fuel_5yr,
        insurance_5yr    = insurance_5yr,
        maintenance_5yr  = maintenance_5yr,
        loan_interest    = loan_interest,
        total_tco        = total_tco,
        monthly_cost     = monthly_cost,
    )


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":

    log.info("=" * 55)
    log.info("TCO TEST — 3 CARS")
    log.info("=" * 55)

    test_cars = [
        {
            "name":            "2020 Toyota Camry",
            "predicted_price": 24_000,
            "make_name":       "Toyota",
            "year":            2020,
            "city_mpg":        29,
            "highway_mpg":     38,
            "state":           "LA",
            "annual_miles":    12_000,
        },
        {
            "name":            "2019 Ford F-150",
            "predicted_price": 38_000,
            "make_name":       "Ford",
            "year":            2019,
            "city_mpg":        20,
            "highway_mpg":     26,
            "state":           "LA",
            "annual_miles":    12_000,
        },
        {
            "name":            "2018 BMW 3 Series",
            "predicted_price": 35_000,
            "make_name":       "BMW",
            "year":            2018,
            "city_mpg":        26,
            "highway_mpg":     36,
            "state":           "LA",
            "annual_miles":    12_000,
        },
    ]

    results = []
    for car in test_cars:
        name = car.pop("name")
        log.info(f"\n{name}")
        result = compute_tco(**car)
        result.display()
        results.append((name, result))

     # Summary comparison
    log.info("\n" + "="*55)
    log.info("SUMMARY — 5-year TCO comparison")
    log.info("="*55)
    log.info(f"  {'Car':<25} {'Price':>8} {'TCO':>10} {'Monthly':>9}")
    log.info(f"  {'-'*55}")
    for name, r in results:
        log.info(f"  {name:<25} "
                 f"${r.predicted_price:>7,.0f} "
                 f"${r.total_tco:>9,.0f} "
                 f"${r.monthly_cost:>8,.0f}")

    log.info("\ntco.py returns a 5-year cost ✅")