"""
api/schemas.py
Input and output schemas with validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal


# ── Predict price ─────────────────────────────────────────────────────────────
class PredictPriceRequest(BaseModel):
    make_name:           str   = Field(..., example="Toyota")
    model_name:          str   = Field(..., example="Camry")
    year:                int   = Field(..., ge=1990, le=2024)
    mileage:             float = Field(..., ge=0, le=400_000)
    body_type:           str   = Field(..., example="Sedan")
    fuel_type:           str   = Field("Gasoline", example="Gasoline")
    transmission:        str   = Field("Automatic", example="Automatic")
    wheel_system:        str   = Field("FWD", example="FWD")
    horsepower:          float = Field(..., ge=50, le=1500)
    engine_displacement: float = Field(2.5, ge=0.5, le=10.0)
    city_fuel_economy:   float = Field(25.0, ge=5, le=100)
    highway_fuel_economy:float = Field(30.0, ge=5, le=100)
    is_new:              bool  = Field(False)
    has_accidents:       bool  = Field(False)
    owner_count:         float = Field(1.0, ge=0, le=20)

    @field_validator("year")
    @classmethod
    def year_not_future(cls, v):
        if v > 2024:
            raise ValueError("Year cannot be in the future")
        return v


class PredictPriceResponse(BaseModel):
    predicted_price:  float
    predicted_log:    float
    make_name:        str
    model_name:       str
    year:             int
    confidence_note:  str


# ── TCO ───────────────────────────────────────────────────────────────────────
class TCORequest(BaseModel):
    predicted_price:   float = Field(..., ge=500, le=500_000)
    make_name:         str   = Field(..., example="Toyota")
    year:              int   = Field(..., ge=1990, le=2024)
    city_mpg:          float = Field(..., ge=5, le=100)
    highway_mpg:       float = Field(..., ge=5, le=100)
    state:             str   = Field(..., min_length=2, max_length=2,
                                     example="TX")
    annual_miles:      float = Field(12_000, ge=1_000, le=50_000)

    @field_validator("state")
    @classmethod
    def state_uppercase(cls, v):
        return v.upper()


class TCOResponse(BaseModel):
    predicted_price:   float
    depreciation:      float
    fuel_5yr:          float
    insurance_5yr:     float
    maintenance_5yr:   float
    loan_interest:     float
    total_tco:         float
    monthly_cost:      float


# ── Recommend ─────────────────────────────────────────────────────────────────
class RecommendRequest(BaseModel):
    monthly_income:          float = Field(..., ge=500,
                                           example=4000)
    max_maintenance_monthly: float = Field(150, ge=0, example=150)
    household_size:          int   = Field(1, ge=1, le=10)
    age_group:               Literal["18-34","35-54","55-64","65+"] \
                                   = Field(..., example="35-54")
    income_band:             Literal["low","middle","high"] \
                                   = Field(..., example="middle")
    area_type:               Literal["urban","suburban","rural"] \
                                   = Field(..., example="suburban")
    state:                   str   = Field(..., min_length=2,
                                           max_length=2, example="TX")
    prefer_new:              Optional[bool] = Field(None)
    n_candidates:            int   = Field(2000, ge=100, le=10_000)

    @field_validator("state")
    @classmethod
    def state_uppercase(cls, v):
        return v.upper()


class CarRecommendation(BaseModel):
    rank:             int
    make_name:        str
    model_name:       str
    year:             int
    listing_price:    float
    predicted_price:  float
    has_fair_price:   bool = True
    tco_5yr:          float
    monthly_cost:     float
    total_score:      float
    score_breakdown:  dict
    why:              str


class RecommendResponse(BaseModel):
    budget_max_price:   float
    budget_monthly:     float
    recommendations:    list[CarRecommendation]
    personas_tested:    bool = True