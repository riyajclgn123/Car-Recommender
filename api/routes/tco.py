from fastapi import APIRouter
from api.schemas import TCORequest, TCOResponse
from src.cost.tco import compute_tco

router = APIRouter()

@router.post("/tco", response_model=TCOResponse)
def get_tco(req: TCORequest):
    result = compute_tco(
        predicted_price = req.predicted_price,
        make_name       = req.make_name,
        year            = req.year,
        city_mpg        = req.city_mpg,
        highway_mpg     = req.highway_mpg,
        state           = req.state,
        annual_miles    = req.annual_miles,
    )
    return TCOResponse(
        predicted_price  = result.predicted_price,
        depreciation     = round(result.depreciation, 2),
        fuel_5yr         = round(result.fuel_5yr, 2),
        insurance_5yr    = round(result.insurance_5yr, 2),
        maintenance_5yr  = round(result.maintenance_5yr, 2),
        loan_interest    = round(result.loan_interest, 2),
        total_tco        = round(result.total_tco, 2),
        monthly_cost     = round(result.monthly_cost, 2),
    )