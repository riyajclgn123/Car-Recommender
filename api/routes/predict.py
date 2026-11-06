from fastapi import APIRouter
from api.schemas import PredictPriceRequest, PredictPriceResponse
from api.predictor import predict_price_from_specs

router = APIRouter()

@router.post("/predict-price", response_model=PredictPriceResponse)
def predict_price(req: PredictPriceRequest):
    specs = req.model_dump()
    price = predict_price_from_specs(specs)
    return PredictPriceResponse(
        predicted_price  = round(price, 2),
        predicted_log    = round(__import__("math").log(price), 4),
        make_name        = req.make_name,
        model_name       = req.model_name,
        year             = req.year,
        confidence_note  = (
            "High confidence ($10k–$60k range)"
            if 10_000 <= price <= 60_000
            else "Lower confidence — outside reliable range"
        ),
    )