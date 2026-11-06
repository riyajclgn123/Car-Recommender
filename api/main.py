"""
api/main.py
FastAPI application entry point.

Run:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, tco, recommend, parse
from api.predictor import load_all_models
from dotenv import load_dotenv
load_dotenv()

import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(message)s")

app = FastAPI(
    title       = "Car Price Predictor & Recommender",
    description = "Predicts fair prices and recommends cars based on "
                  "your budget, lifestyle, and location.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Load models at startup
@app.on_event("startup")
async def startup():
    load_all_models()

app.include_router(predict.router,   tags=["Price"])
app.include_router(tco.router,       tags=["TCO"])
app.include_router(recommend.router, tags=["Recommend"])
app.include_router(parse.router, tags=["Intake"])

@app.get("/")
def root():
    return {
        "name":    "Car Price Predictor & Recommender API",
        "version": "1.0.0",
        "docs":    "/docs",
        "status":  "running",
    }

@app.get("/health")
def health():
    return {"status": "ok"}