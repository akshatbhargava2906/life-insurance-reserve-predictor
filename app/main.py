# main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.predictor import load_model, predict

MODEL_PATH = "reserve_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model(MODEL_PATH)
    yield


app = FastAPI(title="Reserve Predictor", lifespan=lifespan)


class ClaimRequest(BaseModel):
    benefit_category: str
    age_band: str
    claim_severity: str
    plan_tier: str
    claims_per_policy_year: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_reserve(claim: ClaimRequest):
    try:
        reserve = predict(claim.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predicted_reserve": reserve}