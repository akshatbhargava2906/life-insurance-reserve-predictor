# test_model.py

from app.predictor import load_model, predict

load_model("reserve_model.joblib")

result = predict({
    "benefit_category": "ICU",
    "age_band": "71-80",
    "claim_severity": "Severe",
    "plan_tier": "Gold",
    "claims_per_policy_year": 3,
})

print(result)