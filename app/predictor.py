# app/predictor.py

import numpy as np
import pandas as pd
import joblib

AGE_ORDER = ["0-18", "19-30", "31-40", "41-50", "51-60", "61-70", "71-80", "Above 80"]
TIER_ORDER = ["Bronze", "Silver", "Gold"]
SEVERITY_ORDER = ["Minor", "Moderate", "Severe"]

_model = None
_feature_columns = None


def load_model(path):
    global _model, _feature_columns
    bundle = joblib.load(path)
    _model = bundle["model"]
    _feature_columns = bundle["feature_columns"]


def predict(claim):
    if _model is None:
        raise RuntimeError("call load_model() before predict()")

    row = {col: 0 for col in _feature_columns}
    row["Age_Band_Code"] = AGE_ORDER.index(claim["age_band"])
    row["Claim_Severity_Code"] = SEVERITY_ORDER.index(claim["claim_severity"])
    row["Plan_Tier_Code"] = TIER_ORDER.index(claim["plan_tier"])
    row["Claims_Per_Policy_Year"] = claim["claims_per_policy_year"]

    benefit_col = f"Benefit_Category_{claim['benefit_category']}"
    if benefit_col not in row:
        raise ValueError(f"unknown benefit category: {claim['benefit_category']}")
    row[benefit_col] = 1

    X = pd.DataFrame([row])[_feature_columns]
    log_pred = _model.predict(X)[0]
    return round(float(np.expm1(log_pred)), 2)