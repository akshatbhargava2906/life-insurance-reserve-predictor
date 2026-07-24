import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "reserve_model.joblib"

AGE_ORDER = ["0-18", "19-30", "31-40", "41-50", "51-60", "61-70", "71-80", "Above 80"]
TIER_ORDER = ["Bronze", "Silver", "Gold"]
SEVERITY_ORDER = ["Minor", "Moderate", "Severe"]


def build_feature_vector(feature_columns, benefit_category, age_band, claim_severity, plan_tier, claims_per_policy_year):
    row = {col: 0 for col in feature_columns}

    row["Age_Band_Code"] = AGE_ORDER.index(age_band)
    row["Claim_Severity_Code"] = SEVERITY_ORDER.index(claim_severity)
    row["Plan_Tier_Code"] = TIER_ORDER.index(plan_tier)
    row["Claims_Per_Policy_Year"] = claims_per_policy_year

    benefit_col = f"Benefit_Category_{benefit_category}"
    if benefit_col not in row:
        raise ValueError(f"unknown benefit category: {benefit_category}")
    row[benefit_col] = 1

    return pd.DataFrame([row])[feature_columns]


def predict_reserve(benefit_category, age_band, claim_severity, plan_tier, claims_per_policy_year, model_path=MODEL_PATH):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    X = build_feature_vector(feature_columns, benefit_category, age_band, claim_severity, plan_tier, claims_per_policy_year)
    log_pred = model.predict(X)[0]
    return np.expm1(log_pred)


def main():
    test_cases = [
        {"benefit_category": "Dental treatment", "age_band": "31-40", "claim_severity": "Minor", "plan_tier": "Bronze", "claims_per_policy_year": 1},
        {"benefit_category": "Consultation", "age_band": "19-30", "claim_severity": "Minor", "plan_tier": "Silver", "claims_per_policy_year": 2},
        {"benefit_category": "Surgical fee", "age_band": "51-60", "claim_severity": "Moderate", "plan_tier": "Silver", "claims_per_policy_year": 2},
        {"benefit_category": "ICU", "age_band": "71-80", "claim_severity": "Severe", "plan_tier": "Gold", "claims_per_policy_year": 3},
        {"benefit_category": "Brain and nervous system", "age_band": "Above 80", "claim_severity": "Severe", "plan_tier": "Gold", "claims_per_policy_year": 4},
    ]

    for case in test_cases:
        predicted = predict_reserve(**case)
        print(case)
        print(f"  predicted reserve: EUR {predicted:,.2f}\n")


if __name__ == "__main__":
    main()