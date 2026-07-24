import itertools
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = "reserve_model.joblib"
OUTPUT_PATH = "smoke_test_predictions.csv"

AGE_ORDER = ["0-18", "19-30", "31-40", "41-50", "51-60", "61-70", "71-80", "Above 80"]
TIER_ORDER = ["Bronze", "Silver", "Gold"]
SEVERITY_ORDER = ["Minor", "Moderate", "Severe"]
CLAIMS_PER_YEAR_VALUES = [1, 3, 5]


def main():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]

    benefit_categories = [c.replace("Benefit_Category_", "") for c in feature_columns if c.startswith("Benefit_Category_")]

    combinations = list(itertools.product(benefit_categories, AGE_ORDER, SEVERITY_ORDER, TIER_ORDER, CLAIMS_PER_YEAR_VALUES))

    rows = []
    for benefit_category, age_band, claim_severity, plan_tier, claims_per_year in combinations:
        row = {col: 0 for col in feature_columns}
        row["Age_Band_Code"] = AGE_ORDER.index(age_band)
        row["Claim_Severity_Code"] = SEVERITY_ORDER.index(claim_severity)
        row["Plan_Tier_Code"] = TIER_ORDER.index(plan_tier)
        row["Claims_Per_Policy_Year"] = claims_per_year
        row[f"Benefit_Category_{benefit_category}"] = 1
        rows.append(row)

    X = pd.DataFrame(rows)[feature_columns]
    pred_eur = np.expm1(model.predict(X))

    result = pd.DataFrame(combinations, columns=["Benefit_Category", "Age_Band", "Claim_Severity", "Plan_Tier", "Claims_Per_Policy_Year"])
    result["Predicted_Reserve"] = np.round(pred_eur, 2)

    print("total combinations tested:", len(result))
    print("negative predictions:", (result["Predicted_Reserve"] < 0).sum())
    print("min predicted reserve:", result["Predicted_Reserve"].min())
    print("max predicted reserve:", result["Predicted_Reserve"].max())

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nwritten to {OUTPUT_PATH}, {len(result)} rows")


if __name__ == "__main__":
    main()