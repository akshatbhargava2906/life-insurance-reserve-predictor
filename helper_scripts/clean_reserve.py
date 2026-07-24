import numpy as np
import pandas as pd

INPUT_PATH = "Docs/Reserve_Health_claims_synthetic.csv"
OUTPUT_PATH = "Cleaned_Data/reserve_cleaned.csv"

AGE_ORDER = ["0-18", "19-30", "31-40", "41-50", "51-60", "61-70", "71-80", "Above 80"]
TIER_ORDER = ["Bronze", "Silver", "Gold"]
SEVERITY_ORDER = ["Minor", "Moderate", "Severe"]


def clean(df):
    df = df.copy()
    df = df.drop_duplicates()
    df = df.drop(columns=["LOB", "Policy_Product", "Benefit_Code"])
    str_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    df["Policyholder_Age_Band"] = pd.Categorical(df["Policyholder_Age_Band"], categories=AGE_ORDER, ordered=True)
    df["Plan_Tier"] = pd.Categorical(df["Plan_Tier"], categories=TIER_ORDER, ordered=True)
    df["Claim_Severity"] = pd.Categorical(df["Claim_Severity"], categories=SEVERITY_ORDER, ordered=True)
    df["Age_Band_Code"] = df["Policyholder_Age_Band"].cat.codes
    df["Plan_Tier_Code"] = df["Plan_Tier"].cat.codes
    df["Claim_Severity_Code"] = df["Claim_Severity"].cat.codes
    df["log_reserve"] = np.log1p(df["Reserve"])
    df["log_base_reserve"] = np.log1p(df["Base_Reserve_EUR"])
    df["log_sum_insured"] = np.log1p(df["Sum_Insured_EUR"])
    q1 = df["Reserve"].quantile(0.25)
    q3 = df["Reserve"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    df["is_outlier"] = df["Reserve"] > upper
    return df


def print_report(raw, cleaned):
    print("rows before:", len(raw), "rows after:", len(cleaned))
    print("duplicates removed:", len(raw) - raw.drop_duplicates().shape[0])
    print("columns dropped: LOB, Policy_Product, Benefit_Code")
    print("outlier claims flagged:", cleaned["is_outlier"].sum(), f"({cleaned['is_outlier'].mean():.1%})")

    for col in ["Policyholder_Age_Band", "Plan_Tier", "Claim_Severity"]:
        print(f"\n{col} categories in order:")
        print(list(cleaned[col].cat.categories))

    print("\nany nulls left:", cleaned.isnull().sum().sum())
    print("any negative numeric values:")
    num_cols = cleaned.select_dtypes(include=[np.number]).columns
    print((cleaned[num_cols] < 0).sum()[lambda s: s > 0])


def main():
    raw = pd.read_csv(INPUT_PATH)
    cleaned = clean(raw)

    print_report(raw, cleaned)

    cleaned.to_csv(OUTPUT_PATH, index=False)
    print(f"\nwritten to {OUTPUT_PATH}, shape {cleaned.shape}")


if __name__ == "__main__":
    main()