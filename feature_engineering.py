import pandas as pd

INPUT_PATH = "Cleaned_Data/reserve_cleaned.csv"
OUTPUT_PATH = "Cleaned_Data/reserve_model_ready.csv"

ORDINAL_FEATURES = ["Age_Band_Code", "Claim_Severity_Code", "Plan_Tier_Code", "Claims_Per_Policy_Year"]
TARGET = "log_reserve"


def build_features(df):
    dummies = pd.get_dummies(df["Benefit_Category"], prefix="Benefit_Category").astype(int)
    features = pd.concat([df[ORDINAL_FEATURES], dummies], axis=1)

    extras = df[["Reserve", TARGET, "Claim_Severity", "is_outlier"]]
    return pd.concat([features, extras], axis=1)


def print_report(df, result):
    print("rows:", len(result))
    print("feature columns:", [c for c in result.columns if c not in ["Reserve", TARGET, "Claim_Severity", "is_outlier"]])
    print("benefit category levels one-hot encoded:", df["Benefit_Category"].nunique())
    print("any nulls:", result.isnull().sum().sum())


def main():
    df = pd.read_csv(INPUT_PATH)
    result = build_features(df)

    print_report(df, result)

    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nwritten to {OUTPUT_PATH}, shape {result.shape}")


if __name__ == "__main__":
    main()