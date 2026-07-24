import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split

INPUT_PATH = "Cleaned_Data/reserve_model_ready.csv"

ORDINAL_FEATURES = ["Age_Band_Code", "Claim_Severity_Code", "Plan_Tier_Code"]
NUMERIC_FEATURES = ["Claims_Per_Policy_Year"]
TARGET = "log_reserve"


def get_benefit_columns(df):
    return [c for c in df.columns if c.startswith("Benefit_Category_")]


def build_feature_matrix(df, benefit_cols):
    X = df[ORDINAL_FEATURES + NUMERIC_FEATURES + benefit_cols].copy()
    dummy_to_parent = {col: "Benefit_Category" for col in benefit_cols}
    return X, dummy_to_parent


def collapse_importances(importances, feature_names, dummy_to_parent):
    scores = {}
    for name, score in zip(feature_names, importances):
        parent = dummy_to_parent.get(name, name)
        scores[parent] = scores.get(parent, 0) + score
    return pd.Series(scores)


def mutual_info_ranking(df, benefit_cols):
    X = df[ORDINAL_FEATURES + NUMERIC_FEATURES].copy()

    benefit_category = df[benefit_cols].idxmax(axis=1).str.replace("Benefit_Category_", "", regex=False)
    X["Benefit_Category"] = benefit_category.astype("category").cat.codes

    y = df[TARGET]
    discrete_mask = [col not in NUMERIC_FEATURES for col in X.columns]
    mi = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=42)
    return pd.Series(mi, index=X.columns)


def main():
    df = pd.read_csv(INPUT_PATH)
    y = df[TARGET]

    benefit_cols = get_benefit_columns(df)
    X, dummy_to_parent = build_feature_matrix(df, benefit_cols)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("random forest r2 on held-out test set:", model.score(X_test, y_test))

    impurity_scores = collapse_importances(model.feature_importances_, X.columns, dummy_to_parent)
    print("\nrandom forest impurity-based importance")
    print(impurity_scores.sort_values(ascending=False))

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    perm_scores = collapse_importances(perm.importances_mean, X.columns, dummy_to_parent)
    print("\npermutation importance (test set)")
    print(perm_scores.sort_values(ascending=False))

    mi_scores = mutual_info_ranking(df, benefit_cols)
    print("\nmutual information (label-encoded, no one-hot summing)")
    print(mi_scores.sort_values(ascending=False))

    ranking = pd.DataFrame({
        "impurity_importance": impurity_scores,
        "permutation_importance": perm_scores,
        "mutual_info": mi_scores,
    })
    ranking["avg_rank"] = ranking.rank(ascending=False).mean(axis=1)
    ranking = ranking.sort_values("avg_rank")

    print("\ncombined ranking, lower avg_rank means more important")
    print(ranking)


if __name__ == "__main__":
    main()