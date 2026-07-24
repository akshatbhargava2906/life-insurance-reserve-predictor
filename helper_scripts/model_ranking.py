import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_PATH = "Cleaned_Data/reserve_model_ready.csv"

TARGET = "log_reserve"
NON_FEATURE_COLUMNS = ["Reserve", "log_reserve", "Claim_Severity", "is_outlier"]


def get_benefit_columns(df):
    return [c for c in df.columns if c.startswith("Benefit_Category_")]


def recover_benefit_category(df, benefit_cols):
    return df[benefit_cols].idxmax(axis=1).str.replace("Benefit_Category_", "", regex=False)


def group_mean_baseline(train_df, test_df, group_cols, target_col):
    group_means = train_df.groupby(group_cols)[target_col].mean()
    global_mean = train_df[target_col].mean()
    preds = test_df[group_cols].apply(tuple, axis=1).map(group_means)
    return preds.fillna(global_mean).values


def evaluate(name, y_test_log, y_pred_log, severity_test):
    y_test_eur = np.expm1(y_test_log)
    y_pred_eur = np.expm1(y_pred_log)

    log_mae = mean_absolute_error(y_test_log, y_pred_log)
    log_rmse = mean_squared_error(y_test_log, y_pred_log) ** 0.5
    log_r2 = r2_score(y_test_log, y_pred_log)

    eur_mae = mean_absolute_error(y_test_eur, y_pred_eur)
    eur_rmse = mean_squared_error(y_test_eur, y_pred_eur) ** 0.5
    ape = np.abs(np.asarray(y_pred_eur) - np.asarray(y_test_eur)) / np.asarray(y_test_eur)
    median_ape = np.median(ape) * 100

    print(f"\n{name}")
    print(f"  log space   mae={log_mae:.3f}  rmse={log_rmse:.3f}  r2={log_r2:.3f}")
    print(f"  euro space  mae={eur_mae:,.0f}  rmse={eur_rmse:,.0f}  median ape={median_ape:.1f}%")

    print("  euro mae by severity:")
    for sev in ["Minor", "Moderate", "Severe"]:
        mask = (severity_test == sev).values
        if mask.sum() == 0:
            continue
        sev_mae = mean_absolute_error(np.asarray(y_test_eur)[mask], np.asarray(y_pred_eur)[mask])
        print(f"    {sev:10s} mae={sev_mae:,.0f}  n={mask.sum()}")

    return {
        "model": name, "log_mae": log_mae, "log_rmse": log_rmse, "log_r2": log_r2,
        "eur_mae": eur_mae, "eur_rmse": eur_rmse, "median_ape_pct": median_ape,
    }


def main():
    df = pd.read_csv(INPUT_PATH)

    benefit_cols = get_benefit_columns(df)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]

    X = df[feature_cols]
    y = df[TARGET]

    lookup_df = df[["Claim_Severity"]].copy()
    lookup_df["Benefit_Category"] = recover_benefit_category(df, benefit_cols)
    lookup_df[TARGET] = df[TARGET]

    X_train, X_test, y_train, y_test, lookup_train, lookup_test = train_test_split(
        X, y, lookup_df, test_size=0.2, random_state=42, stratify=df["Claim_Severity"]
    )

    results = []

    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    results.append(evaluate("global mean baseline", y_test, dummy.predict(X_test), lookup_test["Claim_Severity"]))

    group_pred = group_mean_baseline(lookup_train, lookup_test, ["Benefit_Category", "Claim_Severity"], TARGET)
    results.append(evaluate("benefit category x severity lookup", y_test, group_pred, lookup_test["Claim_Severity"]))

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results.append(evaluate("linear regression", y_test, lr.predict(X_test), lookup_test["Claim_Severity"]))

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    results.append(evaluate("random forest", y_test, rf.predict(X_test), lookup_test["Claim_Severity"]))

    hgb = HistGradientBoostingRegressor(random_state=42)
    hgb.fit(X_train, y_train)
    results.append(evaluate("hist gradient boosting", y_test, hgb.predict(X_test), lookup_test["Claim_Severity"]))

    summary = pd.DataFrame(results).sort_values("eur_mae")
    print("\nmodel comparison summary, sorted by euro mae")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()