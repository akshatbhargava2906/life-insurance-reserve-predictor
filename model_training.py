import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_PATH = "Cleaned_Data/reserve_model_ready.csv"
MODEL_OUTPUT_PATH = "reserve_model.joblib"

TARGET = "log_reserve"
NON_FEATURE_COLUMNS = ["Reserve", "log_reserve", "Claim_Severity", "is_outlier"]

BEST_PARAMS = {
    "min_samples_leaf": 10,
    "max_leaf_nodes": 15,
    "max_iter": 300,
    "max_depth": 3,
    "learning_rate": 0.2,
    "l2_regularization": 0.1,
    "random_state": 42,
}


def load_data():
    df = pd.read_csv(INPUT_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols]
    y = df[TARGET]
    severity = df["Claim_Severity"]
    return X, y, severity, feature_cols


def evaluate(model, X_test, y_test, severity_test):
    y_pred_log = model.predict(X_test)
    y_test_eur = np.expm1(y_test)
    y_pred_eur = np.expm1(y_pred_log)

    log_mae = mean_absolute_error(y_test, y_pred_log)
    log_rmse = mean_squared_error(y_test, y_pred_log) ** 0.5
    log_r2 = r2_score(y_test, y_pred_log)

    eur_mae = mean_absolute_error(y_test_eur, y_pred_eur)
    eur_rmse = mean_squared_error(y_test_eur, y_pred_eur) ** 0.5
    ape = np.abs(np.asarray(y_pred_eur) - np.asarray(y_test_eur)) / np.asarray(y_test_eur)
    median_ape = np.median(ape) * 100

    print("held out test performance")
    print(f"  log space   mae={log_mae:.3f}  rmse={log_rmse:.3f}  r2={log_r2:.3f}")
    print(f"  euro space  mae={eur_mae:,.0f}  rmse={eur_rmse:,.0f}  median ape={median_ape:.1f}%")

    print("  euro mae by severity:")
    for sev in ["Minor", "Moderate", "Severe"]:
        mask = (severity_test == sev).values
        if mask.sum() == 0:
            continue
        sev_mae = mean_absolute_error(np.asarray(y_test_eur)[mask], np.asarray(y_pred_eur)[mask])
        print(f"    {sev:10s} mae={sev_mae:,.0f}  n={mask.sum()}")


def main():
    X, y, severity, feature_cols = load_data()

    X_train, X_test, y_train, y_test, severity_train, severity_test = train_test_split(
        X, y, severity, test_size=0.2, random_state=42, stratify=severity
    )

    report_model = HistGradientBoostingRegressor(**BEST_PARAMS)
    report_model.fit(X_train, y_train)
    evaluate(report_model, X_test, y_test, severity_test)

    final_model = HistGradientBoostingRegressor(**BEST_PARAMS)
    final_model.fit(X, y)

    joblib.dump({"model": final_model, "feature_columns": feature_cols}, MODEL_OUTPUT_PATH)
    print(f"\nfinal model trained on full data ({len(X)} rows) and saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()