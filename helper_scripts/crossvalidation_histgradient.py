import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

INPUT_PATH = "Cleaned_Data/reserve_model_ready.csv"
TARGET = "log_reserve"
NON_FEATURE_COLUMNS = ["Reserve", "log_reserve", "Claim_Severity", "is_outlier"]


def load_data():
    df = pd.read_csv(INPUT_PATH)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    X = df[feature_cols]
    y = df[TARGET]
    severity = df["Claim_Severity"]
    return X, y, severity


def cross_validate_default(X_train, y_train):
    model = HistGradientBoostingRegressor(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    print("5-fold cross validation, default hyperparameters (log space)")
    print(f"  rmse: {-scores['test_rmse'].mean():.4f} +/- {scores['test_rmse'].std():.4f}")
    print(f"  mae:  {-scores['test_mae'].mean():.4f} +/- {scores['test_mae'].std():.4f}")
    print(f"  r2:   {scores['test_r2'].mean():.4f} +/- {scores['test_r2'].std():.4f}")


def tune_hyperparameters(X_train, y_train):
    model = HistGradientBoostingRegressor(random_state=42)

    param_distributions = {
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "max_iter": [100, 200, 300, 500],
        "max_leaf_nodes": [15, 31, 63, 127],
        "max_depth": [None, 3, 5, 7],
        "l2_regularization": [0.0, 0.1, 1.0],
        "min_samples_leaf": [10, 20, 50],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=40,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print("\nbest hyperparameters found")
    print(search.best_params_)
    print(f"best cv rmse (log space): {-search.best_score_:.4f}")

    return search.best_estimator_


def evaluate_holdout(model, X_test, y_test, severity_test):
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

    print("\nfinal holdout test performance")
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
    X, y, severity = load_data()

    X_train, X_test, y_train, y_test, severity_train, severity_test = train_test_split(
        X, y, severity, test_size=0.2, random_state=42, stratify=severity
    )

    cross_validate_default(X_train, y_train)
    best_model = tune_hyperparameters(X_train, y_train)
    evaluate_holdout(best_model, X_test, y_test, severity_test)


if __name__ == "__main__":
    main()