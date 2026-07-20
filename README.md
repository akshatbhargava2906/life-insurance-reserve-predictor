# Life Insurance Reserve Predictor

Predicts the reserve amount (EUR) that should be held against a health insurance claim, based on claim and policyholder attributes.

## Data

Source file: `Reserve_Health_claims_synthetic.csv` — 3,000 synthetic health claims, 20 columns, single line of business (health / ZHLT product). Not committed to the repo; place it locally before running any script.

Target column: `Reserve`.

## Pipeline

1. `clean_reserve.py` — drops constant/redundant columns, orders the ordinal categoricals (age band, plan tier, severity), adds log1p-transformed and ordinal-coded columns, flags statistical outliers. Reads `Reserve_Health_claims_synthetic.csv`, writes `reserve_cleaned.csv`.
2. `eda_reserve.py` — exploratory analysis on the raw data: distribution shape, correlations, category breakdowns. Writes charts to `outputs/`.
3. `eda_cleaned.py` — deeper analysis on the cleaned data: boxplots by category, outlier concentration by category/severity, Kruskal-Wallis significance tests per categorical field, correlation on the log/ordinal-encoded features. Writes charts to `eda_outputs/`.

Run each script with plain `python <script>.py` — input/output paths are hardcoded at the top of each file, edit them to point at your local files.

## Key findings so far

- `Reserve` is heavily right-skewed (skew 3.58 raw, 0.33 after log1p) — model on `log1p(Reserve)`, invert predictions with `expm1`.
- `Base_Reserve_EUR` (a fixed lookup per `Benefit_Code`/`Benefit_Category`) is the dominant driver — log-log correlation with `Reserve` is 0.94.
- `Claim_Severity` and `Plan_Tier` each scale the reserve multiplicatively on top of the base amount.
- `Policyholder_Age_Band` increases reserve roughly monotonically with age.
- Statistically significant drivers (Kruskal-Wallis, p<0.05): `Benefit_Category`, `Incident_Type`, `Policyholder_Age_Band`, `Claim_Severity`, `Plan_Tier`, `Occupation`.
- Not significant: `Gender`, `Claim_Status`, `BMI_Band`, `Region`, `Smoker_Status`.
- `LOB`, `Policy_Product` are constant columns; `Benefit_Code` is redundant with `Benefit_Category` — all dropped during cleaning.