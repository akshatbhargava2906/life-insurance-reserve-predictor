import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kruskal

INPUT_PATH = "Cleaned_Data/reserve_cleaned.csv"
OUTDIR = "eda_outputs"

CAT_COLS = [
    "Benefit_Category", "Claim_Severity", "Plan_Tier", "Policyholder_Age_Band",
    "Incident_Type", "Occupation", "Region", "Gender", "Smoker_Status",
    "BMI_Band", "Claim_Status",
]

AGE_ORDER = ["0-18", "19-30", "31-40", "41-50", "51-60", "61-70", "71-80", "Above 80"]
TIER_ORDER = ["Bronze", "Silver", "Gold"]
SEVERITY_ORDER = ["Minor", "Moderate", "Severe"]


def boxplot_by_category(df, col, order, outdir):
    data = [df.loc[df[col] == level, "log_reserve"] for level in order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    ax.set_ylabel("log_reserve")
    ax.set_title(f"log(Reserve) spread by {col}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"box_{col}.png"), dpi=150)
    plt.close(fig)


def boxplot_benefit_category(df, outdir):
    order = df.groupby("Benefit_Category")["log_reserve"].median().sort_values().index.tolist()
    data = [df.loc[df["Benefit_Category"] == level, "log_reserve"] for level in order]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.boxplot(data, vert=False, showfliers=True)
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order)
    ax.set_xlabel("log_reserve")
    ax.set_title("log(Reserve) spread by Benefit_Category")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "box_Benefit_Category.png"), dpi=150)
    plt.close(fig)


def outlier_breakdown(df):
    print("outlier share by Benefit_Category")
    g = df.groupby("Benefit_Category")["is_outlier"].mean().sort_values(ascending=False)
    print((g * 100).round(1))

    print("\noutlier share by Claim_Severity")
    g = df.groupby("Claim_Severity", observed=True)["is_outlier"].mean().sort_values(ascending=False)
    print((g * 100).round(1))


def significance_tests(df):
    print("kruskal-wallis test: does each categorical field affect Reserve significantly")
    results = []
    for col in CAT_COLS:
        groups = [g["Reserve"].values for _, g in df.groupby(col, observed=True)]
        stat, p = kruskal(*groups)
        results.append((col, stat, p))

    results.sort(key=lambda r: r[2])
    for col, stat, p in results:
        flag = "significant" if p < 0.05 else "not significant"
        print(f"{col:28s} H={stat:10.1f}  p={p:.2e}  {flag}")


def correlation_on_transformed(df, outdir):
    cols = [
        "log_reserve", "log_base_reserve", "log_sum_insured",
        "Age_Band_Code", "Plan_Tier_Code", "Claim_Severity_Code",
        "Policy_Tenure_Years", "Reporting_Lag_Days", "Claims_Per_Policy_Year",
    ]
    corr = df[cols].corr()
    print("\ncorrelation with log_reserve (transformed/encoded features)")
    print(corr["log_reserve"].sort_values(ascending=False))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("correlation on log/ordinal-encoded features")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "correlation_transformed.png"), dpi=150)
    plt.close(fig)


def log_reserve_distribution(df, outdir):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["log_reserve"], bins=40, color="#2a78d6", edgecolor="white")
    ax.set_xlabel("log_reserve")
    ax.set_ylabel("count")
    ax.set_title("log(Reserve) distribution, cleaned data")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "log_reserve_distribution.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(INPUT_PATH)

    df["Policyholder_Age_Band"] = pd.Categorical(df["Policyholder_Age_Band"], categories=AGE_ORDER, ordered=True)
    df["Plan_Tier"] = pd.Categorical(df["Plan_Tier"], categories=TIER_ORDER, ordered=True)
    df["Claim_Severity"] = pd.Categorical(df["Claim_Severity"], categories=SEVERITY_ORDER, ordered=True)

    log_reserve_distribution(df, OUTDIR)

    boxplot_by_category(df, "Claim_Severity", SEVERITY_ORDER, OUTDIR)
    boxplot_by_category(df, "Plan_Tier", TIER_ORDER, OUTDIR)
    boxplot_by_category(df, "Policyholder_Age_Band", AGE_ORDER, OUTDIR)
    boxplot_benefit_category(df, OUTDIR)

    outlier_breakdown(df)
    print()
    significance_tests(df)
    correlation_on_transformed(df, OUTDIR)

    print(f"\ncharts written to {os.path.abspath(OUTDIR)}")


if __name__ == "__main__":
    main()