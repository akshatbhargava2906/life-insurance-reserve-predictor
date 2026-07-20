"""
Exploratory data analysis for the health-claims reserve dataset.

Usage:
    python eda_reserve.py path/to/Reserve_Health_claims_synthetic.csv [--outdir outputs]

Produces:
    - A text summary printed to stdout (shape, dtypes, nulls, key stats)
    - PNG charts written to --outdir:
        01_reserve_distribution.png
        02_reserve_by_benefit_category.png
        03_reserve_by_severity_tier.png
        04_reserve_by_age_band.png
        05_base_reserve_vs_reserve.png
        06_correlation_heatmap.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AGE_ORDER = ["0-18", "19-30", "31-40", "41-50", "51-60", "61-70", "71-80", "Above 80"]
TIER_ORDER = ["Bronze", "Silver", "Gold"]
SEVERITY_ORDER = ["Minor", "Moderate", "Severe"]


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def print_overview(df: pd.DataFrame) -> None:
    
    print("SHAPE:", df.shape)
    

    print("\nDTYPES:")
    print(df.dtypes)

    print("\nNULL COUNTS:")
    print(df.isnull().sum())

    print("\nDUPLICATE ROWS:", df.duplicated().sum())

    constant_cols = [c for c in df.columns if df[c].nunique() == 1]
    print("\nCONSTANT COLUMNS (no predictive value):", constant_cols)

    print("\nNUMERIC SUMMARY:")
    print(df.describe().T)

    print("\nCATEGORICAL CARDINALITY:")
    cat_cols = df.select_dtypes(include="object").columns
    print(df[cat_cols].nunique().sort_values(ascending=False))


def print_target_stats(df: pd.DataFrame, target: str = "Reserve") -> None:
    s = df[target]
    
    print(f"TARGET: {target}")
    
    print(f"mean:   {s.mean():.2f}")
    print(f"median: {s.median():.2f}")
    print(f"std:    {s.std():.2f}")
    print(f"min:    {s.min():.2f}")
    print(f"p90:    {s.quantile(0.9):.2f}")
    print(f"p99:    {s.quantile(0.99):.2f}")
    print(f"max:    {s.max():.2f}")
    print(f"skew (raw):     {s.skew():.2f}")
    print(f"skew (log1p):   {np.log1p(s).skew():.2f}")

    q1, q3 = s.quantile([0.25, 0.75])
    upper = q3 + 1.5 * (q3 - q1)
    n_out = (s > upper).sum()
    print(f"IQR outlier threshold: {upper:.1f}  ->  {n_out} rows above it ({n_out / len(s):.1%})")


def print_correlations(df: pd.DataFrame, target: str = "Reserve") -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    corr = num.corr()
    
    print(f"NUMERIC CORRELATIONS WITH {target}")
    
    print(corr[target].sort_values(ascending=False))
    return corr


def print_groupby_effects(df: pd.DataFrame, target: str = "Reserve") -> None:
    cols = [
        "Benefit_Category",
        "Claim_Severity",
        "Plan_Tier",
        "Claim_Status",
        "Policyholder_Age_Band",
        "Incident_Type",
        "Smoker_Status",
        "BMI_Band",
        "Occupation",
        "Region",
        "Gender",
    ]
    
    print("MEAN/MEDIAN RESERVE BY CATEGORY")
    
    for col in cols:
        if col not in df.columns:
            continue
        g = df.groupby(col)[target].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)
        print(f"\n--- {col} ---")
        print(g)


def check_base_reserve_lookup(df: pd.DataFrame) -> None:
    if "Benefit_Code" not in df.columns or "Base_Reserve_EUR" not in df.columns:
        return
    nunique = df.groupby("Benefit_Code")["Base_Reserve_EUR"].nunique()
    
    print("BASE_RESERVE_EUR IS A FIXED LOOKUP PER BENEFIT_CODE:", (nunique == 1).all())
    
    if "Reserve" in df.columns:
        mult = df["Reserve"] / df["Base_Reserve_EUR"]
        print("\nReserve / Base_Reserve_EUR ratio (the 'severity multiplier'):")
        print(mult.describe())
        if "Claim_Severity" in df.columns:
            print("\nMultiplier by Claim_Severity:")
            print(df.assign(mult=mult).groupby("Claim_Severity")["mult"].agg(["mean", "median", "std"]))


def plot_reserve_distribution(df: pd.DataFrame, outdir: str, target: str = "Reserve") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(df[target], bins=40, color="#2a78d6", edgecolor="white")
    axes[0].set_title(f"{target} — raw distribution")
    axes[0].set_xlabel(target)
    axes[0].set_ylabel("count")

    axes[1].hist(np.log10(df[target]), bins=40, color="#1baf7a", edgecolor="white")
    axes[1].set_title(f"{target} — log10 distribution")
    axes[1].set_xlabel(f"log10({target})")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_reserve_distribution.png"), dpi=150)
    plt.close(fig)


def plot_reserve_by_benefit_category(df: pd.DataFrame, outdir: str, target: str = "Reserve") -> None:
    g = df.groupby("Benefit_Category")[target].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(g.index, g.values, color="#2a78d6")
    ax.set_xlabel(f"mean {target} (EUR)")
    ax.set_title(f"Mean {target} by Benefit Category")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_reserve_by_benefit_category.png"), dpi=150)
    plt.close(fig)


def plot_reserve_by_severity_tier(df: pd.DataFrame, outdir: str, target: str = "Reserve") -> None:
    g = (
        df.groupby(["Plan_Tier", "Claim_Severity"])[target]
        .mean()
        .unstack()
        .reindex(index=TIER_ORDER, columns=SEVERITY_ORDER)
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(TIER_ORDER))
    width = 0.25
    colors = ["#1baf7a", "#eda100", "#e34948"]
    for i, sev in enumerate(SEVERITY_ORDER):
        ax.bar(x + (i - 1) * width, g[sev].values, width, label=sev, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_ORDER)
    ax.set_ylabel(f"mean {target} (EUR)")
    ax.set_title(f"Mean {target} by Plan Tier x Claim Severity")
    ax.legend(title="Severity")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_reserve_by_severity_tier.png"), dpi=150)
    plt.close(fig)


def plot_reserve_by_age_band(df: pd.DataFrame, outdir: str, target: str = "Reserve") -> None:
    g = df.groupby("Policyholder_Age_Band")[target].mean().reindex(AGE_ORDER)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(g.index, g.values, color="#2a78d6")
    ax.set_ylabel(f"mean {target} (EUR)")
    ax.set_title(f"Mean {target} by Age Band")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_reserve_by_age_band.png"), dpi=150)
    plt.close(fig)


def plot_base_reserve_vs_reserve(df: pd.DataFrame, outdir: str, target: str = "Reserve") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = {"Minor": "#1baf7a", "Moderate": "#eda100", "Severe": "#e34948"}
    for sev, sub in df.groupby("Claim_Severity"):
        ax.scatter(
            sub["Base_Reserve_EUR"], sub[target],
            s=10, alpha=0.4, label=sev, color=colors.get(sev, "#2a78d6"),
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Base_Reserve_EUR (log scale)")
    ax.set_ylabel(f"{target} (log scale)")
    ax.set_title(f"Base_Reserve_EUR vs {target}")
    ax.legend(title="Severity")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_base_reserve_vs_reserve.png"), dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(corr: pd.DataFrame, outdir: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Numeric feature correlation")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="visual_output", help="Directory to write PNG charts to")
    parser.add_argument("--target", default="Reserve", help="Target column name")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_data("Docs/Reserve_Health_claims_synthetic.csv")

    print_overview(df)
    print_target_stats(df, args.target)
    corr = print_correlations(df, args.target)
    check_base_reserve_lookup(df)
    print_groupby_effects(df, args.target)

    plot_reserve_distribution(df, args.outdir, args.target)
    plot_reserve_by_benefit_category(df, args.outdir, args.target)
    plot_reserve_by_severity_tier(df, args.outdir, args.target)
    plot_reserve_by_age_band(df, args.outdir, args.target)
    plot_base_reserve_vs_reserve(df, args.outdir, args.target)
    plot_correlation_heatmap(corr, args.outdir)

    print(f"\nCharts written to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()