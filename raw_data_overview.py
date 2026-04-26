import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pyarrow as pa
    from pathlib import Path

    mo.md("# American Express — Data Overview & Missing Value Analysis")
    return Path, mo, pl, plt


@app.cell
def _(Path, mo, pl):
    DATA = Path("data/processed")

    train = pl.read_parquet(DATA / "train_features.parquet")
    labels = pl.read_parquet(DATA / "train_labels.parquet")
    test = pl.read_parquet(DATA / "test_features.parquet")

    mo.md(f"""
    ## Dataset Shape
    | Dataset | Rows | Columns |
    |---------|------|---------|
    | Train features | {train.shape[0]:,} | {train.shape[1]:,} |
    | Train labels   | {labels.shape[0]:,} | {labels.shape[1]:,} |
    | Test features  | {test.shape[0]:,} | {test.shape[1]:,} |
    """)
    return labels, train


@app.cell
def _(mo, train):
    def _():
        # Count columns by dtype
        dtype_counts = {}
        for col in train.columns:
            t = str(train[col].dtype)
            dtype_counts[t] = dtype_counts.get(t, 0) + 1

        rows = "\n".join(f"| {k} | {v} |" for k, v in sorted(dtype_counts.items()))
        return mo.md(f"""
        ## Column Data Types
        | dtype | count |
        |-------|-------|
        {rows}
        """)


    _()
    return


@app.cell
def _(mo, pl, train):
    # Missing value % per column
    total = train.shape[0]
    missing = (
        train.select([
            pl.col(c).is_null().sum().alias(c)
            for c in train.columns
        ])
        .unpivot(variable_name="column", value_name="missing_count")
        .with_columns([
            (pl.col("missing_count") / total * 100).round(2).alias("missing_pct"),
            pl.col("column").str.slice(0, 1).alias("feature_group"),
        ])
        .filter(pl.col("missing_count") > 0)
        .sort("missing_pct", descending=True)
    )

    total_cols = train.shape[1]
    missing_cols = missing.shape[0]

    mo.md(f"""
    ## Missing Value Summary
    - Total columns: **{total_cols}**
    - Columns with missing values: **{missing_cols}** ({missing_cols/total_cols*100:.1f}%)
    - Columns fully complete: **{total_cols - missing_cols}**
    """)
    return (missing,)


@app.cell
def _(missing, mo, pl):
    # Group missing by feature prefix (D_, B_, P_, R_, S_)
    group_summary = (
        missing
        .group_by("feature_group")
        .agg([
            pl.len().alias("columns_with_missing"),
            pl.col("missing_pct").mean().round(2).alias("avg_missing_pct"),
            pl.col("missing_pct").max().round(2).alias("max_missing_pct"),
        ])
        .sort("avg_missing_pct", descending=True)
    )

    mo.md("## Missing Values by Feature Group")
    return (group_summary,)


@app.cell
def _(group_summary):
    group_summary
    return


@app.cell
def _(missing, mo, pl, plt):
    # Only visualize columns with >30% missing — actionable threshold
    high_missing = missing.filter(pl.col("missing_pct") > 30)

    fig, ax = plt.subplots(figsize=(10, max(4, len(high_missing) * 0.25)))

    if len(high_missing) > 0:
        df_plot = high_missing.to_pandas().sort_values("missing_pct")
        colors = ["#d62728" if x > 70 else "#ff7f0e" if x > 50 else "#1f77b4"
                  for x in df_plot["missing_pct"]]
        ax.barh(df_plot["column"], df_plot["missing_pct"], color=colors)
        ax.axvline(50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
        ax.axvline(70, color="darkred", linestyle="--", alpha=0.5, label="70% threshold")
        ax.set_xlabel("Missing %")
        ax.set_title(f"Columns with >30% Missing Values ({len(high_missing)} columns)")
        ax.legend()
        plt.tight_layout()
    else:
        ax.text(0.5, 0.5, "No columns with >30% missing", ha="center", va="center")

    mo.md(f"## Columns with >30% Missing ({len(high_missing)} columns)")
    return (fig,)


@app.cell
def _(fig):
    fig
    return


@app.cell
def _(missing, mo, pl):
    # Buckets: how bad is the missingness?
    buckets = {
        "0–10%":  missing.filter(pl.col("missing_pct") <= 10).shape[0],
        "10–30%": missing.filter((pl.col("missing_pct") > 10) & (pl.col("missing_pct") <= 30)).shape[0],
        "30–70%": missing.filter((pl.col("missing_pct") > 30) & (pl.col("missing_pct") <= 70)).shape[0],
        ">70%":   missing.filter(pl.col("missing_pct") > 70).shape[0],
    }

    rows = "\n".join(f"| {k} | {v} |" for k, v in buckets.items())
    mo.md(f"""
    ## Missing Severity Breakdown
    | Missing Range | # Columns |
    |---------------|-----------|
    {rows}

    **Cleaning strategy:**
    - **0–10%** → safe to fill with median/mode
    - **10–30%** → fill with median or flag with indicator column
    - **30–70%** → consider dropping or use model that handles nulls (XGBoost/LightGBM)
    - **>70%** → strong candidate for dropping
    """)
    return


@app.cell
def _(labels, mo, plt):
    # Target label distribution
    label_counts = labels["target"].value_counts().sort("target")

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    ax2.bar(
        label_counts["target"].cast(str).to_list(),
        label_counts["count"].to_list(),
        color=["#2196F3", "#F44336"]
    )
    ax2.set_xlabel("Default (1 = defaulted)")
    ax2.set_ylabel("Count")
    ax2.set_title("Target Label Distribution")

    total_labels = label_counts["count"].sum()
    for row in label_counts.iter_rows(named=True):
        ax2.text(str(row["target"]), row["count"] + 500,
                 f'{row["count"]/total_labels*100:.1f}%', ha="center")
    plt.tight_layout()

    mo.md("## Target Label Distribution")
    return (fig2,)


@app.cell
def _(fig2):
    fig2
    return


@app.cell
def _(missing, mo, pl):
    # Columns safe to drop (>70% missing)
    drop_candidates = missing.filter(pl.col("missing_pct") > 70)["column"].to_list()

    mo.md(f"""
    ## Recommended Columns to Drop (>70% missing)
    Found **{len(drop_candidates)}** columns:

    `{'`, `'.join(drop_candidates) if drop_candidates else 'None'}`
    """)
    return (drop_candidates,)


@app.cell
def _(drop_candidates, mo, pl, train):
    # Clean: drop high-missing columns
    train_cleaned = train.drop(drop_candidates)

    # For remaining numeric nulls: fill with median
    numeric_cols = [
        c for c in train_cleaned.columns
        if train_cleaned[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
        and train_cleaned[c].is_null().sum() > 0
    ]

    fills = {c: train_cleaned[c].median() for c in numeric_cols}
    train_cleaned = train_cleaned.with_columns([
        pl.col(col).fill_null(val)
        for col, val in fills.items()
        if col in train_cleaned.columns
    ])

    # For string/categorical nulls: fill with "Unknown"
    str_cols = [
        c for c in train_cleaned.columns
        if train_cleaned[c].dtype == pl.String
        and train_cleaned[c].is_null().sum() > 0
    ]
    for c in str_cols:
        train_cleaned = train_cleaned.with_columns(pl.col(c).fill_null("Unknown"))

    remaining_nulls = train_cleaned.null_count().sum_horizontal()[0]

    mo.md(f"""
    ## After Cleaning
    | | Before | After |
    |-|--------|-------|
    | Columns | {train.shape[1]} | {train_cleaned.shape[1]} |
    | Remaining nulls | — | {remaining_nulls} |

    Cleaned dataset ready. You can save it with:
    ```python
    train_cleaned.write_parquet("data/processed/train_cleaned.parquet")
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
