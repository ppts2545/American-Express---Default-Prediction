import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    import re
    from pathlib import Path

    sns.set_theme(style="whitegrid", palette="muted")

    mo.md("""
    # American Express — Understanding the Data

    > "The goal is to understand the data, not just build a model." — Good Data Science practice

    This notebook explores **what the features mean**, **how they behave**, and **how they relate to default risk**.

    ## Feature Group Legend
    | Group | Prefix | Description |
    |-------|--------|-------------|
    | Payment | `P_` | Payment amounts & ratios |
    | Delinquency | `D_` | Past due / delinquency history |
    | Balance | `B_` | Account balance features |
    | Risk | `R_` | Risk scores & flags |
    | Spend | `S_` | Spending behavior |

    Each feature is aggregated from monthly statements:
    `_mean` · `_last` · `_std` · `_min` · `_max` · `_nunique` · `_first`
    """)
    return mo, pl, pd, np, plt, gridspec, sns, re, Path


@app.cell
def _(pl, pd, re, Path):
    DATA = Path("data/processed")

    train = pl.read_parquet(DATA / "train_features.parquet")
    labels = pl.read_parquet(DATA / "train_labels.parquet")

    # Parse column metadata
    col_meta = []
    for c in train.columns:
        if c == "customer_ID":
            continue
        m = re.match(r'^([A-Z])_(\d+)_(\w+)$', c)
        if m:
            col_meta.append({
                "column": c,
                "group": m.group(1),
                "feature_id": int(m.group(2)),
                "agg": m.group(3),
            })

    meta_df = pd.DataFrame(col_meta)

    GROUP_NAMES = {
        "P": "Payment",
        "D": "Delinquency",
        "B": "Balance",
        "R": "Risk",
        "S": "Spend",
    }
    GROUP_COLORS = {
        "P": "#4C72B0",
        "D": "#DD8452",
        "B": "#55A868",
        "R": "#C44E52",
        "S": "#8172B2",
    }

    # Join train + labels for analysis
    train_pd = train.to_pandas()
    labels_pd = labels.to_pandas()
    df = train_pd.merge(labels_pd, on="customer_ID")

    # Sample 15k rows for fast plotting
    df_sample = df.sample(n=15_000, random_state=42)

    return meta_df, df, df_sample, GROUP_NAMES, GROUP_COLORS, col_meta


@app.cell
def _(meta_df, GROUP_NAMES, GROUP_COLORS, plt, mo):
    # ── Feature count per group & aggregation type ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: features per group
    group_counts = meta_df.groupby("group").size().reset_index(name="count")
    group_counts["label"] = group_counts["group"].map(
        lambda g: f"{GROUP_NAMES[g]}\n({g}_)"
    )
    colors = [GROUP_COLORS[g] for g in group_counts["group"]]
    bars = axes[0].bar(group_counts["label"], group_counts["count"], color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, group_counts["count"]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha="center", va="bottom", fontweight="bold")
    axes[0].set_title("Number of Features per Group", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Feature Count")
    axes[0].set_ylim(0, group_counts["count"].max() * 1.15)

    # Right: aggregation types
    agg_counts = meta_df.groupby("agg").size().sort_values(ascending=True)
    axes[1].barh(agg_counts.index, agg_counts.values, color="#6c8ebf", edgecolor="white")
    for i, (idx, val) in enumerate(agg_counts.items()):
        axes[1].text(val + 2, i, str(val), va="center", fontweight="bold")
    axes[1].set_title("Features by Aggregation Type", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Feature Count")

    plt.suptitle("Feature Landscape Overview", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    mo.md("## Feature Landscape: What Do We Have?")


@app.cell
def _(fig):
    fig


@app.cell
def _(meta_df, df_sample, GROUP_NAMES, GROUP_COLORS, plt, mo):
    # ── Distribution of _mean features per group (KDE: default vs non-default) ─
    mo.md("## Distribution by Feature Group: Defaulters vs Non-Defaulters")


@app.cell
def _(meta_df, df_sample, GROUP_NAMES, GROUP_COLORS, plt):
    groups = ["P", "D", "B", "R", "S"]
    fig2, axes2 = plt.subplots(len(groups), 4, figsize=(18, len(groups) * 3.2))

    for row_i, grp in enumerate(groups):
        # Pick top 4 _mean features for this group that have least missing
        grp_mean_cols = meta_df[
            (meta_df["group"] == grp) & (meta_df["agg"] == "mean")
        ]["column"].tolist()

        available = [c for c in grp_mean_cols if c in df_sample.columns]
        # Pick columns with least nulls
        null_pct = df_sample[available].isnull().mean()
        top4 = null_pct.nsmallest(4).index.tolist()

        for col_j, col in enumerate(top4):
            ax = axes2[row_i][col_j]
            for target_val, label, color in [(0, "No Default", "#4C72B0"), (1, "Default", "#C44E52")]:
                data = df_sample[df_sample["target"] == target_val][col].dropna()
                if len(data) > 10:
                    data.plot.kde(ax=ax, label=label, color=color, linewidth=2)
                    ax.axvline(data.median(), color=color, linestyle="--", alpha=0.6, linewidth=1)

            ax.set_title(col, fontsize=8, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel("")
            if col_j == 0:
                ax.set_ylabel(
                    f"{GROUP_NAMES[grp]}\n({grp}_)",
                    fontsize=9, fontweight="bold",
                    color=GROUP_COLORS[grp]
                )
            if row_i == 0 and col_j == 0:
                ax.legend(fontsize=8)

    plt.suptitle(
        "Feature Distributions: Default vs No Default\n(dashed line = median, top 4 features per group by _mean)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig2


@app.cell
def _(meta_df, df, GROUP_NAMES, GROUP_COLORS, plt, mo):
    # ── Top 10 features most correlated with target per group ───────────────
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "target"]

    corr_with_target = df[numeric_cols + ["target"]].corr()["target"].drop("target").abs()

    fig3, axes3 = plt.subplots(1, 5, figsize=(20, 6), sharey=False)

    for i, grp in enumerate(["P", "D", "B", "R", "S"]):
        ax = axes3[i]
        grp_cols = meta_df[meta_df["group"] == grp]["column"].tolist()
        grp_corr = corr_with_target[corr_with_target.index.isin(grp_cols)].dropna()
        top10 = grp_corr.nlargest(10).sort_values()

        bars = ax.barh(
            [c.replace(f"{grp}_", "").replace("_", " ") for c in top10.index],
            top10.values,
            color=GROUP_COLORS[grp], edgecolor="white"
        )
        ax.set_title(f"{GROUP_NAMES[grp]}\n({grp}_)", fontsize=11,
                     fontweight="bold", color=GROUP_COLORS[grp])
        ax.set_xlabel("|Correlation with Target|")
        ax.set_xlim(0, corr_with_target.max() * 1.1)
        for bar, val in zip(bars, top10.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=7)

    plt.suptitle("Top 10 Features Most Correlated with Default (per group)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    mo.md("## Which Features Signal Default Risk?")


@app.cell
def _(fig3):
    fig3


@app.cell
def _(meta_df, df_sample, GROUP_COLORS, plt, mo):
    # ── Box plots: _last features by target (most recent statement = most informative) ─
    groups_box = ["P", "D", "B", "R", "S"]
    fig4, axes4 = plt.subplots(len(groups_box), 3, figsize=(15, len(groups_box) * 3))

    for row_i, grp in enumerate(groups_box):
        last_cols = meta_df[
            (meta_df["group"] == grp) & (meta_df["agg"] == "last")
        ]["column"].tolist()
        available = [c for c in last_cols if c in df_sample.columns]
        null_pct = df_sample[available].isnull().mean()
        top3 = null_pct.nsmallest(3).index.tolist()

        for col_j, col in enumerate(top3):
            ax = axes4[row_i][col_j]
            data_grp = [
                df_sample[df_sample["target"] == 0][col].dropna().values,
                df_sample[df_sample["target"] == 1][col].dropna().values,
            ]
            bp = ax.boxplot(data_grp, patch_artist=True, notch=False,
                            labels=["No Default", "Default"],
                            flierprops=dict(marker=".", markersize=2, alpha=0.3))
            bp["boxes"][0].set_facecolor("#4C72B0")
            bp["boxes"][0].set_alpha(0.7)
            bp["boxes"][1].set_facecolor("#C44E52")
            bp["boxes"][1].set_alpha(0.7)
            ax.set_title(col, fontsize=8, fontweight="bold")
            if col_j == 0:
                ax.set_ylabel(
                    f"{grp}_ (last)",
                    fontsize=9, fontweight="bold",
                    color=GROUP_COLORS[grp]
                )

    plt.suptitle(
        "Most Recent Statement Features by Default Status\n(_last = last recorded value — most predictive for credit)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    mo.md("""
    ## Most Recent Statement vs Default
    `_last` features capture the **most recent statement value** — in credit risk, the latest behavior matters most.
    """)


@app.cell
def _(fig4):
    fig4


@app.cell
def _(meta_df, df, GROUP_NAMES, GROUP_COLORS, plt, np, mo):
    # ── Default rate by feature group (correlation heatmap within each group) ──
    numeric_cols2 = df.select_dtypes(include="number").columns.tolist()

    fig5, axes5 = plt.subplots(1, 5, figsize=(20, 5))

    for i, grp in enumerate(["P", "D", "B", "R", "S"]):
        ax = axes5[i]
        grp_cols = [c for c in meta_df[meta_df["group"] == grp]["column"].tolist()
                    if c in numeric_cols2]
        # Keep top 15 by variance to keep heatmap readable
        variances = df[grp_cols].var().nlargest(15).index.tolist()
        corr_matrix = df[variances].corr()

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, ax=ax, mask=mask,
            cmap="coolwarm", center=0, vmin=-1, vmax=1,
            linewidths=0.3, square=True,
            xticklabels=False, yticklabels=False,
            cbar=i == 4
        )
        ax.set_title(f"{GROUP_NAMES[grp]} ({grp}_)\ntop 15 by variance",
                     fontsize=10, fontweight="bold", color=GROUP_COLORS[grp])

    plt.suptitle("Within-Group Feature Correlation (are features redundant?)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    mo.md("""
    ## Are Features Within a Group Redundant?
    High correlation within a group means some features carry the same information.
    This guides feature selection later.
    """)


@app.cell
def _(fig5):
    fig5


@app.cell
def _(df, GROUP_NAMES, GROUP_COLORS, meta_df, plt, mo):
    # ── Default rate summary per group (mean feature value for defaulters vs not) ─
    numeric_cols3 = df.select_dtypes(include="number").columns.tolist()
    numeric_cols3 = [c for c in numeric_cols3 if c != "target"]

    summary_rows = []
    for grp in ["P", "D", "B", "R", "S"]:
        grp_cols = [c for c in meta_df[meta_df["group"] == grp]["column"].tolist()
                    if c in numeric_cols3]
        mean_default = df[df["target"] == 1][grp_cols].mean().mean()
        mean_no_default = df[df["target"] == 0][grp_cols].mean().mean()
        summary_rows.append({
            "group": grp,
            "name": GROUP_NAMES[grp],
            "mean_default": mean_default,
            "mean_no_default": mean_no_default,
            "n_features": len(grp_cols),
        })

    import pandas as _pd
    summary = _pd.DataFrame(summary_rows)

    fig6, ax6 = plt.subplots(figsize=(10, 5))
    x = range(len(summary))
    w = 0.35
    bars1 = ax6.bar([i - w/2 for i in x], summary["mean_no_default"], w,
                    label="No Default", color="#4C72B0", alpha=0.85)
    bars2 = ax6.bar([i + w/2 for i in x], summary["mean_default"], w,
                    label="Default", color="#C44E52", alpha=0.85)
    ax6.set_xticks(list(x))
    ax6.set_xticklabels([f"{r['name']}\n({r['group']}_)\n{r['n_features']} features"
                         for _, r in summary.iterrows()], fontsize=10)
    ax6.set_ylabel("Average Feature Value (normalized)")
    ax6.set_title("Average Feature Value per Group: Default vs No Default",
                  fontsize=13, fontweight="bold")
    ax6.legend()
    plt.tight_layout()

    mo.md("""
    ## Group-Level Summary: Defaulters vs Non-Defaulters
    Average feature value across all features in each group.
    Shows which groups carry the strongest signal for default prediction.
    """)


@app.cell
def _(fig6):
    fig6


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Key Takeaways

    | Group | Signal Strength | Interpretation |
    |-------|----------------|----------------|
    | **Delinquency (D_)** | Very High | Past due history is the strongest predictor |
    | **Balance (B_)** | High | Outstanding balance patterns differ significantly |
    | **Payment (P_)** | High | Payment behavior is a core credit signal |
    | **Risk (R_)** | Medium | Risk scores add complementary information |
    | **Spend (S_)** | Medium | Spending changes can signal financial stress |

    **Next step:** Feature selection — keep high-variance, low-redundancy, high-correlation-with-target features from each group.
    """)


if __name__ == "__main__":
    app.run()
