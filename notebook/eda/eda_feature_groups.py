import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import polars.selectors as cs
    import seaborn as sns

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
    return Path, cs, mo, np, pd, pl, plt, re, sns


@app.cell
def _(Path, cs, pd, pl, re):
    ROOT = Path(__file__).resolve().parent
    while not (ROOT / "pixi.toml").exists():
        ROOT = ROOT.parent
    DATA = ROOT / "data/processed"

    train = pl.read_parquet(DATA / "train_features.parquet")
    labels = pl.read_parquet(DATA / "train_labels.parquet")

    # Join in Polars — df stays as Polars throughout, no .to_pandas() on the full frame
    df = train.join(labels.select(["customer_ID", "target"]), on="customer_ID")

    # Sample in Polars (seed= is Polars' equivalent of random_state=)
    df_sample = df.sample(n=15_000, seed=42)

    # Feature metadata from column names only — no data needed
    col_meta = []
    for _c in df.select(cs.numeric().exclude("target")).columns:
        _m = re.match(r"^([A-Z])_(\d+)_(\w+)$", _c)
        if _m:
            col_meta.append({"column": _c, "group": _m.group(1), "agg": _m.group(3)})

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

    return GROUP_COLORS, GROUP_NAMES, df, df_sample, meta_df


@app.cell
def _(GROUP_COLORS, GROUP_NAMES, meta_df, mo, plt):
    def _make_landscape():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        group_counts = meta_df.groupby("group").size().reset_index(name="count")
        group_counts["label"] = group_counts["group"].map(
            lambda g: f"{GROUP_NAMES[g]}\n({g}_)"
        )
        colors = [GROUP_COLORS[g] for g in group_counts["group"]]
        bars = axes[0].bar(
            group_counts["label"],
            group_counts["count"],
            color=colors,
            edgecolor="white",
            linewidth=1.5,
        )
        for bar, val in zip(bars, group_counts["count"]):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(val),
                ha="center",
                va="bottom",
                fontweight="bold",
            )
        axes[0].set_title(
            "Number of Features per Group", fontsize=13, fontweight="bold"
        )
        axes[0].set_ylabel("Feature Count")
        axes[0].set_ylim(0, group_counts["count"].max() * 1.15)

        agg_counts = meta_df.groupby("agg").size().sort_values(ascending=True)
        axes[1].barh(
            agg_counts.index, agg_counts.values, color="#6c8ebf", edgecolor="white"
        )
        for i, (_, val) in enumerate(agg_counts.items()):
            axes[1].text(val + 2, i, str(val), va="center", fontweight="bold")
        axes[1].set_title(
            "Features by Aggregation Type", fontsize=13, fontweight="bold"
        )
        axes[1].set_xlabel("Feature Count")

        plt.suptitle(
            "Feature Landscape Overview", fontsize=15, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        return fig

    fig_landscape = _make_landscape()
    mo.md("## Feature Landscape: What Do We Have?")
    return (fig_landscape,)


@app.cell
def _(fig_landscape):
    fig_landscape


@app.cell
def _(mo):
    mo.md("## Distribution by Feature Group: Defaulters vs Non-Defaulters")


@app.cell
def _(GROUP_COLORS, GROUP_NAMES, df_sample, meta_df, pl, plt):
    def _make_kde():
        groups = ["P", "D", "B", "R", "S"]
        fig, axes = plt.subplots(len(groups), 4, figsize=(18, len(groups) * 3.2))

        for row_i, grp in enumerate(groups):
            grp_mean_cols = meta_df[
                (meta_df["group"] == grp) & (meta_df["agg"] == "mean")
            ]["column"].tolist()
            available = [c for c in grp_mean_cols if c in df_sample.columns]

            # Compute null % in Polars, convert only the tiny summary series
            null_pct = (
                df_sample.select(
                    [pl.col(c).is_null().mean().alias(c) for c in available]
                )
                .unpivot(variable_name="col", value_name="null_pct")
                .sort("null_pct")
            )
            top4 = null_pct.head(4)["col"].to_list()

            for col_j, col in enumerate(top4):
                ax = axes[row_i][col_j]
                for target_val, label, color in [
                    (0, "No Default", "#4C72B0"),
                    (1, "Default", "#C44E52"),
                ]:
                    # Convert only 1 column to numpy — minimal memory usage
                    data = (
                        df_sample.filter(pl.col("target") == target_val)[col]
                        .drop_nulls()
                        .to_numpy()
                    )
                    if len(data) > 10:
                        import pandas as _pd

                        _pd.Series(data).plot.kde(
                            ax=ax, label=label, color=color, linewidth=2
                        )
                        ax.axvline(
                            float(data.mean()),
                            color=color,
                            linestyle="--",
                            alpha=0.6,
                            linewidth=1,
                        )
                ax.set_title(col, fontsize=8, fontweight="bold")
                ax.set_xlabel("")
                ax.set_ylabel("")
                if col_j == 0:
                    ax.set_ylabel(
                        f"{GROUP_NAMES[grp]}\n({grp}_)",
                        fontsize=9,
                        fontweight="bold",
                        color=GROUP_COLORS[grp],
                    )
                if row_i == 0 and col_j == 0:
                    ax.legend(fontsize=8)

        plt.suptitle(
            "Feature Distributions: Default vs No Default\n(dashed = median, top 4 per group by _mean)",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        return fig

    fig_kde = _make_kde()
    return (fig_kde,)


@app.cell
def _(fig_kde):
    fig_kde


@app.cell
def _(GROUP_COLORS, GROUP_NAMES, cs, df_sample, meta_df, mo, pl, plt):
    def _make_corr():
        # Compute correlation with target entirely in Polars — no pandas conversion
        numeric_cols = df_sample.select(cs.numeric().exclude("target")).columns
        corr_df = (
            df_sample.select([pl.corr(c, "target").alias(c) for c in numeric_cols])
            .unpivot(variable_name="feature", value_name="correlation")
            .with_columns(pl.col("correlation").abs())
            .sort("correlation", descending=True)
        )

        fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=False)
        for i, grp in enumerate(["P", "D", "B", "R", "S"]):
            ax = axes[i]
            grp_cols = set(meta_df[meta_df["group"] == grp]["column"].tolist())
            grp_corr = corr_df.filter(pl.col("feature").is_in(grp_cols)).head(10)

            features = grp_corr["feature"].to_list()
            values = grp_corr["correlation"].to_list()
            labels = [f.replace(f"{grp}_", "").replace("_", " ") for f in features]

            bars = ax.barh(labels, values, color=GROUP_COLORS[grp], edgecolor="white")
            ax.set_title(
                f"{GROUP_NAMES[grp]}\n({grp}_)",
                fontsize=11,
                fontweight="bold",
                color=GROUP_COLORS[grp],
            )
            ax.set_xlabel("|Correlation with Target|")
            for bar, val in zip(bars, values):
                ax.text(
                    val + 0.001,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center",
                    fontsize=7,
                )

        plt.suptitle(
            "Top 10 Features Most Correlated with Default (per group)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    fig_corr = _make_corr()
    mo.md("## Which Features Signal Default Risk?")
    return (fig_corr,)


@app.cell
def _(fig_corr):
    fig_corr


@app.cell
def _(GROUP_COLORS, df_sample, meta_df, mo, np, pl, plt):
    def _make_boxplots():
        groups = ["P", "D", "B", "R", "S"]
        fig, axes = plt.subplots(len(groups), 3, figsize=(15, len(groups) * 3))

        for row_i, grp in enumerate(groups):
            last_cols = meta_df[(meta_df["group"] == grp) & (meta_df["agg"] == "last")][
                "column"
            ].tolist()
            available = [c for c in last_cols if c in df_sample.columns]

            null_pct = (
                df_sample.select(
                    [pl.col(c).is_null().mean().alias(c) for c in available]
                )
                .unpivot(variable_name="col", value_name="null_pct")
                .sort("null_pct")
            )
            top3 = null_pct.head(3)["col"].to_list()

            for col_j, col in enumerate(top3):
                ax = axes[row_i][col_j]
                # Convert only 1 column per group to numpy — minimal memory
                data0 = (
                    df_sample.filter(pl.col("target") == 0)[col].drop_nulls().to_numpy()
                )
                data1 = (
                    df_sample.filter(pl.col("target") == 1)[col].drop_nulls().to_numpy()
                )

                bp = ax.boxplot(
                    [data0, data1],
                    patch_artist=True,
                    notch=False,
                    labels=["No Default", "Default"],
                    flierprops=dict(marker=".", markersize=2, alpha=0.3),
                )
                bp["boxes"][0].set_facecolor("#4C72B0")
                bp["boxes"][0].set_alpha(0.7)
                bp["boxes"][1].set_facecolor("#C44E52")
                bp["boxes"][1].set_alpha(0.7)
                ax.set_title(col, fontsize=8, fontweight="bold")
                if col_j == 0:
                    ax.set_ylabel(
                        f"{grp}_ (last)",
                        fontsize=9,
                        fontweight="bold",
                        color=GROUP_COLORS[grp],
                    )

        plt.suptitle(
            "Most Recent Statement Features by Default Status\n(_last = most predictive for credit)",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        return fig

    fig_box = _make_boxplots()
    mo.md("""
    ## Most Recent Statement vs Default
    `_last` = most recent statement value — in credit risk, latest behavior matters most.
    """)
    return (fig_box,)


@app.cell
def _(fig_box):
    fig_box


@app.cell
def _(GROUP_COLORS, GROUP_NAMES, df_sample, meta_df, mo, np, pl, plt, sns):
    def _make_heatmap():
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        for i, grp in enumerate(["P", "D", "B", "R", "S"]):
            ax = axes[i]
            grp_cols = [
                c
                for c in meta_df[meta_df["group"] == grp]["column"].tolist()
                if c in df_sample.columns
            ]

            # Compute variance in Polars, pick top 15
            var_df = (
                df_sample.select([pl.col(c).var().alias(c) for c in grp_cols])
                .unpivot(variable_name="col", value_name="var")
                .sort("var", descending=True)
            )
            top15 = var_df.head(15)["col"].to_list()

            # Convert only 15 columns to pandas for seaborn heatmap
            corr_matrix = df_sample.select(top15).to_pandas().corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                ax=ax,
                mask=mask,
                cmap="coolwarm",
                center=0,
                vmin=-1,
                vmax=1,
                linewidths=0.3,
                square=True,
                xticklabels=False,
                yticklabels=False,
                cbar=i == 4,
            )
            ax.set_title(
                f"{GROUP_NAMES[grp]} ({grp}_)\ntop 15 by variance",
                fontsize=10,
                fontweight="bold",
                color=GROUP_COLORS[grp],
            )

        plt.suptitle(
            "Within-Group Feature Correlation (are features redundant?)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    fig_heat = _make_heatmap()
    mo.md("""
    ## Are Features Within a Group Redundant?
    High correlation = features carry the same information → guides feature selection later.
    """)
    return (fig_heat,)


@app.cell
def _(fig_heat):
    fig_heat


@app.cell
def _(GROUP_NAMES, df_sample, meta_df, mo, pd, pl, plt):
    def _make_summary():
        rows = []
        for grp in ["P", "D", "B", "R", "S"]:
            grp_cols = [
                c
                for c in meta_df[meta_df["group"] == grp]["column"].tolist()
                if c in df_sample.columns
            ]
            # Compute group means entirely in Polars
            mean_default = (
                df_sample.filter(pl.col("target") == 1)
                .select(grp_cols)
                .mean()
                .to_numpy()
                .mean()
            )
            mean_no_default = (
                df_sample.filter(pl.col("target") == 0)
                .select(grp_cols)
                .mean()
                .to_numpy()
                .mean()
            )
            rows.append(
                {
                    "group": grp,
                    "name": GROUP_NAMES[grp],
                    "mean_default": mean_default,
                    "mean_no_default": mean_no_default,
                    "n_features": len(grp_cols),
                }
            )

        summary = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(summary))
        w = 0.35
        ax.bar(
            [i - w / 2 for i in x],
            summary["mean_no_default"],
            w,
            label="No Default",
            color="#4C72B0",
            alpha=0.85,
        )
        ax.bar(
            [i + w / 2 for i in x],
            summary["mean_default"],
            w,
            label="Default",
            color="#C44E52",
            alpha=0.85,
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [
                f"{r['name']}\n({r['group']}_)\n{r['n_features']} features"
                for _, r in summary.iterrows()
            ],
            fontsize=10,
        )
        ax.set_ylabel("Average Feature Value")
        ax.set_title(
            "Average Feature Value per Group: Default vs No Default",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()
        return fig

    fig_summary = _make_summary()
    mo.md("""
    ## Group-Level Summary: Defaulters vs Non-Defaulters
    Average feature value across all features in each group.
    """)
    return (fig_summary,)


@app.cell
def _(fig_summary):
    fig_summary


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
