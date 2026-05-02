import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
    })

    GROUP_COLORS = {
        "P": "#4C72B0",
        "D": "#DD8452",
        "B": "#55A868",
        "R": "#C44E52",
        "S": "#8172B2",
    }
    GROUP_NAMES = {
        "P": "Payment",
        "D": "Delinquency",
        "B": "Balance",
        "R": "Risk",
        "S": "Spend",
    }
    AGG_COLORS = {
        "mean":             "#4C72B0",
        "last":             "#55A868",
        "max":              "#8172B2",
        "min":              "#DD8452",
        "std":              "#C44E52",
        "nunique":          "#937860",
        "last_minus_first": "#bcbd22",
        "last_minus_mean":  "#17becf",
    }

    mo.md("""
    # Class Comparison — Default vs Non-Default

    > Notebook นี้เจาะลึกความต่างระหว่าง defaulter และ non-defaulter
    > บน **clean features** ที่ผ่าน leakage pipeline แล้ว

    | Section | คำถามที่ตอบ |
    |---------|-------------|
    | 1 | Violin Plots — distribution ต่างกันอย่างไร? |
    | 2 | Aggregation Type — `mean`, `last`, `last_minus_mean` อันไหน predictive กว่า? |
    | 3 | Null Pattern — missingness เป็น signal ได้ไหม? |
    | 4 | Correlation Structure — top features มี redundancy ไหม? |
    | 5 | Feature Family — feature เดียว ต่าง agg → ต่างกันแค่ไหน? |
    """)
    return AGG_COLORS, GROUP_COLORS, Path, cs, mo, mpatches, np, pl, plt, re


@app.cell
def _(Path, cs, pl):
    ROOT = Path(__file__).resolve().parent
    while not (ROOT / "pixi.toml").exists():
        ROOT = ROOT.parent
    DATA = ROOT / "data/processed"

    _train = pl.read_parquet(DATA / "train_features.parquet")
    _labels = pl.read_parquet(DATA / "train_labels.parquet")
    df = _train.join(_labels.select(["customer_ID", "target"]), on="customer_ID")

    numeric_cols = df.select(cs.numeric().exclude("target")).columns

    SAMPLE_N  = 40_000
    df_sample = df.sample(n=SAMPLE_N, seed=42)
    df0       = df_sample.filter(pl.col("target") == 0)
    df1       = df_sample.filter(pl.col("target") == 1)
    return DATA, df0, df1, df_sample, numeric_cols


@app.cell
def _(DATA, numeric_cols, pl):
    risk_df    = pl.read_parquet(DATA / "feature_risk_scores.parquet")
    _blocked   = set(risk_df.filter(pl.col("verdict") == "BLOCK")["feature"].to_list())
    clean_cols = [c for c in numeric_cols if c not in _blocked]

    # top features by cohen_d (CLEAN + WATCH)
    top_by_cohen = (
        risk_df
        .filter(pl.col("verdict") != "BLOCK")
        .sort("cohen_d", descending=True)
    )
    return clean_cols, risk_df, top_by_cohen


@app.cell
def _(GROUP_COLORS, df0, df1, mo, mpatches, np, plt, re, top_by_cohen):
    _top20 = top_by_cohen.head(20)["feature"].to_list()
    _cohen = {
        r["feature"]: r["cohen_d"]
        for r in top_by_cohen.head(20).iter_rows(named=True)
    }
    _verdict = {
        r["feature"]: r["verdict"]
        for r in top_by_cohen.head(20).iter_rows(named=True)
    }

    fig_violin, _axes = plt.subplots(4, 5, figsize=(20, 14))
    _axes = _axes.flatten()

    for _i, _feat in enumerate(_top20):
        _ax  = _axes[_i]
        _gm  = re.match(r"^([A-Z])_", _feat)
        _col = GROUP_COLORS.get(_gm.group(1) if _gm else "?", "#999")

        _v0 = df0[_feat].drop_nulls().to_numpy()
        _v1 = df1[_feat].drop_nulls().to_numpy()

        # clip to p2-p98 for visual clarity
        _all = np.concatenate([_v0, _v1])
        _lo, _hi = np.percentile(_all, 2), np.percentile(_all, 98)
        _v0c = _v0[(_v0 >= _lo) & (_v0 <= _hi)]
        _v1c = _v1[(_v1 >= _lo) & (_v1 <= _hi)]

        _parts = _ax.violinplot(
            [_v0c, _v1c], positions=[0, 1],
            showmedians=True, showextrema=False, widths=0.7,
        )
        # color bodies
        _parts["bodies"][0].set_facecolor("#4C72B0")
        _parts["bodies"][0].set_alpha(0.6)
        _parts["bodies"][1].set_facecolor("#C44E52")
        _parts["bodies"][1].set_alpha(0.6)
        _parts["cmedians"].set_color("white")
        _parts["cmedians"].set_linewidth(2)

        _d = _cohen[_feat]
        _sfx = " ⚠️" if _verdict[_feat] == "WATCH" else ""
        _ax.set_title(f"{_feat}{_sfx}\nd={_d:.2f}", fontsize=8, fontweight="bold", color=_col)
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(["No Default", "Default"], fontsize=7)
        _ax.tick_params(axis="y", labelsize=7)

    _legend_handles = [
        mpatches.Patch(color="#4C72B0", alpha=0.7, label="Non-Default"),
        mpatches.Patch(color="#C44E52", alpha=0.7, label="Default"),
    ]
    _axes[-1].legend(handles=_legend_handles, fontsize=10, loc="center")
    _axes[-1].axis("off")

    plt.suptitle(
        "Violin Plots — Top 20 Features by Cohen's d\n"
        "(sorted by class separation, ⚠️ = WATCH verdict)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    mo.md(f"""
    ## Section 1 — Violin Plots: Class Separation

    แสดง distribution ของ 20 features ที่มี **Cohen's d สูงสุด** (class separation แรงสุด)

    **อ่าน violin อย่างไร:**
    - ความกว้างของ violin = density ที่ค่านั้น (กว้าง = มีคนเยอะ)
    - เส้นสีขาวกลาง = **median** ของแต่ละ class
    - ถ้า violin ทั้งสองไม่ซ้อนกัน → feature นี้แยก class ได้ดีมาก

    **Cohen's d** วัด class separation: d > 3.0 = 🔴 แยกดีมาก, d > 1.5 = 🟠 แยกปานกลาง

    > ⚠️ = WATCH verdict จาก leakage pipeline — มี signal จริง แต่ควร monitor
    """)
    return (fig_violin,)


@app.cell
def _(fig_violin):
    fig_violin
    return


@app.cell
def _(AGG_COLORS, mo, np, pl, plt, re, risk_df):
    # เก็บ cohen_d และ single_feat_auc ต่อ agg type
    _agg_data: dict[str, list] = {}
    _agg_auc: dict[str, list] = {}

    for _row in risk_df.filter(pl.col("verdict") != "BLOCK").iter_rows(named=True):
        _m = re.match(r"^[A-Z]_\d+_(.+)$", _row["feature"])
        if _m:
            _a = _m.group(1)
            _agg_data.setdefault(_a, []).append(_row["cohen_d"])
            _agg_auc.setdefault(_a, []).append(_row["single_feat_auc"])

    # sort by median cohen_d
    _agg_order = sorted(_agg_data.keys(), key=lambda a: np.median(_agg_data[a]), reverse=True)
    _agg_labels = {
        "mean":             "mean\n(average over months)",
        "last":             "last\n(most recent month)",
        "max":              "max\n(highest ever)",
        "min":              "min\n(lowest ever)",
        "std":              "std\n(volatility)",
        "nunique":          "nunique\n(# distinct values)",
        "last_minus_first": "last−first\n(total change)",
        "last_minus_mean":  "last−mean\n(recent vs avg)",
    }

    fig_agg, _axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: box plot of cohen_d distribution per agg type
    _box_data   = [_agg_data[a] for a in _agg_order]
    _box_colors = [AGG_COLORS.get(a, "#999") for a in _agg_order]
    _bp = _axes[0].boxplot(
        _box_data,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
    )
    for _patch, _col in zip(_bp["boxes"], _box_colors):
        _patch.set_facecolor(_col)
        _patch.set_alpha(0.75)
    _axes[0].set_xticks(range(1, len(_agg_order) + 1))
    _axes[0].set_xticklabels(
        [_agg_labels.get(a, a) for a in _agg_order],
        fontsize=8, rotation=15, ha="right",
    )
    _axes[0].set_ylabel("Cohen's d (class separation)")
    _axes[0].set_title(
        "Cohen's d Distribution by Aggregation Type\n(median = white line, higher = stronger signal)",
        fontweight="bold", fontsize=11,
    )
    _axes[0].axhline(1.5, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Watchlist d=1.5")
    _axes[0].axhline(3.0, color="red",    linestyle="--", linewidth=1, alpha=0.7, label="Flagged d=3.0")
    _axes[0].legend(fontsize=9)

    # Right: avg single-feat AUC per agg type (bar)
    _auc_means = [np.mean(_agg_auc[a]) for a in _agg_order]
    _bars = _axes[1].bar(
        range(len(_agg_order)), _auc_means,
        color=_box_colors, alpha=0.82, edgecolor="white",
    )
    _axes[1].set_xticks(range(len(_agg_order)))
    _axes[1].set_xticklabels(
        [_agg_labels.get(a, a) for a in _agg_order],
        fontsize=8, rotation=15, ha="right",
    )
    _axes[1].set_ylabel("Average Single-Feature AUC")
    _axes[1].set_ylim(0.5, max(_auc_means) * 1.08)
    _axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Random")
    _axes[1].set_title(
        "Avg Single-Feature AUC by Aggregation Type\n(how well 1 feature alone can classify)",
        fontweight="bold", fontsize=11,
    )
    _axes[1].legend(fontsize=9)
    for _b, _v, _a in zip(_bars, _auc_means, _agg_order):
        _axes[1].text(
            _b.get_x() + _b.get_width() / 2, _b.get_height() + 0.001,
            f"{_v:.3f}\n(n={len(_agg_auc[_a])})",
            ha="center", va="bottom", fontsize=7.5,
        )

    plt.tight_layout()

    # build markdown insight
    _best = _agg_order[0]
    _worst = _agg_order[-1]
    _best_med  = np.median(_agg_data[_best])
    _worst_med = np.median(_agg_data[_worst])

    mo.md(f"""
    ## Section 2 — Aggregation Type Analysis

    ข้อมูล AmEx ถูก aggregate จาก statement รายเดือนด้วย **{len(_agg_order)} วิธี**
    คำถาม: วิธีไหนให้ signal ดีที่สุด?

    | Aggregation | ความหมาย | Median Cohen's d |
    |-------------|----------|----------------|
    {chr(10).join(f"    | `{a}` | {_agg_labels.get(a, a).replace(chr(10), ' ')} | {np.median(_agg_data[a]):.3f} |" for a in _agg_order)}

    **Key Insight:**
    - **`{_best}`** ดีที่สุด — median d = {_best_med:.3f}
    - **`{_worst}`** แย่ที่สุด — median d = {_worst_med:.3f}

    > **Counterintuitive:** "trend features" (`last_minus_mean`, `last_minus_first`) แย่กว่าค่า absolute
    > เพราะใน credit risk **ระดับ** ของ balance/payment สำคัญกว่า **การเปลี่ยนแปลง**
    > → สัญญาณเตือนคือมีหนี้เยอะ ไม่ใช่มีหนี้เพิ่มขึ้น
    """)
    return (fig_agg,)


@app.cell
def _(fig_agg):
    fig_agg
    return


@app.cell
def _(df0, df1, mo, np, pl, plt, risk_df):
    # features with meaningful null diff (CLEAN or WATCH only)
    _null_feats = (
        risk_df
        .filter((pl.col("verdict") != "BLOCK") & (pl.col("null_rate_diff") > 0.02))
        .sort("null_rate_diff", descending=True)
        .head(20)
    )

    _feats   = _null_feats["feature"].to_list()
    _diffs   = _null_feats["null_rate_diff"].to_list()
    _verdicts = _null_feats["verdict"].to_list()

    # compute actual null rates per class from sample
    _null0 = [float(df0[f].is_null().mean()) for f in _feats]
    _null1 = [float(df1[f].is_null().mean()) for f in _feats]

    # sort by diff for display
    _order  = sorted(range(len(_feats)), key=lambda i: _diffs[i], reverse=True)
    _flabels = [_feats[i] + (" ⚠️" if _verdicts[i] == "WATCH" else "") for i in _order]
    _n0      = [_null0[i] * 100 for i in _order]
    _n1      = [_null1[i] * 100 for i in _order]

    fig_null, _ax = plt.subplots(figsize=(10, 8))
    _y   = np.arange(len(_feats))
    _h   = 0.35
    _b0  = _ax.barh(_y + _h / 2, _n0, _h, color="#4C72B0", alpha=0.75, label="Non-Default (target=0)")
    _b1  = _ax.barh(_y - _h / 2, _n1, _h, color="#C44E52", alpha=0.75, label="Default (target=1)")

    # annotate difference
    for _i, (_r0, _r1) in enumerate(zip(_n0, _n1)):
        _diff_val = abs(_r1 - _r0)
        _ax.text(
            max(_r0, _r1) + 0.5, _y[_i],
            f"Δ{_diff_val:.1f}%",
            va="center", fontsize=8,
            color="#C44E52" if _r1 > _r0 else "#4C72B0",
            fontweight="bold",
        )

    _ax.set_yticks(_y)
    _ax.set_yticklabels(_flabels[::-1][::-1], fontsize=8)  # keep order
    _ax.set_yticklabels([_flabels[i] for i in range(len(_flabels))], fontsize=8)
    _ax.set_xlabel("Null Rate (%)")
    _ax.set_title(
        "Null Rate by Class — Missingness as Signal\n"
        "(features where null rate differs between default vs non-default)",
        fontweight="bold", fontsize=12,
    )
    _ax.legend(fontsize=10)
    _ax.axvline(0, color="gray", linewidth=0.5)
    plt.tight_layout()

    _n_meaningful = len(_feats)
    _max_diff_feat = _feats[0]
    _max_diff_val  = _diffs[0]

    mo.md(f"""
    ## Section 3 — Null Pattern: Missingness เป็น Signal ได้

    พบ **{_n_meaningful} features** ที่ null rate ต่างกันระหว่าง class (จาก CLEAN + WATCH เท่านั้น)

    **Feature ที่มีความต่างสูงสุด:** `{_max_diff_feat}` — ต่างกัน **{_max_diff_val:.1%}**

    **ทำไม missingness ถึงเป็น signal?**

    ใน credit data การที่ field ว่างเปล่า **ไม่ใช่ random** เสมอ:

    | สถานการณ์ | ความหมาย |
    |-----------|----------|
    | บัญชีถูกปิด | balance เป็น null เพราะไม่มี transaction |
    | ไม่มี payment ในเดือนนั้น | payment field เป็น null |
    | ข้อมูลหายเพราะ delinquency | บางธนาคารหยุด record เมื่อ default |

    > defaulter มี null rate **สูงกว่า** ใน features บางตัว
    > → การ `fill_null(0)` ก่อนเทรน model อาจซ่อน signal นี้ไว้
    > → ควรสร้าง binary flag `is_null` เป็น feature เพิ่มเติม (feature engineering)
    """)
    return (fig_null,)


@app.cell
def _(fig_null):
    fig_null
    return


@app.cell
def _(clean_cols, df_sample, mo, mpatches, np, plt, re, top_by_cohen):
    _top30_feats = [
        f for f in top_by_cohen["feature"].to_list()
        if f in clean_cols
    ][:30]

    _groups30 = [
        re.match(r"^([A-Z])_", f).group(1) if re.match(r"^([A-Z])_", f) else "?"
        for f in _top30_feats
    ]

    _X        = df_sample.select(_top30_feats).fill_null(0).to_numpy()
    _corr_mat = np.corrcoef(_X.T)

    # compact readable labels: e.g. "P2_last", "D48_mean"
    _short_labels = []
    for _f in _top30_feats:
        _m = re.match(r"^([A-Z])_(\d+)_(.+)$", _f)
        if _m:
            _agg = _m.group(3)
            _agg_short = {"last_minus_mean": "l-mn", "last_minus_first": "l-1st"}.get(_agg, _agg[:5])
            _short_labels.append(f"{_m.group(1)}{_m.group(2)}_{_agg_short}")
        else:
            _short_labels.append(_f[:10])

    _GC = {"P": "#4C72B0", "D": "#DD8452", "B": "#55A868", "R": "#C44E52", "S": "#8172B2"}
    _row_colors = [_GC.get(g, "#999") for g in _groups30]

    fig_corr, _ax = plt.subplots(figsize=(17, 15))
    _im = _ax.imshow(_corr_mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    _ax.set_xticks(range(len(_short_labels)))
    _ax.set_yticks(range(len(_short_labels)))
    # x-labels on top, rotated
    _ax.xaxis.set_label_position("top")
    _ax.xaxis.tick_top()
    _ax.set_xticklabels(_short_labels, rotation=60, ha="left", fontsize=8.5)
    _ax.set_yticklabels(_short_labels, fontsize=8.5)

    # color both axes labels by group — no inset_axes needed
    for _lbl, _col in zip(_ax.get_yticklabels(), _row_colors):
        _lbl.set_color(_col)
        _lbl.set_fontweight("bold")
    for _lbl, _col in zip(_ax.get_xticklabels(), _row_colors):
        _lbl.set_color(_col)
        _lbl.set_fontweight("bold")

    # grid lines between cells for readability
    for _k in range(len(_short_labels) + 1):
        _ax.axhline(_k - 0.5, color="white", linewidth=0.4)
        _ax.axvline(_k - 0.5, color="white", linewidth=0.4)

    plt.colorbar(_im, ax=_ax, fraction=0.025, pad=0.02, label="Pearson Correlation")

    _patches = [
        mpatches.Patch(facecolor=_c, label=f"{_g}_ ({['Payment','Delinquency','Balance','Risk','Spend'][list(_GC).index(_g)]})")
        for _g, _c in _GC.items() if _g in _groups30
    ]
    _ax.legend(handles=_patches, loc="lower right", fontsize=9, framealpha=0.95,
               bbox_to_anchor=(1.0, -0.01))
    _ax.set_title(
        "Correlation Matrix — Top 30 Clean Features\n(label color = feature group)",
        fontweight="bold", fontsize=13, pad=14,
    )
    plt.subplots_adjust(left=0.16, right=0.93, top=0.88, bottom=0.04)

    _high_corr_pairs = []
    for _i in range(len(_top30_feats)):
        for _j in range(_i + 1, len(_top30_feats)):
            _r = abs(_corr_mat[_i, _j])
            if _r > 0.85:
                _high_corr_pairs.append((_top30_feats[_i], _top30_feats[_j], _r))
    _high_corr_pairs.sort(key=lambda x: -x[2])

    _pair_lines = "\n".join(
        f"  - `{a}` ↔ `{b}` — r={r:.3f}"
        for a, b, r in _high_corr_pairs[:5]
    ) or "  - ไม่มี pairs ที่ |r| > 0.85"

    mo.md(f"""
    ## Section 4 — Correlation Structure

    Correlation matrix ของ top 30 clean features (sorted by Cohen's d)
    **สีของ label** = กลุ่มของ feature (P/D/B/R/S)

    **อ่าน heatmap อย่างไร:**
    - 🔴 แดง = corr สูง → feature คู่นี้ redundant (model ไม่ได้ประโยชน์จากทั้งคู่)
    - 🔵 น้ำเงิน = corr ต่ำหรือลบ → feature complement กัน (ให้ข้อมูลคนละมิติ)
    - ⬜ ขาว = independent signal

    **Feature pairs ที่ |r| > 0.85 (highly redundant):**
    {_pair_lines}

    > **Insight:** feature ที่ corr กันสูงมาก model ไม่ได้ประโยชน์จากการมีทั้งคู่
    > → ใน feature engineering รอบต่อไป อาจ drop duplicates หรือ PCA กลุ่มนั้น
    """)
    return (fig_corr,)


@app.cell
def _(fig_corr):
    fig_corr
    return


@app.cell
def _(AGG_COLORS, clean_cols, df0, df1, mo, np, pl, plt, re, risk_df):
    # หา base feature ที่มี variants หลากหลายที่สุดในบรรดา top features
    _base_counts: dict[str, list] = {}
    for _f in clean_cols:
        _m = re.match(r"^([A-Z]_\d+)_(.+)$", _f)
        if _m:
            _base = _m.group(1)
            _base_counts.setdefault(_base, []).append(_f)

    # top base feature = most variants + highest avg cohen_d
    _risk_lookup = {r["feature"]: r["cohen_d"] for r in risk_df.iter_rows(named=True)}
    _best_base = max(
        _base_counts,
        key=lambda b: (
            len(_base_counts[b]),
            sum(_risk_lookup.get(f, 0) for f in _base_counts[b]) / len(_base_counts[b]),
        ),
    )
    _family = sorted(
        _base_counts[_best_base],
        key=lambda f: _risk_lookup.get(f, 0),
        reverse=True,
    )
    _family_cohens = [_risk_lookup.get(f, 0) for f in _family]
    _family_verdicts = {
        r["feature"]: r["verdict"]
        for r in risk_df.filter(pl.col("feature").is_in(_family)).iter_rows(named=True)
    }
    _family_aucs = {
        r["feature"]: r["single_feat_auc"]
        for r in risk_df.filter(pl.col("feature").is_in(_family)).iter_rows(named=True)
    }

    _n_cols   = min(4, len(_family))
    _n_rows_v = (len(_family) + _n_cols - 1) // _n_cols
    fig_family, _axes = plt.subplots(_n_rows_v, _n_cols, figsize=(5 * _n_cols, 4 * _n_rows_v))
    _axes = np.array(_axes).flatten()

    for _i, _feat in enumerate(_family):
        _ax = _axes[_i]
        _m2 = re.match(r"^[A-Z]_\d+_(.+)$", _feat)
        _agg_name = _m2.group(1) if _m2 else _feat
        _agg_col  = AGG_COLORS.get(_agg_name, "#999")

        _v0 = df0[_feat].drop_nulls().to_numpy()
        _v1 = df1[_feat].drop_nulls().to_numpy()

        _all = np.concatenate([_v0, _v1])
        if len(_all) < 2:
            _ax.set_visible(False)
            continue
        _lo, _hi = np.percentile(_all, 2), np.percentile(_all, 98)
        _v0c = _v0[(_v0 >= _lo) & (_v0 <= _hi)]
        _v1c = _v1[(_v1 >= _lo) & (_v1 <= _hi)]

        _parts = _ax.violinplot(
            [_v0c, _v1c], positions=[0, 1],
            showmedians=True, showextrema=False, widths=0.65,
        )
        _parts["bodies"][0].set_facecolor("#4C72B0")
        _parts["bodies"][0].set_alpha(0.6)
        _parts["bodies"][1].set_facecolor("#C44E52")
        _parts["bodies"][1].set_alpha(0.6)
        _parts["cmedians"].set_color("white")
        _parts["cmedians"].set_linewidth(2)

        _d   = _family_cohens[_i]
        _auc = _family_aucs.get(_feat, 0.5)
        _v   = _family_verdicts.get(_feat, "?")
        _sfx = " ⚠️" if _v == "WATCH" else ""
        _ax.set_title(
            f"`{_agg_name}`{_sfx}\nd={_d:.3f}  AUC={_auc:.3f}",
            fontsize=9, fontweight="bold", color=_agg_col,
        )
        _ax.set_xticks([0, 1])
        _ax.set_xticklabels(["No Default", "Default"], fontsize=8)
        _ax.tick_params(axis="y", labelsize=7)

    # hide extra axes
    for _j in range(len(_family), len(_axes)):
        _axes[_j].set_visible(False)

    plt.suptitle(
        f"Feature Family: `{_best_base}_*` — Same Base, Different Aggregations\n"
        f"(sorted by Cohen's d, {len(_family)} variants)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    # insight: rank agg types for this family
    _agg_rank_lines = "\n".join(
        f"    | `{re.match(r'^[A-Z]_d+_(.+)$', f) and re.match(r'^[A-Z]_\\d+_(.+)$', f).group(1) or f}` "
        f"| {_risk_lookup.get(f, 0):.3f} | {_family_aucs.get(f, 0.5):.3f} "
        f"| {'✅' if _family_verdicts.get(f) == 'CLEAN' else '⚠️'} |"
        for f in _family
    )

    mo.md(f"""
    ## Section 5 — Feature Family: `{_best_base}_*`

    Base feature เดียวกัน แต่ aggregate ต่างวิธี → signal ต่างกันแค่ไหน?

    | Aggregation | Cohen's d | Single-feat AUC | Verdict |
    |-------------|-----------|----------------|---------|
    {_agg_rank_lines}

    **Key Takeaway:**
    - Aggregation ที่ดีที่สุดสำหรับ `{_best_base}` คือ **`{re.match(r'^[A-Z]_\\d+_(.+)$', _family[0]).group(1) if re.match(r'^[A-Z]_\\d+_(.+)$', _family[0]) else _family[0]}`**
    - `last_minus_mean` และ `last_minus_first` มักให้ signal น้อยกว่าค่า raw (สอดคล้องกับ Section 2)
    - ใน model ที่ใช้ Top-30 ต่อ fold → มีโอกาสสูงที่จะเลือก aggregation ที่ดีที่สุดของ family นี้เข้าไป

    > **Feature Engineering Suggestion:**
    > ถ้าจะสร้าง feature ใหม่จาก `{_best_base}` → ควร focus ที่ `mean`, `last`, `max`
    > มากกว่า `last_minus_mean` ซึ่ง signal อ่อนกว่า
    """)
    return (fig_family,)


@app.cell
def _(fig_family):
    fig_family
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## สรุป — Class Comparison Insights

    | Section | Insight หลัก | นำไปใช้อะไร? |
    |---------|-------------|-------------|
    | **1 Violin** | top features มีการแยก class ชัดมาก โดยเฉพาะกลุ่ม P_ และ D_ | เห็นภาพ distribution ชัดกว่า histogram |
    | **2 Agg Type** | `mean` และ `last` ดีกว่า `last_minus_mean` อย่างมีนัยสำคัญ | ลด search space ใน feature engineering |
    | **3 Null Pattern** | missingness ไม่ random — defaulter มี null rate ต่างกัน | สร้าง `is_null` flag เป็น feature เพิ่มเติม |
    | **4 Correlation** | top features บางคู่ corr กันสูง → redundant | ใช้ drop redundant หรือ PCA |
    | **5 Family** | aggregation ที่ดีสุดต่อ feature ไม่เหมือนกัน | focus feature engineering ที่ agg ที่ win |

    ### Next Steps ที่แนะนำ

    1. **สร้าง `is_null` binary flags** สำหรับ features ใน Section 3 → อาจเพิ่ม AUC
    2. **Drop redundant pairs** (|r| > 0.85) ออก 1 ตัว → ลด noise ใน model
    3. **Focus feature engineering** ที่ `mean` และ `last` aggregation เป็นหลัก
    4. **Cross-group interactions**: เช่น P_2_last × D_48_last → payment behavior + delinquency
    """)
    return


if __name__ == "__main__":
    app.run()
