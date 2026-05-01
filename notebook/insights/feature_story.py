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
    import pandas as pd
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

    mo.md("""
    # Feature Story — อะไรขับเคลื่อน Credit Default?

    > Notebook นี้ตอบคำถามที่ Data Scientist ต้องการรู้:
    > **feature ไหนสำคัญ ทำไม และมันบอกอะไรเกี่ยวกับ defaulter?**

    เส้นทาง pipeline ที่ผ่านมา:
    ```
    1,261 features raw
        ↓ Gate 1 : ตัด corr > 0.9  (target proxy / red flag)
        ↓ Gate 1.5: ตัด BLOCK จาก 9-technique leakage pipeline
        ↓ Gate 2 : เลือก Top-30 / fold จาก inner train เท่านั้น
    → features ที่รอด = signal จริง ไม่มี leakage
    ```

    | บท | หัวข้อ | คำถามที่ตอบ |
    |----|--------|-------------|
    | 1 | Context | ทำไม credit default prediction ถึงยาก? |
    | 2 | Feature Landscape | features ที่รอดมีอะไรบ้าง? |
    | 3 | Model's Favourites | model เลือก feature ไหนซ้ำๆ ทุก fold? |
    | 4 | Defaulter Portrait | defaulter ต่างจาก non-defaulter อย่างไร? |
    | 5 | Group Contribution | กลุ่มไหนขับเคลื่อน prediction? |
    | 6 | Key Insights | 5 สิ่งที่ต้องบอก stakeholder |
    """)
    return (
        GROUP_COLORS,
        GROUP_NAMES,
        Path,
        cs,
        mo,
        mpatches,
        np,
        pd,
        pl,
        plt,
        re,
    )


@app.cell
def _(Path, cs, pd, pl, re):
    ROOT = Path(__file__).resolve().parent
    while not (ROOT / "pixi.toml").exists():
        ROOT = ROOT.parent
    DATA = ROOT / "data/processed"

    _train_feat = pl.read_parquet(DATA / "train_features.parquet")
    _labels     = pl.read_parquet(DATA / "train_labels.parquet")
    df          = _train_feat.join(_labels.select(["customer_ID", "target"]), on="customer_ID")

    SAMPLE_N  = 30_000
    df_sample = df.sample(n=SAMPLE_N, seed=42)

    numeric_cols = df.select(cs.numeric().exclude("target")).columns

    _col_meta = []
    for _c in numeric_cols:
        _m = re.match(r"^([A-Z])_(\d+)_(\w+)$", _c)
        if _m:
            _col_meta.append({"column": _c, "group": _m.group(1), "agg": _m.group(3)})
    meta_df = pd.DataFrame(_col_meta)
    return DATA, df, df_sample, numeric_cols


@app.cell
def _(DATA, df, mo, numeric_cols, pl):
    _risk_raw = pl.read_parquet(DATA / "feature_risk_scores.parquet")
    risk_df   = _risk_raw

    _blocked   = set(risk_df.filter(pl.col("verdict") == "BLOCK")["feature"].to_list())
    _watch_set = set(risk_df.filter(pl.col("verdict") == "WATCH")["feature"].to_list())
    _clean_set = set(risk_df.filter(pl.col("verdict") == "CLEAN")["feature"].to_list())

    # features ที่ model อนุญาตให้ใช้ = CLEAN + WATCH (ไม่มี BLOCK)
    clean_cols = [c for c in numeric_cols if c in _clean_set or c in _watch_set]

    n_all       = len(numeric_cols)
    n_clean     = len(_clean_set)
    n_watch     = len(_watch_set)
    n_block     = len(_blocked)
    n_model     = len(clean_cols)
    target_rate = float(df["target"].mean())
    n_rows      = df.shape[0]
    n_defaulter = int(df["target"].sum())
    n_normal    = n_rows - n_defaulter

    mo.md(f"""
    ## บทที่ 1 — Context: ทำไม Credit Default ถึงยาก?

    ### American Express Default Prediction

    AmEx ต้องทำนายล่วงหน้าว่า customer จะ default บัตรเครดิตใน 18 เดือนข้างหน้า
    โดยใช้ข้อมูล statement รายเดือนที่ถูก aggregate เป็น feature หลายร้อยตัว

    | ข้อมูล | ค่า |
    |--------|-----|
    | จำนวน customers | **{n_rows:,}** |
    | Defaulters (target=1) | **{n_defaulter:,}** ({target_rate:.1%}) |
    | Non-defaulters (target=0) | **{n_normal:,}** ({1-target_rate:.1%}) |
    | Features ก่อน pipeline | **{n_all:,}** |
    | Features หลัง Gate 1+1.5 | **{n_model:,}** (CLEAN + WATCH) |
    | Features ถูก BLOCK | **{n_block}** (leakage risk สูง) |

    ### ความท้าทาย 4 ข้อ

    - **Class imbalance {target_rate:.0%}:{1-target_rate:.0%}** → ต้องใช้ AUC แทน Accuracy
    - **Temporal prediction 18 เดือน** → feature ต้องไม่ leak ข้อมูลอนาคต
    - **Feature explosion {n_all:,} features** → 5 หมวด × หลายร้อย indicators ต้องกรองให้เหลือ signal จริง
    - **Leakage risk** → บาง feature สะท้อน target โดยตรง ตรวจด้วย 9 techniques แล้ว BLOCK ออก {n_block} ตัว
    """)
    return (
        clean_cols,
        n_block,
        n_clean,
        n_model,
        n_rows,
        n_watch,
        risk_df,
        target_rate,
    )


@app.cell
def _(
    GROUP_COLORS,
    GROUP_NAMES,
    clean_cols,
    mo,
    mpatches,
    n_block,
    n_clean,
    n_watch,
    np,
    plt,
    re,
    risk_df,
):
    _groups   = list(GROUP_NAMES.keys())
    _verdicts = ["CLEAN", "WATCH", "BLOCK"]
    _vcolors  = {"CLEAN": "#55A868", "WATCH": "#FFA500", "BLOCK": "#C44E52"}

    # count per group × verdict
    _bd = {g: {v: 0 for v in _verdicts} for g in _groups}
    for _row in risk_df.iter_rows(named=True):
        _gm = re.match(r"^([A-Z])_", _row["feature"])
        if _gm and _gm.group(1) in _bd:
            _bd[_gm.group(1)][_row["verdict"]] += 1

    # clean features per group
    _clean_by_g = {}
    for _c in clean_cols:
        _gm2 = re.match(r"^([A-Z])_", _c)
        if _gm2 and _gm2.group(1) in GROUP_NAMES:
            _g2 = _gm2.group(1)
            _clean_by_g[_g2] = _clean_by_g.get(_g2, 0) + 1

    fig_landscape, _axes = plt.subplots(1, 2, figsize=(16, 5))

    # Left: stacked bar — verdict breakdown per group
    _x    = np.arange(len(_groups))
    _bots = np.zeros(len(_groups))
    for _v in _verdicts:
        _vals = [_bd[g][_v] for g in _groups]
        _axes[0].bar(
            _x, _vals, bottom=_bots, label=_v,
            color=_vcolors[_v], alpha=0.85, edgecolor="white", linewidth=0.5,
        )
        _bots += np.array(_vals, dtype=float)
    _axes[0].set_xticks(_x)
    _axes[0].set_xticklabels([f"{GROUP_NAMES[g]}\n({g}_)" for g in _groups])
    _axes[0].set_ylabel("# Features")
    _axes[0].set_title("Feature Landscape — Verdict per Group", fontweight="bold", fontsize=12)
    _axes[0].legend(title="Verdict", loc="upper right")
    for _i, _g in enumerate(_groups):
        _tot = sum(_bd[_g].values())
        _axes[0].text(_i, _tot + 4, str(_tot), ha="center", fontsize=9, color="#555")

    # Right: clean features per group
    _cleans = [_clean_by_g.get(g, 0) for g in _groups]
    _bars   = _axes[1].bar(
        [GROUP_NAMES[g] for g in _groups], _cleans,
        color=[GROUP_COLORS[g] for g in _groups], alpha=0.85,
        edgecolor="white", linewidth=0.5,
    )
    _axes[1].set_ylabel("# Features (CLEAN + WATCH)")
    _axes[1].set_title(
        "Clean Features Available to Model\n(Gate 1 + Gate 1.5 passed)",
        fontweight="bold", fontsize=12,
    )
    for _b, _v in zip(_bars, _cleans):
        _axes[1].text(
            _b.get_x() + _b.get_width() / 2, _b.get_height() + 2,
            str(_v), ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    _legend_patches = [
        mpatches.Patch(color=GROUP_COLORS[g], label=f"{GROUP_NAMES[g]} ({g}_)")
        for g in _groups
    ]
    _axes[1].legend(handles=_legend_patches, fontsize=9)
    plt.tight_layout()

    _d_block_rate = f"{_bd['D']['BLOCK'] / max(sum(_bd['D'].values()), 1):.0%}"

    mo.md(f"""
    ## บทที่ 2 — Feature Landscape

    | Verdict | จำนวน | ความหมาย |
    |---------|-------|----------|
    | ✅ CLEAN | **{n_clean}** | ผ่านทุก 9 technique → ใช้ได้เต็มที่ |
    | ⚠️ WATCH | **{n_watch}** | มี signal เล็กน้อย → ใช้ได้ แต่ต้อง monitor |
    | 🚫 BLOCK | **{n_block}** | เสี่ยง leakage → ห้ามเข้า model |

    **ข้อสังเกต:**
    - กลุ่ม **Delinquency (D_)** มี feature มากสุด แต่ก็มี BLOCK rate สูงด้วย ({_d_block_rate}) → ข้อมูลค้างชำระ sensitive ต่อ leakage มากที่สุด
    - กลุ่ม **Payment (P_)** มี feature น้อยที่สุด แต่ clean rate สูง → ข้อมูลการชำระที่น่าเชื่อถือ
    - กลุ่ม **Balance (B_)** และ **Risk (R_)** มี BLOCK กระจาย → ต้องระวังเรื่อง target leakage
    """)
    return (fig_landscape,)


@app.cell
def _(fig_landscape):
    fig_landscape
    return


@app.cell
def _(
    GROUP_COLORS,
    GROUP_NAMES,
    clean_cols,
    df,
    mo,
    mpatches,
    pl,
    plt,
    risk_df,
):
    TOP_N = 30

    # Gate 2 logic: |corr| กับ target บน full df ใช้ polars (fast)
    _corr_df = (
        df.select([pl.corr(c, "target").alias(c) for c in clean_cols])
        .unpivot(variable_name="feature", value_name="corr")
        .with_columns(pl.col("corr").abs().alias("abs_corr"))
        .sort("abs_corr", descending=True)
        .head(TOP_N)
    )

    top_feats_df = (
        _corr_df
        .join(
            risk_df.select(["feature", "verdict", "cohen_d", "single_feat_auc", "risk_score"]),
            on="feature",
            how="left",
        )
        .with_columns(
            pl.col("feature").str.extract(r"^([A-Z])_", 1).alias("group")
        )
    )
    top_feats = top_feats_df["feature"].to_list()

    # horizontal bar chart — colored by group, WATCH marked
    _labels   = top_feats_df["feature"].to_list()[::-1]
    _vals     = top_feats_df["abs_corr"].to_list()[::-1]
    _grps     = top_feats_df["group"].to_list()[::-1]
    _verdicts = top_feats_df["verdict"].to_list()[::-1]
    _sfaucs   = top_feats_df["single_feat_auc"].to_list()[::-1]
    _colors   = [GROUP_COLORS.get(g, "#999") for g in _grps]

    fig_top, _ax = plt.subplots(figsize=(11, 11))
    _bars = _ax.barh(
        _labels, _vals,
        color=_colors, alpha=0.82, edgecolor="white", linewidth=0.4,
    )
    for _b, _v, _verd, _sf in zip(_bars, _vals, _verdicts, _sfaucs):
        _suffix = " ⚠️" if _verd == "WATCH" else ""
        _ax.text(
            _b.get_width() + 0.002,
            _b.get_y() + _b.get_height() / 2,
            f"|r|={_v:.3f}  AUC={_sf:.3f}{_suffix}",
            va="center", fontsize=8, color="#333",
        )

    _ax.set_xlabel("|Correlation with Target|  (Gate 2 selection criterion)")
    _ax.set_title(
        f"Top {TOP_N} Features — Model's Favourites\n(⚠️ = WATCH verdict, still allowed in model)",
        fontweight="bold", fontsize=12,
    )
    _ax.set_xlim(0, (_vals[0] if _vals else 0.5) * 1.3)

    _patches = [
        mpatches.Patch(color=GROUP_COLORS[g], label=f"{GROUP_NAMES.get(g, g)} ({g}_)")
        for g in GROUP_NAMES
    ]
    _ax.legend(handles=_patches, fontsize=9, loc="lower right")
    plt.tight_layout()

    # group summary for markdown
    _g_counts = top_feats_df.group_by("group").len().sort("len", descending=True)
    _g_lines  = "\n".join(
        f"  - **{GROUP_NAMES.get(r['group'], r['group'])} ({r['group']}_)**: {r['len']} features"
        for r in _g_counts.iter_rows(named=True)
    )
    _n_watch_top = top_feats_df.filter(pl.col("verdict") == "WATCH").shape[0]

    mo.md(f"""
    ## บทที่ 3 — Model's Favourites

    Top {TOP_N} features ที่ model เลือกก่อน วัดด้วย **|correlation with target|**
    → นี่คือ Gate 2 logic เดียวกับที่ใช้ใน 5-fold CV ของ baseline model

    แต่ละ feature แสดง:
    - **|r|** = correlation กับ target (ค่าสูง = linear signal แรง)
    - **AUC** = single-feature AUC จาก leakage pipeline (วัดทั้ง linear + non-linear)
    - **⚠️** = WATCH verdict (pipeline พบ signal เล็กน้อย แต่ยังอนุญาตให้ใช้)

    **การกระจายตาม group:**
    {_g_lines}

    **{_n_watch_top} จาก {TOP_N}** features ใน top เป็น WATCH → ควร manual review ก่อน production
    """)
    return TOP_N, fig_top, top_feats, top_feats_df


@app.cell
def _(fig_top):
    fig_top
    return


@app.cell
def _(GROUP_COLORS, df_sample, mo, mpatches, np, pl, plt, re, top_feats):
    _show = top_feats[:9]
    _df0  = df_sample.filter(pl.col("target") == 0)
    _df1  = df_sample.filter(pl.col("target") == 1)

    fig_dist, _axes = plt.subplots(3, 3, figsize=(15, 12))
    _axes = _axes.flatten()

    for _i, _feat in enumerate(_show):
        _ax  = _axes[_i]
        _gm  = re.match(r"^([A-Z])_", _feat)
        _col = GROUP_COLORS.get(_gm.group(1) if _gm else "?", "#999")

        _v0 = _df0[_feat].drop_nulls().to_numpy()
        _v1 = _df1[_feat].drop_nulls().to_numpy()

        # clip to p1–p99 for display clarity
        _all = np.concatenate([_v0, _v1])
        _lo, _hi = np.percentile(_all, 1), np.percentile(_all, 99)
        _v0c = _v0[(_v0 >= _lo) & (_v0 <= _hi)]
        _v1c = _v1[(_v1 >= _lo) & (_v1 <= _hi)]

        _ax.hist(_v0c, bins=50, alpha=0.55, color="#4C72B0", density=True,
                 label=f"No Default")
        _ax.hist(_v1c, bins=50, alpha=0.55, color="#C44E52", density=True,
                 label=f"Default")

        _ax.set_title(_feat, fontsize=9, fontweight="bold", color=_col)
        _ax.set_xlabel("Value (p1–p99 clipped)", fontsize=7)
        _ax.set_ylabel("Density", fontsize=7)
        _ax.tick_params(labelsize=7)

        # annotate direction of effect
        _m0 = float(np.mean(_v0c)) if len(_v0c) > 0 else 0.0
        _m1 = float(np.mean(_v1c)) if len(_v1c) > 0 else 0.0
        _arrow = "↑ in default" if _m1 > _m0 else "↓ in default"
        _ax.text(0.97, 0.97, _arrow, transform=_ax.transAxes,
                 ha="right", va="top", fontsize=8, color="#C44E52", fontweight="bold")

    # legend panel
    _axes[-1].legend(
        handles=[
            mpatches.Patch(color="#4C72B0", alpha=0.7, label="Non-Default (target=0)"),
            mpatches.Patch(color="#C44E52", alpha=0.7, label="Default (target=1)"),
        ],
        fontsize=11, loc="center",
    )
    _axes[-1].set_title("Legend", fontsize=10, fontweight="bold")
    _axes[-1].axis("off")

    plt.suptitle(
        "Defaulter Portrait — Top 9 Feature Distributions",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    mo.md(f"""
    ## บทที่ 4 — Defaulter Portrait

    Distribution ของ 9 features สำคัญอันดับแรก แยกตาม class

    - **สีน้ำเงิน** = Non-Default (target=0)
    - **สีแดง** = Default (target=1)

    **อ่าน chart อย่างไร:**
    | Pattern | ความหมาย |
    |---------|----------|
    | 2 bars แยกกันชัด | feature นี้เป็น **strong signal** |
    | 2 bars ซ้อนกันเยอะ | feature นี้แยก class ได้น้อย |
    | ↑ in default | defaulter มีค่า **สูงกว่า** → เพิ่มความเสี่ยง |
    | ↓ in default | defaulter มีค่า **ต่ำกว่า** → เพิ่มความเสี่ยง |

    > **Note:** ค่าถูก clip ที่ p1–p99 เพื่อความชัดเจน
    > (outlier สุดขอบถูกตัดออกจาก plot แต่ไม่ได้ถูกลบออกจาก model)
    """)
    return (fig_dist,)


@app.cell
def _(fig_dist):
    fig_dist
    return


@app.cell
def _(
    GROUP_COLORS,
    GROUP_NAMES,
    TOP_N,
    clean_cols,
    mo,
    np,
    plt,
    re,
    top_feats,
):
    _groups = list(GROUP_NAMES.keys())

    # count top features per group
    _top_by_g = {}
    for _f in top_feats:
        _gm = re.match(r"^([A-Z])_", _f)
        if _gm and _gm.group(1) in GROUP_NAMES:
            _g = _gm.group(1)
            _top_by_g[_g] = _top_by_g.get(_g, 0) + 1

    # count clean features per group
    _clean_by_g2 = {}
    for _f2 in clean_cols:
        _gm2 = re.match(r"^([A-Z])_", _f2)
        if _gm2 and _gm2.group(1) in GROUP_NAMES:
            _g2 = _gm2.group(1)
            _clean_by_g2[_g2] = _clean_by_g2.get(_g2, 0) + 1

    _top_counts   = np.array([_top_by_g.get(g, 0) for g in _groups], dtype=float)
    _clean_counts = np.array([_clean_by_g2.get(g, 0) for g in _groups], dtype=float)
    _top_pct      = _top_counts / _top_counts.sum() * 100
    _clean_pct    = _clean_counts / _clean_counts.sum() * 100

    fig_groups, _axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: grouped bar — share in pool vs share in top 30
    _x = np.arange(len(_groups))
    _w = 0.35
    _axes[0].bar(
        _x - _w / 2, _clean_pct, _w,
        label="All Clean Features (pool)", color="#aec6cf", alpha=0.85, edgecolor="white",
    )
    _axes[0].bar(
        _x + _w / 2, _top_pct, _w,
        label=f"Top {TOP_N} (selected by model)",
        color=[GROUP_COLORS[g] for g in _groups], alpha=0.85, edgecolor="white",
    )
    _axes[0].set_xticks(_x)
    _axes[0].set_xticklabels([f"{GROUP_NAMES[g]}\n({g}_)" for g in _groups])
    _axes[0].set_ylabel("% of Features")
    _axes[0].set_title(
        f"Group Share: Pool vs Top {TOP_N}\n(over-represented = model prefers this group)",
        fontweight="bold", fontsize=12,
    )
    _axes[0].legend()

    for _i, (_cp, _tp) in enumerate(zip(_clean_pct, _top_pct)):
        if _cp > 0:
            _ratio = _tp / _cp
            if _ratio > 1.15:
                _axes[0].annotate(
                    f"↑{_ratio:.1f}x", (_x[_i] + _w / 2, _tp + 0.5),
                    ha="center", fontsize=9, color="#C44E52", fontweight="bold",
                )
            elif _ratio < 0.85:
                _axes[0].annotate(
                    f"↓{_ratio:.1f}x", (_x[_i] + _w / 2, _tp + 0.5),
                    ha="center", fontsize=9, color="#4C72B0",
                )

    # Right: donut chart — top 30 composition
    _sizes        = [_top_by_g.get(g, 0) for g in _groups]
    _donut_labels = [f"{GROUP_NAMES[g]}\n({_top_by_g.get(g, 0)})" for g in _groups]
    _wedges, _texts, _autotexts = _axes[1].pie(
        _sizes,
        labels=_donut_labels,
        colors=[GROUP_COLORS[g] for g in _groups],
        autopct=lambda p: f"{p:.0f}%" if p > 4 else "",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
    )
    for _at in _autotexts:
        _at.set_fontsize(9)
        _at.set_fontweight("bold")
        _at.set_color("white")
    _centre = plt.Circle((0, 0), 0.55, fc="white")
    _axes[1].add_patch(_centre)
    _axes[1].text(
        0, 0, f"Top\n{TOP_N}",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#333",
    )
    _axes[1].set_title(
        f"Top {TOP_N} Feature Composition by Group",
        fontweight="bold", fontsize=12,
    )
    plt.tight_layout()

    # over/under-represented analysis for markdown
    _over  = [(GROUP_NAMES[g], _top_pct[i], _clean_pct[i], _top_pct[i] / max(_clean_pct[i], 0.1))
              for i, g in enumerate(_groups) if _clean_pct[i] > 0 and _top_pct[i] / _clean_pct[i] > 1.15]
    _under = [(GROUP_NAMES[g], _top_pct[i], _clean_pct[i], _top_pct[i] / max(_clean_pct[i], 0.1))
              for i, g in enumerate(_groups) if _clean_pct[i] > 0 and _top_pct[i] / _clean_pct[i] < 0.85]
    _over.sort(key=lambda x: -x[3])

    _over_md = "\n".join(
        f"  - **{n}**: {pt:.0f}% ของ top {TOP_N} แต่มีแค่ {ct:.0f}% ของ pool → model เลือก **{r:.1f}x** เกินสัดส่วน"
        for n, pt, ct, r in _over
    ) or "  - (none)"
    _under_md = "\n".join(
        f"  - **{n}**: {pt:.0f}% ของ top {TOP_N} น้อยกว่าสัดส่วน ({ct:.0f}%) → model ให้ weight น้อย"
        for n, pt, ct, _ in _under
    ) or "  - (none)"

    mo.md(f"""
    ## บทที่ 5 — Group Contribution

    เปรียบเทียบสัดส่วน group ระหว่าง:
    - **Pool (All Clean)** = สิ่งที่ model **สามารถ** เลือกได้
    - **Top {TOP_N}** = สิ่งที่ model **เลือกจริง**

    ความต่างระหว่างสองนี้ = **preference ของ model** → บอกว่า dimension ไหนสำคัญจริง

    **Over-represented (model เลือกมากกว่าสัดส่วน):**
    {_over_md}

    **Under-represented (model เลือกน้อยกว่าสัดส่วน):**
    {_under_md}

    > **Business implication:** group ที่ over-represented คือ **มิติที่ขับเคลื่อน credit default จริงๆ**
    > → data quality และ completeness ของ group นั้นส่งผลต่อ model โดยตรง
    > → ควร prioritize เมื่อ collect ข้อมูล customer ใหม่
    """)
    return (fig_groups,)


@app.cell
def _(fig_groups):
    fig_groups
    return


@app.cell
def _(
    GROUP_NAMES,
    TOP_N,
    mo,
    n_block,
    n_clean,
    n_model,
    n_rows,
    n_watch,
    pl,
    target_rate,
    top_feats_df,
):
    _top3         = top_feats_df.head(3)["feature"].to_list()
    _top3_groups  = [GROUP_NAMES.get(f[0], "?") for f in _top3]
    _top3_aucs    = top_feats_df.head(3)["single_feat_auc"].to_list()

    _g_counts     = top_feats_df.group_by("group").len().sort("len", descending=True)
    _dom_key      = _g_counts[0]["group"][0]
    _dom_name     = GROUP_NAMES.get(_dom_key, _dom_key)
    _dom_count    = int(_g_counts[0]["len"][0])
    _n_watch_top  = top_feats_df.filter(pl.col("verdict") == "WATCH").shape[0]

    _top_table = "\n".join(
        f"    | {i+1} | `{f}` | {GROUP_NAMES.get(f[0], '?')} | {auc:.3f} |"
        for i, (f, auc) in enumerate(zip(_top3, _top3_aucs))
    )

    mo.md(f"""
    ## บทที่ 6 — Key Insights

    > 5 สิ่งที่ Data Scientist ควรบอก Business Stakeholder

    ---

    ### 1️⃣  ปัญหาเป็น Class Imbalance — ห้ามใช้ Accuracy เป็น metric
    Dataset มี **{n_rows:,} customers** แต่ defaulter มีแค่ **{target_rate:.1%}**
    → model ที่ทายว่าทุกคนไม่ default ก็ได้ accuracy **{1-target_rate:.0%}** แต่ไร้ประโยชน์ทางธุรกิจ
    → ต้องใช้ **AUC-ROC** หรือ **AP (Average Precision)** แทน

    ---

    ### 2️⃣  Pipeline กรอง Leakage ออกได้ {n_block} Features
    จาก {n_model + n_block:,} features ที่ผ่าน corr-filter → **{n_block} ถูก BLOCK** เพราะเสี่ยง leakage

    | Verdict | จำนวน | ใช้ใน model? |
    |---------|-------|-------------|
    | ✅ CLEAN | {n_clean} | ใช่ |
    | ⚠️ WATCH | {n_watch} | ใช่ (monitor) |
    | 🚫 BLOCK | {n_block} | ไม่ — leakage risk |

    ถ้าใช้ BLOCK features model จะได้ AUC สูงใน training
    แต่จะ **fail ใน production** เพราะ feature นั้นไม่มีจริงตอน predict

    ---

    ### 3️⃣  กลุ่ม {_dom_name} ขับเคลื่อน Prediction มากสุด
    ใน top {TOP_N} features → **{_dom_count} จาก {TOP_N}** มาจากกลุ่ม **{_dom_name} ({_dom_key}_)**
    → ประวัติ{_dom_name.lower()}ของ customer เป็น signal ที่แข็งแกร่งที่สุด

    ---

    ### 4️⃣  Top 3 Features ที่ Model เชื่อมากสุด
    | อันดับ | Feature | กลุ่ม | Single-feat AUC |
    |--------|---------|-------|----------------|
    {_top_table}

    features เหล่านี้มี linear + non-linear signal กับ target สูงสุด
    → ถ้าจะ explain model ให้ stakeholder ให้เริ่มจากตรงนี้

    ---

    ### 5️⃣  {_n_watch_top} จาก Top {TOP_N} Features เป็น WATCH — ต้อง Review
    WATCH = leakage pipeline พบ signal เล็กน้อย แต่ยังไม่ถึง BLOCK
    อาจเป็น:
    - **Genuine signal** ที่ดูเหมือน leakage แต่จริงๆ มีอยู่ใน production → ควรเก็บไว้
    - **Soft leakage** ที่รั่วนิดหน่อย → ควรตัดออก

    **Next step:** ให้ domain expert review WATCH features ทีละตัว
    ก่อนนำ model ไปใช้จริงใน production

    ---

    > **Roadmap ต่อไป:**
    > 1. Manual review WATCH features ร่วมกับ domain expert
    > 2. Feature engineering จากกลุ่มที่ over-represented → อาจได้ signal ที่แรงขึ้น
    > 3. ลอง interaction terms ระหว่าง top features ต่างกลุ่ม
    > 4. Calibration — แปลง AUC model เป็น probability ที่ถูกต้องสำหรับ credit scoring
    > 5. Threshold optimization — balance ระหว่าง precision vs recall ตาม business cost
    """)
    return


if __name__ == "__main__":
    app.run()
