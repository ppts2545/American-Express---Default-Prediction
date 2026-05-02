import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    from collections import Counter
    import marimo as mo
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import roc_auc_score

    mo.md("""
    # Risk Feature Analysis
    ## วิเคราะห์เชิงลึก: Feature ที่ถูก Flag จาก Leakage Detection

    **เป้าหมาย:** ตัดสินใจว่า feature ไหนเป็น leakage จริง และไหนเป็น False Positive

    | ขั้นตอน | คำถาม |
    |---------|-------|
    | **1. Overview** | ภาพรวม risk scores เป็นอย่างไร? |
    | **2. Group Analysis** | กลุ่ม feature ไหน (P_, B_, D_...) มี risk สูงสุด? |
    | **3. BLOCK Deep-Dive** | distribution ของ BLOCK features ดูสมเหตุสมผลไหม? |
    | **4. WATCH Overview** | WATCH features มี pattern น่าสังเกตอะไรบ้าง? |
    | **5. Time Stability** | BLOCK+WATCH เสถียรตามเวลาไหม? |
    | **6. Correlation Structure** | BLOCK+WATCH ซ้ำซ้อนกันเองหรือเปล่า? |
    | **7. A/B Experiment** | ถ้าตัด BLOCK ออก AUC เปลี่ยนแค่ไหน? → หา False Positive |
    | **8. False Positive Checklist** | สรุปหลักฐานรายตัว — ตัดสินใจ KEEP หรือ DROP |
    """)
    return (
        Path,
        StratifiedKFold,
        cs,
        lgb,
        mo,
        np,
        pl,
        plt,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _(Path, cs, pl):
    _ROOT = Path(__file__).resolve().parent
    while not (_ROOT / "pixi.toml").exists():
        _ROOT = _ROOT.parent

    # ── โหลด risk scores จาก leakage_detection.py ────────────────────────────
    _risk_path = _ROOT / "data/processed/feature_risk_scores.parquet"
    assert _risk_path.exists(), (
        "ไม่พบ feature_risk_scores.parquet — รัน notebook/eda/leakage_detection.py ก่อน"
    )
    risk_df = pl.read_parquet(_risk_path)

    # ── โหลด training data ────────────────────────────────────────────────────
    _train  = pl.read_parquet(_ROOT / "data/processed/train_features.parquet")
    _labels = pl.read_parquet(_ROOT / "data/processed/train_labels.parquet")
    df = _train.join(_labels.select(["customer_ID", "target"]), on="customer_ID")

    numeric_cols = df.select(cs.numeric().exclude("target")).columns
    df_sample    = df.sample(n=min(20_000, df.shape[0]), seed=42)

    # ── แยกกลุ่มตาม verdict ──────────────────────────────────────────────────
    block_features = risk_df.filter(pl.col("verdict") == "BLOCK")["feature"].to_list()
    watch_features = risk_df.filter(pl.col("verdict") == "WATCH")["feature"].to_list()
    clean_features = risk_df.filter(pl.col("verdict") == "CLEAN")["feature"].to_list()
    return (
        block_features,
        clean_features,
        df,
        df_sample,
        numeric_cols,
        risk_df,
        watch_features,
    )


@app.cell
def _(block_features, clean_features, mo, np, plt, risk_df, watch_features):
    def _make_overview():
        fig = plt.figure(figsize=(16, 5))
        gs  = fig.add_gridspec(1, 3, wspace=0.35)

        # ── Verdict pie ──────────────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0])
        _counts  = [len(block_features), len(watch_features), len(clean_features)]
        _labels  = [f"BLOCK\n({_counts[0]})", f"WATCH\n({_counts[1]})", f"CLEAN\n({_counts[2]})"]
        _colors  = ["#d62728", "#ff7f0e", "#2ca02c"]
        _explode = [0.05, 0.02, 0]
        ax0.pie(_counts, labels=_labels, colors=_colors, explode=_explode,
                autopct="%1.0f%%", startangle=140,
                textprops={"fontsize": 9, "fontweight": "bold"})
        ax0.set_title("Verdict Distribution", fontweight="bold")

        # ── Risk score histogram ─────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[1])
        _scores = risk_df["risk_score"].to_numpy()
        _bins   = np.arange(-0.5, 6.5, 1)
        _score_colors = {0: "#2ca02c", 1: "#98df8a", 2: "#ff7f0e", 3: "#d62728", 4: "#8B0000", 5: "#4B0000"}
        for _s in range(6):
            _cnt = (_scores == _s).sum()
            ax1.bar(_s, _cnt, color=_score_colors[_s], edgecolor="white", width=0.7)
            if _cnt > 0:
                ax1.text(_s, _cnt + 0.5, str(_cnt), ha="center", va="bottom",
                         fontsize=8, fontweight="bold")
        ax1.set_xticks(range(6))
        ax1.set_xticklabels([f"score={i}" for i in range(6)], fontsize=8)
        ax1.set_ylabel("# Features")
        ax1.set_title("Risk Score Distribution\n(0 = clean, 5 = flagged ทุกเทคนิค)",
                      fontweight="bold")
        ax1.axvspan(2.5, 5.5, alpha=0.08, color="red", label="BLOCK zone")
        ax1.legend(fontsize=8)

        # ── Cohen's d ranking top 20 ─────────────────────────────────────────
        ax2 = fig.add_subplot(gs[2])
        _top = risk_df.sort("cohen_d", descending=True).head(20)
        _feats  = _top["feature"].to_list()
        _ds     = _top["cohen_d"].to_numpy()
        _vdict  = dict(zip(_top["feature"].to_list(), _top["verdict"].to_list()))
        _bar_colors = ["#d62728" if _vdict[f] == "BLOCK"
                       else "#ff7f0e" if _vdict[f] == "WATCH"
                       else "#2ca02c" for f in _feats]
        ax2.barh(_feats[::-1], _ds[::-1], color=_bar_colors[::-1])
        ax2.axvline(3.0, color="red",    linestyle="--", linewidth=1.2, label="d=3 (BLOCK)")
        ax2.axvline(1.5, color="orange", linestyle="--", linewidth=1.2, label="d=1.5 (WATCH)")
        ax2.set_xlabel("Cohen's d (robust)")
        ax2.set_title("Top 20 Features by Cohen's d", fontweight="bold")
        ax2.legend(fontsize=8)

        plt.suptitle("Feature Risk Overview", fontsize=13, fontweight="bold", y=1.01)
        plt.tight_layout()
        return fig

    fig_overview = _make_overview()

    mo.md(f"""
    ## 1. Overview Dashboard

    | Verdict | จำนวน | ความหมาย |
    |---------|-------|---------|
    | 🚨 BLOCK | **{len(block_features)}** | risk_score ≥ 3 — ถูกกรองออกจาก baseline CV |
    | ⚠️ WATCH | **{len(watch_features)}** | risk_score 1–2 — ยังอยู่ใน pool แต่ต้อง monitor |
    | ✅ CLEAN | **{len(clean_features)}** | risk_score = 0 — Golden Features |
    """)
    return (fig_overview,)


@app.cell
def _(fig_overview):
    fig_overview
    return


@app.cell
def _(
    block_features,
    clean_features,
    mo,
    np,
    pl,
    plt,
    risk_df,
    watch_features,
):
    def _make_group():
        # ดึง prefix ของชื่อ feature (ตัวอักษรแรกก่อน underscore)
        _get_group = lambda f: f.split("_")[0] if "_" in f else f[0]

        _groups = sorted(set(_get_group(f) for f in risk_df["feature"].to_list()))

        _data = {}
        for _g in _groups:
            _data[_g] = {
                "BLOCK": sum(1 for f in block_features if _get_group(f) == _g),
                "WATCH": sum(1 for f in watch_features if _get_group(f) == _g),
                "CLEAN": sum(1 for f in clean_features if _get_group(f) == _g),
            }

        _groups_sorted = sorted(_groups, key=lambda g: _data[g]["BLOCK"] + _data[g]["WATCH"], reverse=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        # ── stacked bar: BLOCK/WATCH/CLEAN per group ─────────────────────────
        _x       = np.arange(len(_groups_sorted))
        _b_vals  = [_data[g]["BLOCK"] for g in _groups_sorted]
        _w_vals  = [_data[g]["WATCH"] for g in _groups_sorted]
        _c_vals  = [_data[g]["CLEAN"] for g in _groups_sorted]
        _b_bars  = axes[0].bar(_x, _b_vals, color="#d62728", label="BLOCK", alpha=0.9)
        _w_bars  = axes[0].bar(_x, _w_vals, bottom=_b_vals, color="#ff7f0e", label="WATCH", alpha=0.9)
        axes[0].bar(_x, _c_vals,
                    bottom=[b + w for b, w in zip(_b_vals, _w_vals)],
                    color="#2ca02c", label="CLEAN", alpha=0.7)
        axes[0].set_xticks(_x)
        axes[0].set_xticklabels(_groups_sorted, fontsize=10, fontweight="bold")
        axes[0].set_ylabel("# Features")
        axes[0].set_title("Risk Distribution by Feature Group\n(P=Payment, B=Balance, D=Delinquency, S=Spend, R=Risk, F=?)",
                          fontweight="bold")
        axes[0].legend()

        # ── BLOCK ratio per group ─────────────────────────────────────────────
        _totals     = [_data[g]["BLOCK"] + _data[g]["WATCH"] + _data[g]["CLEAN"] for g in _groups_sorted]
        _block_ratio = [b / t if t > 0 else 0 for b, t in zip(_b_vals, _totals)]
        _bar_colors  = ["#d62728" if r > 0.5 else "#ff7f0e" if r > 0.2 else "#2ca02c"
                        for r in _block_ratio]
        axes[1].bar(_groups_sorted, _block_ratio, color=_bar_colors, edgecolor="white")
        axes[1].axhline(0.5, color="red",    linestyle="--", linewidth=1.2, label="50% BLOCK")
        axes[1].axhline(0.2, color="orange", linestyle="--", linewidth=1.2, label="20% BLOCK")
        axes[1].set_ylabel("BLOCK Ratio")
        axes[1].set_title("BLOCK Rate per Group\n(สูง = กลุ่มนี้เสี่ยงสูง)",
                          fontweight="bold")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()

        for _i, (_g, _r) in enumerate(zip(_groups_sorted, _block_ratio)):
            if _r > 0:
                axes[1].text(_i, _r + 0.02, f"{_r:.0%}", ha="center", va="bottom",
                             fontsize=9, fontweight="bold")

        plt.tight_layout()
        return fig, _data, _groups_sorted

    import re as _re

    fig_group, _group_data, _groups_sorted = _make_group()

    _top_risk_group = max(_groups_sorted, key=lambda g: _group_data[g]["BLOCK"])

    # คำอธิบาย group ใน AmEx dataset
    _group_desc = risk_df.with_columns(
        pl.col("feature").str.split("_").list.first().alias("group")
    ).group_by("group").agg([
        pl.len().alias("total"),
        (pl.col("verdict") == "BLOCK").sum().alias("block"),
        (pl.col("verdict") == "WATCH").sum().alias("watch"),
        pl.col("cohen_d").mean().alias("avg_d"),
    ]).sort("block", descending=True)

    _rows = "\n    ".join(
        f"| **{r['group']}** | {r['total']} | {r['block']} 🚨 | {r['watch']} ⚠️ | {r['avg_d']:.2f} |"
        for r in _group_desc.iter_rows(named=True)
    )

    mo.md(f"""
    ## 2. Feature Group Analysis

    กลุ่ม **{_top_risk_group}_** มี BLOCK features มากที่สุด — ต้องตรวจพิเศษ

    | Group | Total | BLOCK | WATCH | Avg Cohen's d |
    |-------|-------|-------|-------|--------------|
    {_rows}
    """)
    return (fig_group,)


@app.cell
def _(fig_group):
    fig_group
    return


@app.cell
def _(block_features, df_sample, mo, np, pl, plt, risk_df):
    def _make_block_kde():
        if not block_features:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "ไม่มี BLOCK features ✅", ha="center", va="center",
                    fontsize=13, color="#2ca02c", fontweight="bold")
            ax.axis("off")
            return fig

        _risk_lookup = dict(zip(risk_df["feature"].to_list(), risk_df["cohen_d"].to_list()))
        _score_lookup = dict(zip(risk_df["feature"].to_list(), risk_df["risk_score"].to_list()))

        N_COLS  = 4
        N_ROWS  = -(-len(block_features) // N_COLS)
        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, N_ROWS * 3.5), squeeze=False)
        flat    = axes.flatten()

        for i, feat in enumerate(block_features):
            ax  = flat[i]
            v0  = df_sample.filter(pl.col("target") == 0)[feat].drop_nulls().to_numpy()
            v1  = df_sample.filter(pl.col("target") == 1)[feat].drop_nulls().to_numpy()

            if len(v0) > 1 and len(v1) > 1:
                lo, hi = np.percentile(np.concatenate([v0, v1]), [1, 99])
                v0 = v0[(v0 >= lo) & (v0 <= hi)]
                v1 = v1[(v1 >= lo) & (v1 <= hi)]

            ax.hist(v0, bins=50, density=True, alpha=0.55, color="#2196F3", label="Non-default (0)")
            ax.hist(v1, bins=50, density=True, alpha=0.55, color="#F44336", label="Default (1)")

            _d     = _risk_lookup.get(feat, 0)
            _score = _score_lookup.get(feat, 0)
            ax.set_title(f"{feat}\nd={_d:.2f}  score={_score}/5 🚨",
                         color="#d62728", fontsize=8, fontweight="bold")
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)

        for j in range(len(block_features), len(flat)):
            flat[j].set_visible(False)

        plt.suptitle(f"BLOCK Features — Distribution by Class ({len(block_features)} features)",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    fig_block_kde = _make_block_kde()

    mo.md(f"""
    ## 3. BLOCK Features — Distribution Deep-Dive

    **วิธีอ่าน:**
    - 🔵 Non-default (0) vs 🔴 Default (1)
    - ถ้า 2 สีแยกกันชัดเจน = feature แยก class ได้ดี → **อาจเป็น signal จริง (False Positive)**
    - ถ้า Default (1) กระจุกตัวที่ค่าเดียว เช่น 0 หรือ max → **น่าสงสัย leakage**

    พบ **{len(block_features)}** BLOCK features — ดู plot ด้านล่าง
    """)
    return (fig_block_kde,)


@app.cell
def _(fig_block_kde):
    fig_block_kde
    return


@app.cell
def _(df_sample, mo, np, pl, plt, risk_df, watch_features):
    def _make_watch_kde():
        if not watch_features:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "ไม่มี WATCH features ✅", ha="center", va="center",
                    fontsize=13, color="#2ca02c", fontweight="bold")
            ax.axis("off")
            return fig

        _show    = watch_features[:16]
        _risk_lookup  = dict(zip(risk_df["feature"].to_list(), risk_df["cohen_d"].to_list()))
        _score_lookup = dict(zip(risk_df["feature"].to_list(), risk_df["risk_score"].to_list()))

        N_COLS = 4
        N_ROWS = -(-len(_show) // N_COLS)
        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, N_ROWS * 3.2), squeeze=False)
        flat = axes.flatten()

        for i, feat in enumerate(_show):
            ax = flat[i]
            v0 = df_sample.filter(pl.col("target") == 0)[feat].drop_nulls().to_numpy()
            v1 = df_sample.filter(pl.col("target") == 1)[feat].drop_nulls().to_numpy()

            if len(v0) > 1 and len(v1) > 1:
                lo, hi = np.percentile(np.concatenate([v0, v1]), [1, 99])
                v0 = v0[(v0 >= lo) & (v0 <= hi)]
                v1 = v1[(v1 >= lo) & (v1 <= hi)]

            ax.hist(v0, bins=40, density=True, alpha=0.55, color="#2196F3", label="Non-default (0)")
            ax.hist(v1, bins=40, density=True, alpha=0.55, color="#F44336", label="Default (1)")

            _d     = _risk_lookup.get(feat, 0)
            _score = _score_lookup.get(feat, 0)
            ax.set_title(f"{feat}\nd={_d:.2f}  score={_score}/5 ⚠️",
                         color="#ff7f0e", fontsize=8, fontweight="bold")
            ax.legend(fontsize=6)
            ax.tick_params(labelsize=7)

        for j in range(len(_show), len(flat)):
            flat[j].set_visible(False)

        _extra = f" (แสดง {len(_show)} จาก {len(watch_features)})" if len(watch_features) > 16 else ""
        plt.suptitle(f"WATCH Features — Distribution by Class{_extra}",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    fig_watch_kde = _make_watch_kde()

    mo.md(f"""
    ## 4. WATCH Features Overview

    **{len(watch_features)}** features อยู่ใน watchlist — ยังอยู่ใน feature pool แต่ต้อง monitor

    pattern ที่น่าสังเกต:
    - ถ้า distribution ทับซ้อนกันมาก → signal อ่อน อาจ noise
    - ถ้าแยกกันได้ดีพอสมควร → อาจมี signal — ดูร่วมกับ AUC ใน Cell 7
    """)
    return (fig_watch_kde,)


@app.cell
def _(fig_watch_kde):
    fig_watch_kde
    return


@app.cell
def _(block_features, df, mo, np, pl, plt, watch_features):
    def _make_time_stability():
        _feats = (block_features + watch_features)[:12]
        if not _feats:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "ไม่มี feature ที่ต้องตรวจ ✅", ha="center", va="center", fontsize=13)
            ax.axis("off")
            return fig

        n   = df.shape[0]
        _df = df.with_row_index("__ridx").with_columns(
            pl.when(pl.col("__ridx") < n // 4).then(pl.lit("Q1"))
            .when(pl.col("__ridx") < n // 2).then(pl.lit("Q2"))
            .when(pl.col("__ridx") < 3 * n // 4).then(pl.lit("Q3"))
            .otherwise(pl.lit("Q4"))
            .alias("__bin")
        )

        _agg = (
            _df.group_by(["__bin", "target"])
            .agg([pl.col(c).mean().alias(c) for c in _feats])
            .sort(["__bin", "target"])
        )
        _t0   = _agg.filter(pl.col("target") == 0)
        _t1   = _agg.filter(pl.col("target") == 1)
        _bins = ["Q1", "Q2", "Q3", "Q4"]
        _x    = list(range(4))

        N_COLS = 3
        N_ROWS = -(-len(_feats) // N_COLS)
        fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(15, N_ROWS * 3.5), squeeze=False)
        flat = axes.flatten()

        for i, feat in enumerate(_feats):
            ax       = flat[i]
            t0_means = _t0[feat].fill_null(0).to_list()
            t1_means = _t1[feat].fill_null(0).to_list()
            gaps     = [abs(a - b) for a, b in zip(t0_means, t1_means)]
            gap_cv   = float(np.std(gaps) / (np.mean(gaps) + 1e-8))

            ax.plot(_x, t0_means, "o-", color="#2196F3", label="Non-default", linewidth=2)
            ax.plot(_x, t1_means, "s-", color="#F44336", label="Default", linewidth=2)
            ax.fill_between(_x, t0_means, t1_means, alpha=0.1, color="gray")

            _is_block = feat in block_features
            _color    = "#d62728" if _is_block else "#ff7f0e"
            _badge    = "🚨" if _is_block else "⚠️"
            ax.set_title(f"{feat} {_badge}\ngap_cv = {gap_cv:.2f}",
                         color=_color, fontsize=8, fontweight="bold")
            ax.set_xticks(_x)
            ax.set_xticklabels(_bins, fontsize=8)
            ax.set_ylabel("Mean", fontsize=7)
            ax.legend(fontsize=6)
            ax.grid(alpha=0.3)

        for j in range(len(_feats), len(flat)):
            flat[j].set_visible(False)

        plt.suptitle("Time Stability — BLOCK 🚨 + WATCH ⚠️ Features\n"
                     "gap_cv สูง = gap ระหว่าง class เปลี่ยนตามเวลา = temporal leakage เสี่ยง",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        return fig

    fig_time = _make_time_stability()

    mo.md(f"""
    ## 5. Time Stability — BLOCK + WATCH

    **วิธีอ่าน `gap_cv`:**
    - ต่ำ (< 0.1) → gap ระหว่าง class คงที่ตลอด = เสถียร ✅
    - กลาง (0.1–0.3) → เปลี่ยนบ้างตามธรรมชาติ — ปกติ
    - สูง (> 0.3) → gap แปรปรวนมาก = temporal pattern → เสี่ยง leakage

    แสดง top 12 features (BLOCK ก่อน → WATCH)
    """)
    return (fig_time,)


@app.cell
def _(fig_time):
    fig_time
    return


@app.cell
def _(block_features, df_sample, mo, np, plt, watch_features):
    def _make_corr_heatmap():
        _feats = (block_features + watch_features)[:40]
        if len(_feats) < 2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "ต้องมีอย่างน้อย 2 features", ha="center", va="center", fontsize=13)
            ax.axis("off")
            return fig

        _X   = df_sample.select(_feats).fill_null(0).to_numpy()
        _mat = np.corrcoef(_X.T)

        fig, ax = plt.subplots(figsize=(max(10, len(_feats) * 0.4),
                                        max(8,  len(_feats) * 0.4)))
        im = ax.imshow(_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

        ax.set_xticks(range(len(_feats)))
        ax.set_yticks(range(len(_feats)))
        ax.set_xticklabels(_feats, rotation=90, fontsize=6.5)
        ax.set_yticklabels(_feats, fontsize=6.5)

        # ขีดเส้นแบ่ง BLOCK / WATCH
        _n_block = min(len(block_features), len(_feats))
        if 0 < _n_block < len(_feats):
            ax.axhline(_n_block - 0.5, color="black", linewidth=2)
            ax.axvline(_n_block - 0.5, color="black", linewidth=2)
            ax.text(_n_block / 2, -1.5, "BLOCK 🚨", ha="center", fontsize=8,
                    color="#d62728", fontweight="bold")
            ax.text(_n_block + (len(_feats) - _n_block) / 2, -1.5, "WATCH ⚠️",
                    ha="center", fontsize=8, color="#ff7f0e", fontweight="bold")

        ax.set_title(
            "Correlation Structure: BLOCK + WATCH Features\n"
            "แดงเข้ม = corr สูง (ซ้ำซ้อน) | น้ำเงินเข้ม = corr ลบสูง | ขาว = ไม่สัมพันธ์",
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    fig_corr = _make_corr_heatmap()

    _n_shown = min(len(block_features) + len(watch_features), 40)
    mo.md(f"""
    ## 6. Correlation Structure

    **วิธีอ่าน:**
    - ถ้า BLOCK features corr กันสูงมาก (แดงเข้ม) = encode ข้อมูลเดียวกัน
      → drop บางตัวไม่กระทบ AUC (ไม่ใช่ False Positive)
    - ถ้า BLOCK corr กับ WATCH สูง = อาจเป็น "คู่แฝด" ของกันและกัน
    - เส้นดำขีดแบ่ง BLOCK (บนซ้าย) กับ WATCH (ล่างขวา)

    แสดง {_n_shown} features (BLOCK ก่อน → WATCH)
    """)
    return (fig_corr,)


@app.cell
def _(fig_corr):
    fig_corr
    return


@app.cell
def _(
    StratifiedKFold,
    block_features,
    df,
    lgb,
    mo,
    np,
    numeric_cols,
    plt,
    roc_auc_score,
    train_test_split,
):
    def _run_ab():
        _y   = df["target"].to_numpy()
        _idx = np.arange(len(_y))
        _skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # ใช้ numeric_cols (cs.numeric().exclude("target")) — ไม่มี string columns
        _feats_with    = numeric_cols
        _feats_without = [c for c in numeric_cols if c not in set(block_features)]

        _params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.05, "num_leaves": 31, "max_depth": 5,
            "min_child_samples": 100, "feature_fraction": 0.8,
            "verbosity": -1, "random_state": 42,
        }

        def _cv_auc(feats):
            _oof = np.zeros(len(_y))
            for _tr, _val in _skf.split(_idx, _y):
                _inner, _es = train_test_split(_tr, test_size=0.1, random_state=0, stratify=_y[_tr])
                _Xtr = df[list(map(int, _inner))].select(feats).fill_null(0).to_numpy()
                _yes = df[list(map(int, _es))].select(feats).fill_null(0).to_numpy()
                _Xv  = df[list(map(int, _val))].select(feats).fill_null(0).to_numpy()
                _ytr = _y[_inner]; _yyes = _y[_es]; _yv = _y[_val]
                _dtrain = lgb.Dataset(_Xtr, label=_ytr)
                _dval   = lgb.Dataset(_yes, label=_yyes, reference=_dtrain)
                _m      = lgb.train(
                    _params, _dtrain, num_boost_round=500, valid_sets=[_dval],
                    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(False)],
                )
                _oof[_val] = _m.predict(_Xv)
            return roc_auc_score(_y, _oof)

        auc_with    = _cv_auc(_feats_with)
        auc_without = _cv_auc(_feats_without)
        return auc_with, auc_without

    auc_with, auc_without = _run_ab()
    auc_diff = auc_with - auc_without

    def _plot_ab():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # ── bar comparison ────────────────────────────────────────────────────
        _labels = ["WITH BLOCK\n(ก่อนกรอง)", "WITHOUT BLOCK\n(หลังกรอง)"]
        _vals   = [auc_with, auc_without]
        _colors = ["#ff7f0e", "#2196F3"]
        _bars   = axes[0].bar(_labels, _vals, color=_colors, edgecolor="white", width=0.5)
        for _bar, _v in zip(_bars, _vals):
            axes[0].text(_bar.get_x() + _bar.get_width() / 2,
                         _bar.get_height() + 0.0005,
                         f"{_v:.5f}", ha="center", va="bottom",
                         fontsize=11, fontweight="bold")
        _ymin = min(_vals) - 0.01
        _ymax = max(_vals) + 0.005
        axes[0].set_ylim(_ymin, _ymax)
        axes[0].set_ylabel("OOF AUC (3-fold)")
        axes[0].set_title("A/B: WITH vs WITHOUT BLOCK Features", fontweight="bold")

        # ── interpretation ────────────────────────────────────────────────────
        if auc_diff > 0.003:
            _verdict = f"🔴 AUC ลด {auc_diff:.4f} → BLOCK มี signal จริง\n→ False Positive! ตรวจรายตัวก่อน drop"
            _col     = "#d62728"
        elif auc_diff > 0:
            _verdict = f"🟠 AUC ลดเล็กน้อย {auc_diff:.4f}\n→ BLOCK มี signal บ้าง ตรวจรายตัว"
            _col     = "#ff7f0e"
        elif auc_diff >= -0.001:
            _verdict = f"✅ AUC แทบไม่เปลี่ยน (Δ={auc_diff:.4f})\n→ BLOCK ไม่มี signal — drop ได้อย่างปลอดภัย"
            _col     = "#2ca02c"
        else:
            _verdict = f"✅ AUC ดีขึ้น {abs(auc_diff):.4f} เมื่อตัด BLOCK ออก\n→ BLOCK เป็น noise/leakage — drop แน่นอน"
            _col     = "#2ca02c"

        axes[1].text(0.5, 0.5, _verdict, ha="center", va="center", fontsize=13,
                     color=_col, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.6", facecolor="#f9f9f9",
                               edgecolor=_col, linewidth=2))
        axes[1].set_title("Interpretation", fontweight="bold")
        axes[1].axis("off")

        plt.tight_layout()
        return fig

    fig_ab = _plot_ab()

    if auc_diff > 0.003:
        _action = "**ระวัง False Positive** — BLOCK features มี signal จริง ตรวจรายตัวใน Cell 8 ก่อนตัดสินใจ"
    elif auc_diff > 0:
        _action = "**ตรวจรายตัว** — BLOCK มี signal บ้างเล็กน้อย บางตัวอาจ False Positive"
    else:
        _action = "**drop ได้อย่างปลอดภัย** — BLOCK features ไม่ได้ช่วย model เลย"

    mo.md(f"""
    ## 7. A/B Experiment — หา False Positive

    เปรียบเทียบ OOF AUC ระหว่าง:
    - **WITH BLOCK**: ใช้ทุก feature
    - **WITHOUT BLOCK**: เอา {len(block_features)} BLOCK features ออก

    | | AUC |
    |-|-----|
    | WITH BLOCK | **{auc_with:.5f}** |
    | WITHOUT BLOCK | **{auc_without:.5f}** |
    | **Δ AUC** | **{auc_diff:+.5f}** |

    → {_action}

    **เกณฑ์อ่านผล:**
    | Δ AUC | ความหมาย |
    |-------|---------|
    | > +0.003 | 🔴 False Positive — BLOCK มี signal จริง ห้าม drop ทันที |
    | +0.001 ถึง +0.003 | 🟠 ตรวจรายตัว — บางตัวอาจ False Positive |
    | -0.001 ถึง +0.001 | ✅ ปลอดภัย — BLOCK ไม่มี signal drop ได้ |
    | < -0.001 | ✅ BLOCK เป็น noise — AUC ดีขึ้นด้วยซ้ำเมื่อตัดออก |
    """)
    return auc_diff, auc_with, auc_without, fig_ab


@app.cell
def _(fig_ab):
    fig_ab
    return


@app.cell
def _(
    auc_diff,
    auc_with,
    auc_without,
    block_features,
    df_sample,
    mo,
    np,
    pl,
    plt,
    risk_df,
):
    def _make_checklist():
        if not block_features:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "ไม่มี BLOCK features ✅", ha="center", va="center",
                    fontsize=13, color="#2ca02c", fontweight="bold")
            ax.axis("off")
            return fig

        _risk_lookup  = {r["feature"]: r for r in risk_df.iter_rows(named=True)}
        _show = block_features[:8]

        N_COLS = 4
        N_ROWS = -(-len(_show) // N_COLS)
        fig, axes = plt.subplots(N_ROWS, N_COLS * 2,
                                 figsize=(18, N_ROWS * 4), squeeze=False)

        for i, feat in enumerate(_show):
            _row  = i // N_COLS
            _col  = (i % N_COLS) * 2
            ax_dist = axes[_row][_col]
            ax_info = axes[_row][_col + 1]

            # ── distribution ────────────────────────────────────────────────
            v0 = df_sample.filter(pl.col("target") == 0)[feat].drop_nulls().to_numpy()
            v1 = df_sample.filter(pl.col("target") == 1)[feat].drop_nulls().to_numpy()
            if len(v0) > 1 and len(v1) > 1:
                lo, hi = np.percentile(np.concatenate([v0, v1]), [1, 99])
                v0 = v0[(v0 >= lo) & (v0 <= hi)]
                v1 = v1[(v1 >= lo) & (v1 <= hi)]
            ax_dist.hist(v0, bins=40, density=True, alpha=0.55, color="#2196F3", label="Non-default")
            ax_dist.hist(v1, bins=40, density=True, alpha=0.55, color="#F44336", label="Default")
            ax_dist.set_title(feat, fontsize=8, fontweight="bold", color="#d62728")
            ax_dist.legend(fontsize=6)
            ax_dist.tick_params(labelsize=6)

            # ── info panel ──────────────────────────────────────────────────
            _r = _risk_lookup.get(feat, {})
            _flags = {
                "Class Sep":    "🔴" if _r.get("flag_class_sep")     else "✅",
                "Time Stable":  "🔴" if _r.get("flag_time_stability") else "✅",
                "Adv(self)":    "🔴" if _r.get("flag_adv_self")      else "✅",
                "Adv(T/T)":     "🔴" if _r.get("flag_adv_tt")       else "✅",
                "Interaction":  "🔴" if _r.get("flag_interaction")   else "✅",
            }
            _text = "\n".join([
                f"Score: {_r.get('risk_score', '?')}/5",
                f"Cohen's d: {_r.get('cohen_d', '?'):.3f}",
                "",
                *[f"{k:13s} {v}" for k, v in _flags.items()],
                "",
                "→ ตรวจ business logic:",
                f"  '{feat}'",
                "  รู้ 'อนาคต' ไหม?",
            ])
            ax_info.text(0.05, 0.95, _text, transform=ax_info.transAxes,
                         va="top", ha="left", fontsize=7.5, fontfamily="monospace",
                         bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff8f0",
                                   edgecolor="#d62728", linewidth=1.2))
            ax_info.axis("off")

        # ซ่อน axes ที่เหลือ
        total_axes = N_ROWS * N_COLS
        for j in range(len(_show), total_axes):
            _row = j // N_COLS
            _col = (j % N_COLS) * 2
            axes[_row][_col].set_visible(False)
            axes[_row][_col + 1].set_visible(False)

        plt.suptitle(f"False Positive Checklist — BLOCK Features (top {len(_show)})\n"
                     "ดู distribution + flag ทีละตัว แล้วตัดสินใจ KEEP หรือ DROP",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        return fig

    fig_checklist = _make_checklist()

    # สรุปการตัดสินใจ
    _rows = "\n    ".join(
        f"| `{f}` | ❓ | ❓ | (ตรวจ business logic) |"
        for f in block_features
    )

    mo.md(f"""
    ## 8. False Positive Checklist

    **ขั้นตอนการตัดสินใจ:**
    1. ดู distribution (ซ้าย) — ถ้าแยก class ได้สมเหตุสมผล = อาจ False Positive
    2. ดู flag breakdown (ขวา) — flag เยอะ = น่าสงสัย leakage มากกว่า
    3. อ่านชื่อ feature — มี "last", "latest", "future" ในชื่อไหม? = leakage ชัดเจน
    4. อ้างอิง A/B (Cell 7): Δ AUC = **{auc_diff:+.5f}**
       - WITH: {auc_with:.5f} → WITHOUT: {auc_without:.5f}

    | Feature | KEEP? | DROP? | เหตุผล |
    |---------|-------|-------|--------|
    {_rows if _rows else "| ไม่มี BLOCK features ✅ | | | |"}

    ---
    หลังตรวจแล้ว → ใส่ features ที่ยืนยันว่าเป็น leakage จริงใน `MANUAL_BLOCK` ใน baseline.py
    หรือปล่อยให้ Gate 1.5 auto-block ต่อไปก็ได้ถ้ายังไม่แน่ใจ
    """)
    return (fig_checklist,)


@app.cell
def _(fig_checklist):
    fig_checklist
    return


if __name__ == "__main__":
    app.run()
