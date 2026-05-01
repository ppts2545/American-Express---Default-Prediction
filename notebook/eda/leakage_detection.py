import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import tomllib
    import marimo as mo
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import matplotlib.pyplot as plt
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.feature_selection import mutual_info_classif

    mo.md("""
    # Feature-Target Interaction Analysis
    ## Professional EDA สำหรับดักจับ Leakage — 9 เทคนิค (เรียงจากเร็ว → ช้า)

    **เป้าหมาย:** พิสูจน์ว่าความแม่นยำมาจาก signal จริง ไม่ใช่ leakage

    | # | เทคนิค | กลุ่ม | จับอะไร | ความเร็ว |
    |---|--------|-------|---------|---------|
    | 1 | Null Pattern | ⚡ Fast | missingness เองเป็น signal? | O(n) |
    | 2 | Class Separation (Cohen's d) | ⚡ Fast | แยก class สมบูรณ์เกินไป? | O(n) vectorized |
    | 3 | Mutual Information | ⚡ Fast | non-linear relationship ซ่อนอยู่? | O(n log n) |
    | 4 | Variance Ratio | ⚡ Fast | spread ต่างกันระหว่าง class ผิดปกติ? | O(n) vectorized |
    | 5 | PSI | 🕐 Temporal | distribution drift ตามเวลา (marginal)? | full data |
    | 6 | Time Stability | 🕐 Temporal | relationship drift ตามเวลา (conditional)? | full data |
    | 7 | Single-Feature AUC | 🐢 Medium | 1 feature ทายได้ดีเกินไป? | O(n log n) + LGB |
    | 8 | Adversarial (Train/Train) | 🔥 Slow | data stationary ตลอดไหม? | LGB training |
    | 9 | Adversarial (Train/Test) | 🔥 Slow | train/test drift? | LGB training |

    Thresholds ทั้งหมดอ่านจาก `config/leakage_thresholds.toml` — แก้ที่นั่นที่เดียว
    """)
    return (
        Path,
        StratifiedKFold,
        cs,
        lgb,
        mo,
        mutual_info_classif,
        np,
        pl,
        plt,
        roc_auc_score,
        tomllib,
    )


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🗺️ Project Workflow — Notebook นี้อยู่ตรงไหนใน Pipeline

    ```
    raw data (Kaggle)
         │
         ▼
    01_preprocess.py
    สร้าง train_features.parquet + train_labels.parquet
         │
         ▼
    ┌─────────────────────────────────────┐
    │  leakage_detection.py  ← คุณอยู่นี่ │
    │  ตรวจ 9 เทคนิค (Fast → Slow)        │
    │  → feature_risk_scores.parquet      │
    └─────────────────────────────────────┘
         │
         ▼
    risk_feature_analysis.py
    วิเคราะห์ผลลึก + A/B Experiment หา False Positive
         │
         ▼
    baseline.py
    Gate 1.5 อ่าน parquet → block BLOCK features → train model
    ```

    ---
    ## 📖 วิธีอ่าน Notebook นี้

    ### Leakage คืออะไร?
    > ข้อมูลที่ "รั่ว" จากอนาคตเข้ามาใน training data
    >
    > ตัวอย่าง: บันทึกการชำระเงิน **หลังจาก** ลูกค้า default
    > → model เห็นผลลัพธ์ตั้งแต่ก่อน predict = โกงข้อสอบ

    ### วิธีอ่านสี
    - **🔴 Flagged** = สัญญาณอันตราย ต้องตรวจ business logic
    - **🟠 Suspicious / Watch** = ระวัง อาจเป็น leakage หรือ false positive
    - **✅ Clean** = ผ่านเทคนิคนี้

    ### Risk Score (0–9)
    - score ≥ 3 → **BLOCK** — ไม่ให้เข้า model ชั่วคราว
    - score ≥ 1 → **WATCH** — เข้า model แต่ติดตาม
    - score = 0 → **CLEAN** — Golden Feature

    > BLOCK ≠ ตัดทิ้งถาวร — ต้องพิสูจน์ต่อใน `risk_feature_analysis.py`

    ### ทำไมต้อง 9 เทคนิค?
    Leakage มีหลายรูปแบบ — แต่ละเทคนิคจับคนละอย่าง:

    | กลุ่ม | เทคนิค | จับอะไรที่กลุ่มอื่นพลาด |
    |-------|--------|------------------------|
    | ⚡ Fast | T1-T4 | static patterns: null, mean, MI, spread |
    | 🕐 Temporal | T5-T6 | การเปลี่ยนแปลงตามเวลา 2 มุมมอง |
    | 🔥 Slow | T7-T9 | confirmation ด้วย model จริง |

    > T5 (PSI) vs T6 (Time Stability) ต่างกัน:
    > - PSI: distribution ของ feature เองเปลี่ยนไหม? (ไม่สนใจ class)
    > - Time Stability: **gap ระหว่าง class** เปลี่ยนไหม? (class-conditional)
    """)
    return


@app.cell
def _(Path, mo, tomllib):
    _ROOT = Path(__file__).resolve().parent
    while not (_ROOT / "pixi.toml").exists():
        _ROOT = _ROOT.parent

    with open(_ROOT / "config" / "leakage_thresholds.toml", "rb") as _f:
        cfg = tomllib.load(_f)

    mo.md(f"""
    ### Config โหลดแล้ว: `config/leakage_thresholds.toml`

    | # | Technique | Threshold |
    |---|-----------|-----------|
    | T1 | Null Pattern | flagged > **{cfg['null_pattern']['flagged_diff']*100:.0f}%** diff |
    | T2 | Cohen's d | flagged d > **{cfg['class_separation']['flagged_d']}** |
    | T3 | Mutual Information | flagged > **{cfg['mutual_information']['flagged']}** |
    | T4 | Variance Ratio | flagged > **{cfg['variance_ratio']['flagged']}x** |
    | T5 | PSI | unstable > **{cfg['psi']['unstable']}** |
    | T6 | Time Stability | gap_cv > **{cfg['time_stability']['unstable_gap_cv']}** |
    | T7 | Single-Feature AUC | flagged > **{cfg['single_feat_auc']['flagged']}** |
    | T8/T9 | Adversarial | suspicious > **{cfg['adversarial']['suspicious_auc']}** |
    | | **BLOCK threshold** | risk_score ≥ **{cfg['risk_score']['block_threshold']}** / 9 |
    """)
    return (cfg,)


@app.cell
def _(Path, cs, pl):
    _ROOT = Path(__file__).resolve().parent
    while not (_ROOT / "pixi.toml").exists():
        _ROOT = _ROOT.parent

    _train  = pl.read_parquet(_ROOT / "data/processed/train_features.parquet")
    _labels = pl.read_parquet(_ROOT / "data/processed/train_labels.parquet")

    df = _train.join(_labels.select(["customer_ID", "target"]), on="customer_ID")
    numeric_cols = df.select(cs.numeric().exclude("target")).columns

    # sample 20k สำหรับ Fast/Medium techniques (T1–T7)
    df_sample = df.sample(n=min(20_000, df.shape[0]), seed=42)
    return df, df_sample, numeric_cols


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🗂️ โครงสร้าง Data

    | Prefix | ประเภท | ตัวอย่าง |
    |--------|--------|---------|
    | **P_** | Payment | ยอดชำระ, % ที่ชำระ |
    | **D_** | Delinquency | วันค้างชำระ, สถานะบัญชี |
    | **S_** | Spend | ยอดใช้จ่าย |
    | **B_** | Balance | ยอดหนี้, วงเงิน |
    | **R_** | Risk | คะแนนความเสี่ยง |

    > ข้อมูลดิบเป็น time series รายเดือน → preprocessor aggregate เป็น `_mean, _min, _max, _last`
    >
    > ⚠️ `_last` features เสี่ยง leakage สูง — ลูกค้าที่ default อาจ "หยุด" ทำธุรกรรม
    > → `payment_last = 0` หรือ null เฉพาะ class default

    ### df vs df_sample
    - `df` = full data — ใช้ใน T5 PSI, T6 Time Stability (ต้องการ time order จริง)
    - `df_sample` = 20k rows — ใช้ใน T1–T4, T7–T9 (ต้องการความเร็ว)
    """)
    return


@app.cell
def _(df_sample, mo, numeric_cols, pl):
    _corr = (
        df_sample
        .select([pl.corr(c, "target").alias(c) for c in numeric_cols])
        .unpivot(variable_name="feature", value_name="corr")
        .with_columns(pl.col("corr").abs().alias("abs_corr"))
        .sort("abs_corr", descending=True)
    )

    red_flag_cols   = _corr.filter(pl.col("abs_corr") > 0.9)["feature"].to_list()
    suspicious_cols = _corr.filter(
        (pl.col("abs_corr") >= 0.5) & (pl.col("abs_corr") < 0.9)
    )["feature"].to_list()
    safe_cols_low   = _corr.filter(pl.col("abs_corr") < 0.5)["feature"].to_list()

    _red_set       = set(red_flag_cols)
    all_check_cols = [c for c in numeric_cols if c not in _red_set]

    mo.md(f"""
    ## Gate 0: Pre-screen ด้วย Correlation

    | กลุ่ม | จำนวน | สถานะ |
    |-------|-------|-------|
    | 🔴 Red Flag (corr > 0.9) | **{len(red_flag_cols)}** | ข้ามไป 9 เทคนิค — obvious leakage |
    | 🟠 Suspicious (0.5–0.9) | **{len(suspicious_cols)}** | เข้า 9 เทคนิค |
    | 🟡 Low corr (< 0.5) | **{len(safe_cols_low)}** | เข้า 9 เทคนิคด้วย! |
    | **✅ ตรวจ 9 เทคนิค** | **{len(all_check_cols)}** | |

    > Correlation วัด linear เท่านั้น — leakage ซ่อนใน non-linear pattern ได้
    > → ต้องตรวจทุก feature ที่เข้า model ด้วย 9 เทคนิคเสมอ
    """)
    return (all_check_cols,)


@app.cell
def _(all_check_cols, cfg, df_sample, mo, np, pl, plt):
    def _make_null():
        _df0 = df_sample.filter(pl.col("target") == 0)
        _df1 = df_sample.filter(pl.col("target") == 1)

        _null0 = np.array([_df0[c].is_null().mean() for c in all_check_cols])
        _null1 = np.array([_df1[c].is_null().mean() for c in all_check_cols])
        _diff  = np.abs(_null1 - _null0)

        null_scores    = dict(zip(all_check_cols, _diff))
        null_rate_0    = dict(zip(all_check_cols, _null0))
        null_rate_1    = dict(zip(all_check_cols, _null1))
        _thr_flag      = cfg["null_pattern"]["flagged_diff"]
        _thr_sus       = cfg["null_pattern"]["suspicious_diff"]
        null_flagged   = {k: v for k, v in null_scores.items() if v > _thr_flag}
        null_suspicious = {k: v for k, v in null_scores.items() if _thr_sus < v <= _thr_flag}

        fig1, ax = plt.subplots(figsize=(11, 4))
        ax.hist(_diff, bins=60, color="#6c8ebf", edgecolor="white")
        ax.axvline(_thr_flag, color="red",    linestyle="--", linewidth=1.5,
                   label=f"🔴 Flagged > {_thr_flag*100:.0f}%")
        ax.axvline(_thr_sus,  color="orange", linestyle="--", linewidth=1.5,
                   label=f"🟠 Suspicious > {_thr_sus*100:.0f}%")
        ax.set_xlabel("|Null Rate (class 1) − Null Rate (class 0)|")
        ax.set_ylabel("# Features")
        ax.set_title(
            f"T1 Null Pattern — {len(all_check_cols)} features\n"
            f"🔴 >{_thr_flag*100:.0f}%: {len(null_flagged)}   "
            f"🟠 {_thr_sus*100:.0f}–{_thr_flag*100:.0f}%: {len(null_suspicious)}",
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()

        _show = sorted(null_flagged.items(), key=lambda x: -x[1])[:12]
        if not _show:
            _show = sorted(null_suspicious.items(), key=lambda x: -x[1])[:12]

        if _show:
            N_COLS = 4
            N_ROWS = -(-len(_show) // N_COLS)
            fig2, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, N_ROWS * 3), squeeze=False)
            flat = axes.flatten()
            for i, (feat, diff) in enumerate(_show):
                ax2    = flat[i]
                _r0    = null_rate_0[feat]
                _r1    = null_rate_1[feat]
                _color = "#d62728" if diff > _thr_flag else "#ff7f0e"
                ax2.bar(["Non-default", "Default"], [_r0, _r1],
                        color=["#2196F3", "#F44336"], alpha=0.8, edgecolor="white")
                ax2.set_ylabel("Null Rate")
                ax2.set_ylim(0, max(_r0, _r1) * 1.3 + 0.01)
                ax2.set_title(f"{feat}\nΔ null = {diff:.1%}", color=_color,
                              fontsize=8, fontweight="bold")
                for j, v in enumerate([_r0, _r1]):
                    ax2.text(j, v + 0.005, f"{v:.1%}", ha="center", va="bottom", fontsize=8)
            for j in range(len(_show), len(flat)):
                flat[j].set_visible(False)
            plt.suptitle("T1 — Null Pattern Top Features", fontweight="bold")
            plt.tight_layout()
        else:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.text(0.5, 0.5, "ทุก feature มี null pattern สม่ำเสมอ ✅",
                     ha="center", va="center", fontsize=13, color="#2ca02c", fontweight="bold")
            ax2.axis("off")

        return fig1, fig2, null_scores, null_flagged, null_suspicious

    fig_null_hist, fig_null_bar, null_scores, null_flagged, null_suspicious = _make_null()

    mo.md(f"""
    ## T1: Null Pattern Analysis ⚡

    **เหตุผลที่ตรวจเป็นอันดับแรก:** ใน financial data นี่คือ leakage ที่พบบ่อยที่สุด
    และตรวจได้เร็วที่สุด — ไม่ต้องใช้ model เลย

    **หลักการ:** "การที่ข้อมูลหายไป" เองอาจเป็น signal
    - ลูกค้าที่ default → หยุดชำระ → payment records หลัง default = null
    - ถ้า model เรียนรู้ว่า "P_2 = null → default" = leakage (รู้ "อนาคต")

    | Δ Null Rate | ความหมาย |
    |------------|---------|
    | > {cfg['null_pattern']['flagged_diff']*100:.0f}% | 🔴 Flagged — missingness encode target |
    | {cfg['null_pattern']['suspicious_diff']*100:.0f}–{cfg['null_pattern']['flagged_diff']*100:.0f}% | 🟠 Suspicious |
    | < {cfg['null_pattern']['suspicious_diff']*100:.0f}% | ✅ Clean |

    พบ: 🔴 **{len(null_flagged)}** | 🟠 **{len(null_suspicious)}**
    """)
    return (
        fig_null_bar,
        fig_null_hist,
        null_flagged,
        null_scores,
        null_suspicious,
    )


@app.cell
def _(fig_null_hist):
    fig_null_hist
    return


@app.cell
def _(fig_null_bar):
    fig_null_bar
    return


@app.cell
def _(all_check_cols, cfg, df_sample, mo, np, pl, plt):
    def _make_sep():
        _arr_all = df_sample.select(all_check_cols).fill_null(0).to_numpy()
        _p1  = np.percentile(_arr_all, 1,  axis=0)
        _p99 = np.percentile(_arr_all, 99, axis=0)

        _mask0   = (df_sample["target"] == 0).to_numpy()
        _mask1   = (df_sample["target"] == 1).to_numpy()
        _clipped = np.clip(_arr_all, _p1, _p99)
        _arr0    = _clipped[_mask0]
        _arr1    = _clipped[_mask1]

        _m0, _m1 = _arr0.mean(axis=0), _arr1.mean(axis=0)
        _s0, _s1 = _arr0.std(axis=0),  _arr1.std(axis=0)
        pooled   = np.sqrt((_s0 ** 2 + _s1 ** 2) / 2) + 1e-8
        d_vals   = np.abs(_m1 - _m0) / pooled
        scores   = {c: float(d) for c, d in zip(all_check_cols, d_vals)}

        _thr_flag = cfg["class_separation"]["flagged_d"]
        _thr_wat  = cfg["class_separation"]["watchlist_d"]
        flagged   = {k: v for k, v in scores.items() if v > _thr_flag}
        watchlist = {k: v for k, v in scores.items() if _thr_wat < v <= _thr_flag}

        fig1, ax = plt.subplots(figsize=(11, 4))
        ax.hist(d_vals, bins=80, color="#6c8ebf", edgecolor="white")
        ax.axvline(_thr_wat,  color="orange", linestyle="--", linewidth=1.5,
                   label=f"🟠 Watchlist (d={_thr_wat})")
        ax.axvline(_thr_flag, color="red",    linestyle="--", linewidth=1.5,
                   label=f"🔴 Flagged (d={_thr_flag})")
        ax.set_xlabel("Cohen's d (Robust: clipped p1–p99)")
        ax.set_ylabel("# Features")
        ax.set_title(
            f"T2 Class Separation — {len(all_check_cols)} features\n"
            f"🔴 d>{_thr_flag}: {len(flagged)}   🟠 d {_thr_wat}–{_thr_flag}: {len(watchlist)}",
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()

        _top_red    = sorted(flagged.items(),   key=lambda x: -x[1])[:8]
        _top_orange = sorted(watchlist.items(), key=lambda x: -x[1])[:4]
        TOP = _top_red + _top_orange

        if not TOP:
            fig2, _ax = plt.subplots(figsize=(8, 3))
            _ax.text(0.5, 0.5, f"ทุก feature มี d < {_thr_wat} ✅",
                     ha="center", va="center", fontsize=13, color="#2ca02c", fontweight="bold")
            _ax.axis("off")
            plt.tight_layout()
        else:
            N_COLS = 4
            N_ROWS = -(-len(TOP) // N_COLS)
            fig2, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, N_ROWS * 3.2), squeeze=False)
            flat = axes.flatten()
            for i, (col, d) in enumerate(TOP):
                ax = flat[i]
                v0 = df_sample.filter(pl.col("target") == 0)[col].drop_nulls().to_numpy()
                v1 = df_sample.filter(pl.col("target") == 1)[col].drop_nulls().to_numpy()
                if len(v0) > 1 and len(v1) > 1:
                    lo, hi = np.percentile(np.concatenate([v0, v1]), [1, 99])
                    v0 = v0[(v0 >= lo) & (v0 <= hi)]
                    v1 = v1[(v1 >= lo) & (v1 <= hi)]
                ax.hist(v0, bins=50, density=True, alpha=0.55, color="#2196F3", label="Non-default")
                ax.hist(v1, bins=50, density=True, alpha=0.55, color="#F44336", label="Default")
                title_color = "#d62728" if d > _thr_flag else "#ff7f0e"
                badge       = "🔴" if d > _thr_flag else "🟠"
                ax.set_title(f"{col}\nd = {d:.2f} {badge}", color=title_color,
                             fontsize=8.5, fontweight="bold")
                ax.legend(fontsize=7)
            for j in range(len(TOP), len(flat)):
                flat[j].set_visible(False)
            plt.suptitle(f"T2 — KDE Flagged + Watchlist", fontweight="bold")
            plt.tight_layout()

        return fig1, fig2, scores, flagged, watchlist

    fig_sep_hist, fig_sep_kde, cohen_d_scores, flagged_sep, watchlist_sep = _make_sep()

    mo.md(f"""
    ## T2: Class Separation (Cohen's d) ⚡

    **หลักการ:** วัดว่า distribution ของ feature แยกกันระหว่าง Default / Non-default มากแค่ไหน

    สูตร: **d = |mean₁ − mean₀| / pooled_std** (Robust: clip p1–p99 ก่อน)

    > ทำไมต้อง clip? outlier inflate std → d ดูต่ำกว่าจริง

    | d | ความหมาย | จำนวน |
    |----|---------|-------|
    | < {cfg['class_separation']['watchlist_d']} | ✅ distribution ทับซ้อนปกติ | {sum(1 for v in cohen_d_scores.values() if v <= cfg['class_separation']['watchlist_d'])} |
    | {cfg['class_separation']['watchlist_d']}–{cfg['class_separation']['flagged_d']} | 🟠 Watchlist | {len(watchlist_sep)} |
    | > {cfg['class_separation']['flagged_d']} | 🔴 Flagged | {len(flagged_sep)} |

    พบ: 🔴 **{len(flagged_sep)}** | 🟠 **{len(watchlist_sep)}**
    """)
    return (
        cohen_d_scores,
        fig_sep_hist,
        fig_sep_kde,
        flagged_sep,
        watchlist_sep,
    )


@app.cell
def _(fig_sep_hist):
    fig_sep_hist
    return


@app.cell
def _(fig_sep_kde):
    fig_sep_kde
    return


@app.cell
def _(all_check_cols, cfg, df_sample, mo, mutual_info_classif, np, plt):
    def _make_mi():
        _y      = df_sample["target"].to_numpy()
        _X      = df_sample.select(all_check_cols).fill_null(0).to_numpy()
        _mi_arr = mutual_info_classif(_X, _y, discrete_features=False, random_state=42)
        mi_scores = dict(zip(all_check_cols, _mi_arr))

        _thr_flag = cfg["mutual_information"]["flagged"]
        _thr_sus  = cfg["mutual_information"]["suspicious"]
        mi_flagged    = {k: v for k, v in mi_scores.items() if v > _thr_flag}
        mi_suspicious = {k: v for k, v in mi_scores.items() if _thr_sus < v <= _thr_flag}

        _vals = np.array(list(mi_scores.values()))
        fig1, ax = plt.subplots(figsize=(11, 4))
        ax.hist(_vals, bins=60, color="#6c8ebf", edgecolor="white")
        ax.axvline(_thr_flag, color="red",    linestyle="--", linewidth=1.5,
                   label=f"🔴 Flagged > {_thr_flag}")
        ax.axvline(_thr_sus,  color="orange", linestyle="--", linewidth=1.5,
                   label=f"🟠 Suspicious > {_thr_sus}")
        ax.set_xlabel("Mutual Information (nats)")
        ax.set_ylabel("# Features")
        ax.set_title(
            f"T3 Mutual Information — {len(all_check_cols)} features\n"
            f"🔴 >{_thr_flag}: {len(mi_flagged)}   🟠 {_thr_sus}–{_thr_flag}: {len(mi_suspicious)}",
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()

        _top25 = sorted(mi_scores.items(), key=lambda x: -x[1])[:25]
        _feats, _mis = zip(*_top25)
        _colors = ["#d62728" if v > _thr_flag else "#ff7f0e" if v > _thr_sus else "#4C72B0"
                   for v in _mis]
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        ax2.barh(list(_feats)[::-1], list(_mis)[::-1], color=list(_colors)[::-1])
        ax2.axvline(_thr_flag, color="red",    linestyle="--", linewidth=1.2)
        ax2.axvline(_thr_sus,  color="orange", linestyle="--", linewidth=1.2)
        ax2.set_xlabel("Mutual Information (nats)")
        ax2.set_title("T3 — Top 25 by Mutual Information", fontweight="bold")
        plt.tight_layout()
        return fig1, fig2, mi_scores, mi_flagged, mi_suspicious

    fig_mi_hist, fig_mi_bar, mi_scores, mi_flagged, mi_suspicious = _make_mi()

    mo.md(f"""
    ## T3: Mutual Information ⚡

    **หลักการ:** วัด "ข้อมูลที่ feature บอกเกี่ยวกับ target" โดยไม่สมมติ linearity

    | | Correlation (T gate) | MI (T3) |
    |-|---------------------|---------|
    | Linear | ✅ | ✅ |
    | Non-linear | ❌ พลาด | ✅ จับได้ |
    | ตัวอย่าง | X² vs X → corr=0 | X² vs X → MI > 0 |

    **Pattern ที่ MI จับได้แต่ Correlation พลาด:**
    - feature = 0 เสมอสำหรับ default, random สำหรับ non-default
    - feature ที่ถูก binned หรือ one-hot encode

    | MI | ความหมาย |
    |----|---------|
    | > {cfg['mutual_information']['flagged']} | 🔴 Flagged |
    | {cfg['mutual_information']['suspicious']}–{cfg['mutual_information']['flagged']} | 🟠 Suspicious |
    | < {cfg['mutual_information']['suspicious']} | ✅ Clean |

    พบ: 🔴 **{len(mi_flagged)}** | 🟠 **{len(mi_suspicious)}**
    """)
    return fig_mi_bar, fig_mi_hist, mi_flagged, mi_scores, mi_suspicious


@app.cell
def _(fig_mi_hist):
    fig_mi_hist
    return


@app.cell
def _(fig_mi_bar):
    fig_mi_bar
    return


@app.cell
def _(all_check_cols, cfg, df_sample, mo, np, plt):
    def _make_var_ratio():
        # Reuse clipped arrays pattern from T2 — Cohen's d วัด mean diff
        # Variance Ratio วัด spread diff — จับ leakage ที่ T2 พลาด
        #
        # ตัวอย่าง: payment_last สำหรับ default = 0 เสมอ → std ≈ 0 (collapsed)
        #           payment_last สำหรับ non-default = หลากหลาย → std สูง
        # Cohen's d: mean diff เล็กน้อย → อาจไม่ flag
        # Variance Ratio: std₀/std₁ สูงมาก → flag ทันที
        _arr_all = df_sample.select(all_check_cols).fill_null(0).to_numpy()
        _p1      = np.percentile(_arr_all, 1,  axis=0)
        _p99     = np.percentile(_arr_all, 99, axis=0)
        _clipped = np.clip(_arr_all, _p1, _p99)

        _mask0 = (df_sample["target"] == 0).to_numpy()
        _mask1 = (df_sample["target"] == 1).to_numpy()
        _arr0, _arr1 = _clipped[_mask0], _clipped[_mask1]

        _std0  = _arr0.std(axis=0) + 1e-8
        _std1  = _arr1.std(axis=0) + 1e-8
        # max(r, 1/r) → ≥ 1 เสมอ — ไม่สนใจว่า class ไหน concentrated กว่า
        _ratio = np.maximum(_std1 / _std0, _std0 / _std1)

        var_scores = dict(zip(all_check_cols, _ratio))
        _thr_flag  = cfg["variance_ratio"]["flagged"]
        _thr_sus   = cfg["variance_ratio"]["suspicious"]
        var_flagged    = {k: v for k, v in var_scores.items() if v > _thr_flag}
        var_suspicious = {k: v for k, v in var_scores.items() if _thr_sus < v <= _thr_flag}

        fig1, ax = plt.subplots(figsize=(11, 4))
        ax.hist(list(_ratio), bins=60, color="#6c8ebf", edgecolor="white")
        ax.axvline(_thr_flag, color="red",    linestyle="--", linewidth=1.5,
                   label=f"🔴 Flagged > {_thr_flag}x")
        ax.axvline(_thr_sus,  color="orange", linestyle="--", linewidth=1.2,
                   label=f"🟠 Suspicious > {_thr_sus}x")
        ax.set_xlabel("Variance Ratio: max(std₁/std₀, std₀/std₁)")
        ax.set_ylabel("# Features")
        ax.set_title(
            f"T4 Variance Ratio — {len(all_check_cols)} features\n"
            f"🔴 >{_thr_flag}x: {len(var_flagged)}   🟠 {_thr_sus}x–{_thr_flag}x: {len(var_suspicious)}",
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()

        _show = sorted(var_flagged.items(), key=lambda x: -x[1])[:12]
        if not _show:
            _show = sorted(var_suspicious.items(), key=lambda x: -x[1])[:12]

        if _show:
            N_COLS = 4
            N_ROWS = -(-len(_show) // N_COLS)
            fig2, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, N_ROWS * 3), squeeze=False)
            flat = axes.flatten()
            for i, (feat, ratio) in enumerate(_show):
                ax2   = flat[i]
                _fidx = all_check_cols.index(feat)
                s0, s1 = float(_std0[_fidx]), float(_std1[_fidx])
                _color  = "#d62728" if ratio > _thr_flag else "#ff7f0e"
                ax2.bar(["Non-default", "Default"], [s0, s1],
                        color=["#2196F3", "#F44336"], alpha=0.8, edgecolor="white")
                ax2.set_ylabel("Std Dev (clipped)")
                ax2.set_title(f"{feat}\nratio = {ratio:.2f}x", color=_color,
                              fontsize=8, fontweight="bold")
                for j, v in enumerate([s0, s1]):
                    ax2.text(j, v * 1.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            for j in range(len(_show), len(flat)):
                flat[j].set_visible(False)
            plt.suptitle("T4 — Variance Ratio Top Features", fontweight="bold")
            plt.tight_layout()
        else:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.text(0.5, 0.5, "ทุก feature มี variance ratio ต่ำกว่า threshold ✅",
                     ha="center", va="center", fontsize=13, color="#2ca02c", fontweight="bold")
            ax2.axis("off")

        return fig1, fig2, var_scores, var_flagged, var_suspicious

    fig_var_hist, fig_var_bar, var_scores, var_flagged, var_suspicious = _make_var_ratio()

    mo.md(f"""
    ## T4: Variance Ratio ⚡

    **หลักการ:** วัด spread ของ feature ต่างกันระหว่าง class ไหม?

    สูตร: **ratio = max(std₁/std₀, std₀/std₁)** — ≥ 1 เสมอ

    **ทำไม Variance Ratio จับ leakage ที่ Cohen's d (T2) พลาด?**

    | กรณี | Cohen's d | Variance Ratio |
    |------|-----------|---------------|
    | Mean ต่าง, Spread เท่ากัน | ✅ จับ | ✅ |
    | Mean ใกล้กัน แต่ class=default "collapse" ที่ค่าเดียว | ❌ พลาด | ✅ ratio สูง |
    | `_last = 0` เสมอหลัง default | ❌ อาจพลาด | ✅ จับทันที |

    | Ratio | ความหมาย |
    |-------|---------|
    | > {cfg['variance_ratio']['flagged']}x | 🔴 Flagged |
    | {cfg['variance_ratio']['suspicious']}x–{cfg['variance_ratio']['flagged']}x | 🟠 Suspicious |
    | < {cfg['variance_ratio']['suspicious']}x | ✅ Clean |

    พบ: 🔴 **{len(var_flagged)}** | 🟠 **{len(var_suspicious)}**
    """)
    return fig_var_bar, fig_var_hist, var_flagged, var_scores, var_suspicious


@app.cell
def _(fig_var_hist):
    fig_var_hist
    return


@app.cell
def _(fig_var_bar):
    fig_var_bar
    return


@app.cell
def _(all_check_cols, cfg, df, mo, np, plt):
    def _make_psi():
        # PSI วัด marginal distribution stability — ไม่สนใจ class
        # "distribution ของ feature เองเปลี่ยนระหว่างช่วงต้น vs ช่วงหลังไหม?"
        # ต่างจาก T6 Time Stability ตรงที่:
        # T5 PSI: วัด feature distribution (marginal) — banking standard
        # T6 Time Stability: วัด gap ระหว่าง class ตามเวลา (class-conditional)
        _n    = df.shape[0]
        _half = _n // 2
        _df0  = df.head(_half)         # ช่วงแรก = expected
        _df1  = df.tail(_n - _half)    # ช่วงหลัง = actual
        N_BKT = cfg["psi"]["buckets"]

        def _psi_one(col):
            _exp = _df0[col].drop_nulls().to_numpy()
            _act = _df1[col].drop_nulls().to_numpy()
            if len(_exp) < 10 or len(_act) < 10:
                return 0.0
            _breaks = np.unique(np.percentile(_exp, np.linspace(0, 100, N_BKT + 1)))
            if len(_breaks) < 3:
                return 0.0
            _e_cnt  = np.histogram(_exp, bins=_breaks)[0].astype(float)
            _a_cnt  = np.histogram(_act, bins=_breaks)[0].astype(float)
            _e_pct  = np.clip(_e_cnt / _e_cnt.sum(), 1e-6, None)
            _a_pct  = np.clip(_a_cnt / _a_cnt.sum(), 1e-6, None)
            return float(np.sum((_a_pct - _e_pct) * np.log(_a_pct / _e_pct)))

        psi_scores    = {c: _psi_one(c) for c in all_check_cols}
        _vals         = np.array(list(psi_scores.values()))
        _thr_flag     = cfg["psi"]["unstable"]
        _thr_sus      = cfg["psi"]["monitor"]
        psi_flagged   = {k: v for k, v in psi_scores.items() if v > _thr_flag}
        psi_suspicious = {k: v for k, v in psi_scores.items() if _thr_sus < v <= _thr_flag}

        fig1, ax = plt.subplots(figsize=(11, 4))
        ax.hist(_vals, bins=60, color="#6c8ebf", edgecolor="white")
        ax.axvline(_thr_flag, color="red",    linestyle="--", linewidth=1.5,
                   label=f"🔴 Unstable > {_thr_flag}")
        ax.axvline(_thr_sus,  color="orange", linestyle="--", linewidth=1.5,
                   label=f"🟠 Monitor > {_thr_sus}")
        ax.set_xlabel("PSI (Population Stability Index)")
        ax.set_ylabel("# Features")
        ax.set_title(
            f"T5 PSI — {len(all_check_cols)} features (Q1-Q2 vs Q3-Q4)\n"
            f"🔴 >{_thr_flag}: {len(psi_flagged)}   🟠 {_thr_sus}–{_thr_flag}: {len(psi_suspicious)}",
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()

        _show = sorted(psi_flagged.items(), key=lambda x: -x[1])[:8]
        if not _show:
            _show = sorted(psi_suspicious.items(), key=lambda x: -x[1])[:8]

        if _show:
            N_COLS = 4
            N_ROWS = -(-len(_show) // N_COLS)
            fig2, axes = plt.subplots(N_ROWS, N_COLS, figsize=(16, N_ROWS * 3.5), squeeze=False)
            flat = axes.flatten()
            for i, (feat, psi_val) in enumerate(_show):
                ax2  = flat[i]
                _exp = _df0[feat].drop_nulls().to_numpy()
                _act = _df1[feat].drop_nulls().to_numpy()
                _lo  = np.percentile(np.concatenate([_exp, _act]), 1)
                _hi  = np.percentile(np.concatenate([_exp, _act]), 99)
                _exp = _exp[(_exp >= _lo) & (_exp <= _hi)]
                _act = _act[(_act >= _lo) & (_act <= _hi)]
                ax2.hist(_exp, bins=40, density=True, alpha=0.55,
                         color="#2196F3", label="Q1-Q2 (expected)")
                ax2.hist(_act, bins=40, density=True, alpha=0.55,
                         color="#F44336", label="Q3-Q4 (actual)")
                _color = "#d62728" if psi_val > _thr_flag else "#ff7f0e"
                ax2.set_title(f"{feat}\nPSI = {psi_val:.3f}", color=_color,
                              fontsize=8, fontweight="bold")
                ax2.legend(fontsize=6)
            for j in range(len(_show), len(flat)):
                flat[j].set_visible(False)
            plt.suptitle("T5 PSI — Top Features ที่ Distribution เปลี่ยนตามเวลา",
                         fontweight="bold")
            plt.tight_layout()
        else:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            ax2.text(0.5, 0.5, "ทุก feature มี PSI < 0.10 ✅ — distributions เสถียร",
                     ha="center", va="center", fontsize=13, color="#2ca02c", fontweight="bold")
            ax2.axis("off")

        return fig1, fig2, psi_scores, psi_flagged, psi_suspicious

    fig_psi_hist, fig_psi_bar, psi_scores, psi_flagged, psi_suspicious = _make_psi()

    mo.md(f"""
    ## T5: PSI (Population Stability Index) 🕐 — มาตรฐาน Banking

    **หลักการ:** วัดว่า distribution ของ feature (marginal, ไม่สนใจ class) เปลี่ยนระหว่างช่วงเวลา

    สูตร: **PSI = Σ (Actual% − Expected%) × ln(Actual% / Expected%)**

    > ใช้ full data แบ่ง Q1-Q2 = expected, Q3-Q4 = actual

    | PSI | ความหมาย | Action |
    |-----|---------|--------|
    | < {cfg['psi']['monitor']} | ✅ Stable | ใช้ได้ปกติ |
    | {cfg['psi']['monitor']}–{cfg['psi']['unstable']} | 🟠 Monitor | ติดตามใน production |
    | > {cfg['psi']['unstable']} | 🔴 Unstable | ตรวจ/rebuild feature |

    พบ: 🔴 **{len(psi_flagged)}** unstable | 🟠 **{len(psi_suspicious)}** monitor
    """)
    return fig_psi_bar, fig_psi_hist, psi_flagged, psi_scores, psi_suspicious


@app.cell
def _(fig_psi_hist):
    fig_psi_hist
    return


@app.cell
def _(fig_psi_bar):
    fig_psi_bar
    return


@app.cell
def _(all_check_cols, cfg, df, mo, np, pl, plt):
    def _make_time():
        n = df.shape[0]

        _date_col = None
        for _c in df.columns:
            if df[_c].dtype in (pl.Date, pl.Datetime):
                _date_col = _c
                break

        if _date_col:
            _df     = df.with_columns(pl.col(_date_col).dt.strftime("%Y-%m").alias("__time_bin"))
            x_label = f"Month ({_date_col})"
        else:
            _df = df.with_row_index("__ridx").with_columns(
                pl.when(pl.col("__ridx") < n // 4).then(pl.lit("Q1"))
                .when(pl.col("__ridx") < n // 2).then(pl.lit("Q2"))
                .when(pl.col("__ridx") < 3 * n // 4).then(pl.lit("Q3"))
                .otherwise(pl.lit("Q4"))
                .alias("__time_bin")
            )
            x_label = "Quartile (Row Order)"

        _agg = (
            _df.group_by(["__time_bin", "target"])
            .agg([pl.col(c).mean().alias(c) for c in all_check_cols])
            .sort(["__time_bin", "target"])
        )
        _t0   = _agg.filter(pl.col("target") == 0)
        _t1   = _agg.filter(pl.col("target") == 1)
        _bins = _t0["__time_bin"].to_list()

        unstable  = []
        gap_cvs   = {}
        _thr_cv   = cfg["time_stability"]["unstable_gap_cv"]
        for col in all_check_cols:
            t0_means = _t0[col].fill_null(0).to_list()
            t1_means = _t1[col].fill_null(0).to_list()
            gaps     = [abs(a - b) for a, b in zip(t0_means, t1_means)]
            gap_cv   = float(np.std(gaps) / (np.mean(gaps) + 1e-8))
            gap_cvs[col] = gap_cv
            if gap_cv > _thr_cv:
                unstable.append(col)

        TOP  = sorted(unstable, key=lambda c: -gap_cvs[c])[:6]
        SHOW = max(1, len(TOP))
        N_C  = min(3, SHOW)
        N_R  = -(-SHOW // N_C)

        fig, axes = plt.subplots(N_R, N_C, figsize=(15, N_R * 3.5))
        flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
        x_pos = list(range(len(_bins)))

        for i, col in enumerate(TOP):
            ax       = flat[i]
            t0_means = _t0[col].fill_null(0).to_list()
            t1_means = _t1[col].fill_null(0).to_list()
            ax.plot(x_pos, t0_means, "o-", color="#2196F3", label="Non-default", linewidth=2)
            ax.plot(x_pos, t1_means, "s-", color="#F44336", label="Default",     linewidth=2)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(_bins, rotation=30, fontsize=8)
            ax.set_title(f"{col}\ngap_cv = {gap_cvs[col]:.2f} 🔴", fontsize=8.5,
                         fontweight="bold", color="#d62728")
            ax.set_xlabel(x_label, fontsize=7)
            ax.set_ylabel("Mean Value", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.set_facecolor("#fff0f0")

        for j in range(SHOW, len(flat)):
            flat[j].set_visible(False)

        _title = (f"T6 Time Stability — Top {len(TOP)} Unstable จาก {len(all_check_cols)}"
                  if TOP else f"T6 — ทุก {len(all_check_cols)} features เสถียร ✅")
        plt.suptitle(_title, fontweight="bold")
        plt.tight_layout()
        return fig, unstable, gap_cvs

    fig_time, unstable_cols, gap_cvs = _make_time()

    mo.md(f"""
    ## T6: Time Stability (Class-Conditional) 🕐

    **ต่างจาก T5 PSI ตรงไหน?**

    | | T5 PSI | T6 Time Stability |
    |-|--------|------------------|
    | วัดอะไร | marginal distribution ของ feature | gap ระหว่าง class ตามเวลา |
    | คำถาม | "feature เองเปลี่ยนไหม?" | "ความสัมพันธ์กับ target เปลี่ยนไหม?" |
    | ตัวอย่าง flag | feature ทุก class drift | gap ระหว่าง default/non-default กระโดด |

    **metric:** gap_cv = std(gaps over time) / mean(gaps)
    - gap = |mean(default) − mean(non-default)| ต่อช่วงเวลา
    - gap_cv สูง = ความสัมพันธ์ไม่เสถียร = model อาจเรียนรู้ pattern ชั่วคราว

    พบ: 🔴 **{len(unstable_cols)}** unstable (gap_cv > {cfg['time_stability']['unstable_gap_cv']})
    """)
    return fig_time, unstable_cols


@app.cell
def _(fig_time):
    fig_time
    return


@app.cell
def _(
    StratifiedKFold,
    all_check_cols,
    cfg,
    df_sample,
    lgb,
    mo,
    np,
    plt,
    roc_auc_score,
):
    def _run_sf_auc():
        _y = df_sample["target"].to_numpy()
        _X = df_sample.select(all_check_cols).fill_null(0).to_numpy()

        # Pass 1: Raw AUC (vectorized) — เทียบเท่า Wilcoxon rank-sum test
        # จับ monotone relationship ได้ทุกรูปแบบ — เร็วมาก
        # max(auc, 1-auc): leakage อาจเป็น negative correlation ก็ได้
        _raw = np.array([roc_auc_score(_y, _X[:, i]) for i in range(_X.shape[1])])
        _raw = np.maximum(_raw, 1 - _raw)
        raw_auc_scores = dict(zip(all_check_cols, _raw))

        # Pass 2: LGB — เฉพาะ raw_auc > screen_at (จับ non-linear/non-monotone)
        _screen   = cfg["single_feat_auc"]["screen_at"]
        _sus_idx  = [i for i, v in enumerate(_raw) if v > _screen]
        _params   = {
            "objective": "binary", "metric": "auc",
            "num_leaves": 7, "max_depth": 3,
            "learning_rate": 0.1, "min_child_samples": 50,
            "verbosity": -1, "random_state": 42,
        }
        _skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        lgb_auc_scores = {}
        for _i in _sus_idx:
            _feat = all_check_cols[_i]
            _x    = _X[:, _i : _i + 1]
            _oof  = np.zeros(len(_y))
            for _tr, _val in _skf.split(_x, _y):
                _dtrain = lgb.Dataset(_x[_tr], label=_y[_tr])
                _m = lgb.train(
                    _params, _dtrain, num_boost_round=100,
                    callbacks=[lgb.log_evaluation(False)],
                )
                _oof[_val] = _m.predict(_x[_val])
            lgb_auc_scores[_feat] = roc_auc_score(_y, _oof)

        _thr_flag = cfg["single_feat_auc"]["flagged"]
        _thr_sus  = cfg["single_feat_auc"]["suspicious"]
        _thr_wat  = cfg["single_feat_auc"]["watch"]
        sf_scores     = {f: max(raw_auc_scores[f], lgb_auc_scores.get(f, 0)) for f in all_check_cols}
        sf_flagged    = {k: v for k, v in sf_scores.items() if v > _thr_flag}
        sf_suspicious = {k: v for k, v in sf_scores.items() if _thr_sus < v <= _thr_flag}
        sf_watch      = {k: v for k, v in sf_scores.items() if _thr_wat < v <= _thr_sus}
        return raw_auc_scores, lgb_auc_scores, sf_scores, sf_flagged, sf_suspicious, sf_watch

    (raw_auc_scores, lgb_auc_scores,
     sf_scores, sf_flagged, sf_suspicious, sf_watch) = _run_sf_auc()

    def _plot_sf():
        _thr_flag = cfg["single_feat_auc"]["flagged"]
        _thr_sus  = cfg["single_feat_auc"]["suspicious"]
        _thr_wat  = cfg["single_feat_auc"]["watch"]
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))

        _vals = list(sf_scores.values())
        axes[0].hist(_vals, bins=60, color="#6c8ebf", edgecolor="white")
        axes[0].axvline(_thr_flag, color="red",    linestyle="--", linewidth=1.5,
                        label=f"🔴 Flagged  > {_thr_flag}")
        axes[0].axvline(_thr_sus,  color="orange", linestyle="--", linewidth=1.5,
                        label=f"🟠 Suspicious > {_thr_sus}")
        axes[0].axvline(_thr_wat,  color="gold",   linestyle="--", linewidth=1.5,
                        label=f"🟡 Watch > {_thr_wat}")
        axes[0].set_xlabel("Single-Feature AUC (max raw & LGB)")
        axes[0].set_ylabel("# Features")
        axes[0].set_title(
            f"T7 Single-Feature AUC — {len(all_check_cols)} features\n"
            f"🔴 {len(sf_flagged)}  🟠 {len(sf_suspicious)}  🟡 {len(sf_watch)}",
            fontweight="bold",
        )
        axes[0].legend()

        _top25 = sorted(sf_scores.items(), key=lambda x: -x[1])[:25]
        _feats, _aucs = zip(*_top25)
        _colors = ["#d62728" if v > _thr_flag else "#ff7f0e" if v > _thr_sus else "#FFD700"
                   for v in _aucs]
        axes[1].barh(list(_feats)[::-1], list(_aucs)[::-1], color=list(_colors)[::-1])
        axes[1].axvline(_thr_flag, color="red",    linestyle="--", linewidth=1.2)
        axes[1].axvline(_thr_sus,  color="orange", linestyle="--", linewidth=1.2)
        axes[1].axvline(0.5,       color="gray",   linestyle=":",  linewidth=1.0, label="Random (0.5)")
        axes[1].set_xlabel("Single-Feature AUC")
        axes[1].set_title("Top 25 Features by AUC", fontweight="bold")
        axes[1].legend(fontsize=8)

        plt.suptitle("T7: Single-Feature AUC — Production-Grade Leakage Screen",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        return fig

    fig_sf = _plot_sf()

    mo.md(f"""
    ## T7: Single-Feature AUC 🐢 — Most Direct Leakage Test

    **หลักการ (2 Pass) — ทำไมทำ 2 รอบ?**

    | Pass | วิธี | จับอะไร | Speed |
    |------|------|---------|-------|
    | **Pass 1: Raw AUC** | ใช้ feature value โดยตรง | Monotone relationship | ⚡ เร็วมาก |
    | **Pass 2: LGB AUC** | Train LGB 1 feature, 3-fold CV | Non-linear/U-shape | 🐢 เฉพาะ raw > {cfg['single_feat_auc']['screen_at']} |

    Raw AUC พลาด U-shape (feature = 0 สำหรับ class 1 แต่กระจายสำหรับ class 0)
    → LGB จับได้ทุก pattern

    | AUC | ความหมาย |
    |-----|---------|
    | > {cfg['single_feat_auc']['flagged']} | 🔴 Flagged |
    | {cfg['single_feat_auc']['suspicious']}–{cfg['single_feat_auc']['flagged']} | 🟠 Suspicious |
    | {cfg['single_feat_auc']['watch']}–{cfg['single_feat_auc']['suspicious']} | 🟡 Watch |
    | < {cfg['single_feat_auc']['watch']} | ✅ Clean |

    พบ: 🔴 **{len(sf_flagged)}** | 🟠 **{len(sf_suspicious)}** | 🟡 **{len(sf_watch)}**
    (LGB ยืนยันบน {sum(1 for v in raw_auc_scores.values() if v > cfg['single_feat_auc']['screen_at'])} candidates)
    """)
    return fig_sf, sf_flagged, sf_scores, sf_suspicious


@app.cell
def _(fig_sf):
    fig_sf
    return


@app.cell
def _(
    StratifiedKFold,
    cfg,
    df_sample,
    lgb,
    mo,
    np,
    numeric_cols,
    plt,
    roc_auc_score,
):
    def _run_adv():
        n      = df_sample.shape[0]
        _adv_y = np.zeros(n, dtype=int)
        _adv_y[n // 2:] = 1
        _X = df_sample.select(numeric_cols).to_numpy()

        _params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.05, "num_leaves": 31,
            "verbosity": -1, "random_state": 42,
        }
        _skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        _oof = np.zeros(n)
        _imp = np.zeros(len(numeric_cols))

        for _tr, _val in _skf.split(_X, _adv_y):
            _dtrain = lgb.Dataset(_X[_tr], label=_adv_y[_tr])
            _dval   = lgb.Dataset(_X[_val], label=_adv_y[_val], reference=_dtrain)
            _m = lgb.train(
                _params, _dtrain, num_boost_round=300, valid_sets=[_dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(False)],
            )
            _oof[_val] = _m.predict(_X[_val])
            _imp      += _m.feature_importance(importance_type="gain")

        _auc     = roc_auc_score(_adv_y, _oof)
        _top_idx = np.argsort(_imp)[::-1][:20]
        return _auc, [(numeric_cols[i], float(_imp[i])) for i in _top_idx]

    adv_auc, adv_top_features = _run_adv()
    adv_ok = adv_auc < cfg["adversarial"]["suspicious_auc"]

    def _plot_adv():
        feats, imps = zip(*adv_top_features)
        colors = ["#d62728" if i < 5 else "#ff7f0e" if i < 10 else "#4C72B0"
                  for i in range(len(feats))]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(feats)[::-1], list(imps)[::-1], color=list(colors)[::-1])
        ax.set_xlabel("Feature Importance (Gain) — ยิ่งสูง ยิ่ง non-stationary")
        ax.set_title(
            f"T8 Adversarial (Train/Train) — AUC = {adv_auc:.4f}\n"
            f"{'✅ AUC < 0.6 → data stationary' if adv_ok else '🔴 AUC ≥ 0.6 → มี feature ที่ไม่เสถียรตามเวลา'}",
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    fig_adv = _plot_adv()

    mo.md(f"""
    ## T8: Adversarial Validation (Train/Train) 🔥

    **หลักการ:** แบ่ง training data เป็น "ครึ่งแรก" vs "ครึ่งหลัง"
    แล้วสอน model ให้ทายว่า row นี้มาจากครึ่งไหน

    **ต่างจาก T5 PSI / T6 Time Stability ตรงไหน?**
    - T5, T6: ตรวจ feature ทีละตัว
    - T8: ใช้ **LGB บน feature ทั้งหมดพร้อมกัน** — จับ interaction ระหว่าง feature ที่ทำให้ data ไม่ stationary

    | AUC | ความหมาย |
    |-----|---------|
    | ≈ 0.5 | ✅ data stationary — ปลอดภัย |
    | 0.6–0.7 | 🟠 มี feature บางตัวไม่เสถียร |
    | > 0.7 | 🔴 non-stationary รุนแรง |

    **ผลลัพธ์: AUC = {adv_auc:.4f}** → {"✅ stationary" if adv_ok else "🔴 ไม่เสถียร"}
    """)
    return adv_auc, adv_ok, adv_top_features, fig_adv


@app.cell
def _(fig_adv):
    fig_adv
    return


@app.cell
def _(
    Path,
    StratifiedKFold,
    cfg,
    df_sample,
    lgb,
    mo,
    np,
    numeric_cols,
    pl,
    plt,
    roc_auc_score,
):
    def _run_tt_adv():
        _ROOT = Path(__file__).resolve().parent
        while not (_ROOT / "pixi.toml").exists():
            _ROOT = _ROOT.parent
        _test_path = _ROOT / "data/processed/test_features.parquet"
        if not _test_path.exists():
            return None, []

        _test        = pl.read_parquet(_test_path)
        _test_sample = _test.sample(n=min(20_000, _test.shape[0]), seed=42)
        _X_train     = df_sample.select(numeric_cols).to_numpy()
        _X_test      = _test_sample.select(numeric_cols).to_numpy()
        _X           = np.vstack([_X_train, _X_test])
        _is_test     = np.concatenate([
            np.zeros(len(_X_train), dtype=int),
            np.ones(len(_X_test),  dtype=int),
        ])

        _params = {
            "objective": "binary", "metric": "auc",
            "learning_rate": 0.05, "num_leaves": 31,
            "verbosity": -1, "random_state": 42,
        }
        _skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        _oof = np.zeros(len(_X))
        _imp = np.zeros(len(numeric_cols))

        for _tr, _val in _skf.split(_X, _is_test):
            _dtrain = lgb.Dataset(_X[_tr], label=_is_test[_tr])
            _dval   = lgb.Dataset(_X[_val], label=_is_test[_val], reference=_dtrain)
            _m = lgb.train(
                _params, _dtrain, num_boost_round=300, valid_sets=[_dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(False)],
            )
            _oof[_val] = _m.predict(_X[_val])
            _imp      += _m.feature_importance(importance_type="gain")

        _auc     = roc_auc_score(_is_test, _oof)
        _top_idx = np.argsort(_imp)[::-1][:20]
        return _auc, [(numeric_cols[i], float(_imp[i])) for i in _top_idx]

    adv_tt_auc, adv_tt_top_features = _run_tt_adv()
    adv_tt_ok = (adv_tt_auc < cfg["adversarial"]["suspicious_auc"]) if adv_tt_auc is not None else None

    def _plot_tt():
        if adv_tt_auc is None:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.text(0.5, 0.5, "ไม่พบ test_features.parquet\nข้ามเทคนิคนี้",
                    ha="center", va="center", fontsize=13)
            ax.axis("off")
            return fig
        feats, imps = zip(*adv_tt_top_features)
        colors = ["#d62728" if i < 5 else "#ff7f0e" if i < 10 else "#4C72B0"
                  for i in range(len(feats))]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(feats)[::-1], list(imps)[::-1], color=list(colors)[::-1])
        ax.set_xlabel("Feature Importance (Gain) — ยิ่งสูง ยิ่ง drift")
        _st = "✅ No drift" if adv_tt_ok else "🔴 Drift detected"
        ax.set_title(f"T9 Adversarial (Train/Test) — AUC = {adv_tt_auc:.4f}\n{_st}",
                     fontweight="bold")
        plt.tight_layout()
        return fig

    fig_adv_tt = _plot_tt()

    if adv_tt_auc is None:
        _md = "## T9: Adversarial (Train/Test) 🔥\n\n⚠️ ไม่พบ `test_features.parquet` — ข้ามเทคนิคนี้"
    else:
        _md = f"""
    ## T9: Adversarial Validation (Train/Test) 🔥

    **หลักการ:** เอา train + test รวมกัน สร้าง label `is_test` แล้วสอน model ให้แยก
    ถ้าแยกได้ = มี feature ที่ "drift" ระหว่างสองชุด = model จะพังเมื่อ submit จริง

    | AUC | ความหมาย |
    |-----|---------|
    | ≈ 0.5 | ✅ Train/Test มาจาก distribution เดียวกัน |
    | > {cfg['adversarial']['suspicious_auc']} | 🔴 Drift รุนแรง — feature สีแดงต้องตรวจ |

    **ผลลัพธ์: AUC = {adv_tt_auc:.4f}** → {"✅ distributions similar" if adv_tt_ok else "🔴 drift detected"}
    """

    mo.md(_md)
    return adv_tt_auc, adv_tt_ok, adv_tt_top_features, fig_adv_tt


@app.cell
def _(fig_adv_tt):
    fig_adv_tt
    return


@app.cell
def _(
    adv_auc,
    adv_ok,
    adv_top_features,
    adv_tt_auc,
    adv_tt_ok,
    adv_tt_top_features,
    all_check_cols,
    cfg,
    flagged_sep,
    mi_flagged,
    mi_suspicious,
    mo,
    null_flagged,
    null_suspicious,
    psi_flagged,
    psi_suspicious,
    sf_flagged,
    sf_suspicious,
    unstable_cols,
    var_flagged,
    var_suspicious,
    watchlist_sep,
):
    _adv_flagged    = {f for f, _ in adv_top_features[:5]}
    _adv_tt_flagged = {f for f, _ in adv_tt_top_features[:5]}
    _null_set       = set(null_flagged)    | set(null_suspicious)
    _sep_set        = set(flagged_sep)
    _mi_set         = set(mi_flagged)      | set(mi_suspicious)
    _var_set        = set(var_flagged)     | set(var_suspicious)
    _psi_set        = set(psi_flagged)     | set(psi_suspicious)
    _time_set       = set(unstable_cols)
    _sf_set         = set(sf_flagged)      | set(sf_suspicious)

    _all_flagged = sorted(
        _null_set | _sep_set | _mi_set | _var_set | _psi_set |
        _time_set | _sf_set | _adv_flagged | _adv_tt_flagged
    )

    _BLOCK = cfg["risk_score"]["block_threshold"]
    _rows = []
    for _f in _all_flagged:
        _f1 = "🔴" if _f in _null_set     else "✅"
        _f2 = "🔴" if _f in _sep_set      else "✅"
        _f3 = "🔴" if _f in _mi_set       else "✅"
        _f4 = "🔴" if _f in _var_set      else "✅"
        _f5 = "🔴" if _f in _psi_set      else "✅"
        _f6 = "🔴" if _f in _time_set     else "✅"
        _f7 = "🔴" if _f in _sf_set       else "✅"
        _f8 = "🔴" if _f in _adv_flagged  else "✅"
        _f9 = "🔴" if _f in _adv_tt_flagged else "✅"
        _n  = sum([
            _f in _null_set, _f in _sep_set, _f in _mi_set, _f in _var_set,
            _f in _psi_set, _f in _time_set, _f in _sf_set,
            _f in _adv_flagged, _f in _adv_tt_flagged,
        ])
        _verdict = "🚨" if _n >= _BLOCK else "⚠️" if _n >= 2 else "👀"
        _rows.append(
            f"| `{_f}` | {_f1} | {_f2} | {_f3} | {_f4} | {_f5} | {_f6} | {_f7} | {_f8} | {_f9} | **{_n}/9** {_verdict} |"
        )

    _table   = "\n    ".join(_rows) if _rows else "| ไม่มี feature ที่ถูก flag ✅ | | | | | | | | | | |"
    _tt_line = f"AUC={adv_tt_auc:.4f} {'✅' if adv_tt_ok else '🔴'}" if adv_tt_auc else "ไม่มี test data"

    mo.md(f"""
    ## สรุปผล: Feature Risk Matrix — {len(all_check_cols)} Features × 9 เทคนิค

    | # | เทคนิค | กลุ่ม | ผลลัพธ์ |
    |---|--------|-------|--------|
    | T1 | Null Pattern | ⚡ Fast | {len(null_flagged)} 🔴, {len(null_suspicious)} 🟠 |
    | T2 | Class Separation | ⚡ Fast | {len(flagged_sep)} 🔴, {len(watchlist_sep)} 🟠 |
    | T3 | Mutual Information | ⚡ Fast | {len(mi_flagged)} 🔴, {len(mi_suspicious)} 🟠 |
    | T4 | Variance Ratio | ⚡ Fast | {len(var_flagged)} 🔴, {len(var_suspicious)} 🟠 |
    | T5 | PSI | 🕐 Temporal | {len(psi_flagged)} 🔴, {len(psi_suspicious)} 🟠 |
    | T6 | Time Stability | 🕐 Temporal | {len(unstable_cols)} 🔴 |
    | T7 | Single-Feature AUC | 🐢 Medium | {len(sf_flagged)} 🔴, {len(sf_suspicious)} 🟠 |
    | T8 | Adversarial T/T | 🔥 Slow | AUC={adv_auc:.4f} {"✅" if adv_ok else "🔴"} |
    | T9 | Adversarial T/Test | 🔥 Slow | {_tt_line} |

    ### Feature Risk Table (BLOCK threshold = {_BLOCK}/9)

    | Feature | Null | Sep | MI | VarR | PSI | Time | SF-AUC | Adv | AdvTT | Score |
    |---------|------|-----|----|------|-----|------|--------|-----|-------|-------|
    {_table}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## 🚦 ขั้นตอนต่อไปหลังอ่าน Risk Matrix

    ### BLOCK ≠ ตัดทิ้ง — ต้องพิสูจน์ก่อน

    ```
    BLOCK feature
         │
         ├─ ถาม: feature นี้ควรรู้ค่าได้ตอน predict จริงไหม?
         │    NO  → Leakage จริง → ตัดทิ้ง
         │    YES → อาจเป็น False Positive → ทดสอบต่อ
         │
         ├─ A/B Test: train WITH vs WITHOUT feature นี้
         │    Δ AUC > 0.003 → มี real signal → False Positive → เก็บไว้
         │    Δ AUC < 0.003 → ไม่มี unique signal → Leakage → ตัดทิ้ง
         │
         └─ ตรวจละเอียดใน risk_feature_analysis.py
    ```

    ### ถ้าอยากปรับ threshold
    แก้ที่ `config/leakage_thresholds.toml` แล้วรัน notebook ใหม่ — ไม่ต้องแตะ code เลย

    ### Notebook ถัดไป
    1. `risk_feature_analysis.py` — วิเคราะห์ BLOCK/WATCH features ทีละตัว
    2. `baseline.py` — Gate 1.5 block features อัตโนมัติจาก parquet
    """)
    return


@app.cell
def _(
    Path,
    adv_top_features,
    adv_tt_top_features,
    all_check_cols,
    cfg,
    cohen_d_scores,
    flagged_sep,
    mi_flagged,
    mi_scores,
    mi_suspicious,
    mo,
    null_flagged,
    null_scores,
    null_suspicious,
    pl,
    psi_flagged,
    psi_scores,
    psi_suspicious,
    sf_flagged,
    sf_scores,
    sf_suspicious,
    unstable_cols,
    var_flagged,
    var_scores,
    var_suspicious,
    watchlist_sep,
):
    import datetime as _dt

    _ROOT = Path(__file__).resolve().parent
    while not (_ROOT / "pixi.toml").exists():
        _ROOT = _ROOT.parent

    _adv_flagged    = {f for f, _ in adv_top_features[:5]}
    _adv_tt_flagged = {f for f, _ in adv_tt_top_features[:5]}
    _null_set       = set(null_flagged)    | set(null_suspicious)
    _mi_set         = set(mi_flagged)      | set(mi_suspicious)
    _var_set        = set(var_flagged)     | set(var_suspicious)
    _psi_set        = set(psi_flagged)     | set(psi_suspicious)
    _sf_set         = set(sf_flagged)      | set(sf_suspicious)

    _BLOCK = cfg["risk_score"]["block_threshold"]
    _WATCH = cfg["risk_score"]["watch_threshold"]

    _rows = []
    for _feat in all_check_cols:
        _f1 = _feat in _null_set
        _f2 = _feat in set(flagged_sep)
        _f3 = _feat in _mi_set
        _f4 = _feat in _var_set
        _f5 = _feat in _psi_set
        _f6 = _feat in set(unstable_cols)
        _f7 = _feat in _sf_set
        _f8 = _feat in _adv_flagged
        _f9 = _feat in _adv_tt_flagged
        _score      = sum([_f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8, _f9])
        _watch_flag = _feat in set(watchlist_sep)
        _rows.append({
            "feature":              _feat,
            "cohen_d":              round(cohen_d_scores.get(_feat, 0.0), 4),
            "single_feat_auc":      round(sf_scores.get(_feat, 0.5), 4),
            "mutual_information":   round(mi_scores.get(_feat, 0.0), 4),
            "null_rate_diff":       round(null_scores.get(_feat, 0.0), 4),
            "psi":                  round(psi_scores.get(_feat, 0.0), 4),
            "variance_ratio":       round(var_scores.get(_feat, 1.0), 4),
            "flag_null_pattern":    _f1,
            "flag_class_sep":       _f2,
            "flag_mutual_info":     _f3,
            "flag_variance_ratio":  _f4,
            "flag_psi":             _f5,
            "flag_time_stability":  _f6,
            "flag_single_feat_auc": _f7,
            "flag_adv_self":        _f8,
            "flag_adv_tt":          _f9,
            "risk_score":           _score,
            "verdict": "BLOCK" if _score >= _BLOCK else "WATCH" if (_score >= _WATCH or _watch_flag) else "CLEAN",
        })

    _ts     = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    risk_df = pl.DataFrame(_rows).sort("risk_score", descending=True)

    _out      = _ROOT / "data/processed/feature_risk_scores.parquet"
    _runs_dir = _ROOT / "data/processed/risk_runs"
    _runs_dir.mkdir(exist_ok=True)
    risk_df.write_parquet(_out)
    risk_df.with_columns(pl.lit(_ts).alias("run_id")).write_parquet(
        _runs_dir / f"risk_{_ts}.parquet"
    )

    _prev_runs = sorted(r for r in _runs_dir.glob("risk_*.parquet") if r.stem != f"risk_{_ts}")
    _change_rows = []
    if _prev_runs:
        _prev    = pl.read_parquet(_prev_runs[-1])
        _prev_lk = dict(zip(_prev["feature"].to_list(), _prev["verdict"].to_list()))
        for _r in risk_df.iter_rows(named=True):
            _old = _prev_lk.get(_r["feature"])
            _new = _r["verdict"]
            if _old and _old != _new:
                _arrow = "⬆️" if _new == "BLOCK" else "⬇️" if _new == "CLEAN" else "↔️"
                _change_rows.append(f"| `{_r['feature']}` | {_old} → {_new} {_arrow} |")
        _prev_ts = _prev_runs[-1].stem.replace("risk_", "")
    else:
        _prev_ts = None

    _n_block = (risk_df["verdict"] == "BLOCK").sum()
    _n_watch = (risk_df["verdict"] == "WATCH").sum()
    _n_clean = (risk_df["verdict"] == "CLEAN").sum()

    _change_section = ""
    if _change_rows:
        _change_section = f"""
    ### ∆ เปลี่ยนแปลงจาก run ก่อนหน้า ({_prev_ts})
    | Feature | Verdict Change |
    |---------|---------------|
    {"    ".join(_change_rows)}
    """
    elif _prev_ts:
        _change_section = f"\n    ✅ ไม่มีการเปลี่ยนแปลงจาก run `{_prev_ts}`\n    "

    mo.md(f"""
    ## บันทึก Feature Risk Scores → baseline.py Gate 1.5

    **Run:** `{_ts}` | **Techniques:** 9 | **BLOCK threshold:** {_BLOCK}/9

    | Verdict | จำนวน | เงื่อนไข |
    |---------|-------|---------|
    | 🚨 BLOCK | **{_n_block}** | risk_score ≥ {_BLOCK}/9 |
    | ⚠️ WATCH | **{_n_watch}** | risk_score ≥ {_WATCH} หรืออยู่ใน watchlist |
    | ✅ CLEAN | **{_n_clean}** | ผ่านทุกเทคนิค |

    📁 `data/processed/feature_risk_scores.parquet` — baseline.py Gate 1.5 อ่านตัวนี้
    📁 `data/processed/risk_runs/risk_{_ts}.parquet` — history
    {_change_section}
    """)
    return


if __name__ == "__main__":
    app.run()
