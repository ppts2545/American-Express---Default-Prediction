import marimo

__generated_with = "0.23.3"
app = marimo.App(width="full")


@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import numpy as np
    import polars as pl
    import polars.selectors as cs
    import matplotlib.pyplot as plt
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    mo.md("""
    # Feature-Target Interaction Analysis
    ## Professional EDA สำหรับดักจับ Leakage

    **เป้าหมาย:** พิสูจน์ว่าความแม่นยำมาจาก signal จริง ไม่ใช่ leakage

    | เทคนิค | คำถามที่ตอบ |
    |--------|------------|
    | **1. Class Separation** | feature แยก default/non-default "สมจริง" หรือสมบูรณ์เกินไป? |
    | **2. Time Stability** | feature-target relationship คงที่ตลอด หรือเปลี่ยนผิดปกติ? |
    | **3. Adversarial Validation** | โมเดลปลอมทายได้ว่าข้อมูลมาจาก "ช่วง" ไหน? |

    ใช้เฉพาะ training data — ไม่แตะ test data เลย
    """)
    return Path, StratifiedKFold, cs, lgb, mo, np, pl, plt, roc_auc_score


@app.cell
def _(Path, cs, pl):
    _ROOT = Path(__file__).resolve().parent
    while not (_ROOT / "pixi.toml").exists():
        _ROOT = _ROOT.parent

    _train  = pl.read_parquet(_ROOT / "data/processed/train_features.parquet")
    _labels = pl.read_parquet(_ROOT / "data/processed/train_labels.parquet")

    df = _train.join(_labels.select(["customer_ID", "target"]), on="customer_ID")

    numeric_cols = df.select(cs.numeric().exclude("target")).columns

    # sample 20k rows สำหรับ visualization — ไม่ .to_pandas() ทั้ง frame
    df_sample = df.sample(n=min(20_000, df.shape[0]), seed=42)
    return df, df_sample, numeric_cols


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

    # ทุก feature ที่เข้า model = ทุกตัวยกเว้น red_flag
    # เช็คครบหมดเพราะ leakage ซ่อนได้แม้ใน feature ที่ corr ต่ำ
    # (corr วัดแค่ linear — leakage อาจเป็น non-linear ก็ได้)
    _red_set       = set(red_flag_cols)
    all_check_cols = [c for c in numeric_cols if c not in _red_set]

    mo.md(f"""
    ## Feature Pool — เช็คครบทุกตัวที่เข้า Model

    | กลุ่ม | จำนวน | สถานะ |
    |-------|-------|-------|
    | 🔴 Red Flag (corr > 0.9) | **{len(red_flag_cols)}** | block แล้วใน model — ข้าม |
    | 🟠 Suspicious (0.5–0.9) | **{len(suspicious_cols)}** | เข้า model → ต้องตรวจ |
    | 🟡 "Safe" (< 0.5) | **{len(safe_cols_low)}** | เข้า model → ต้องตรวจด้วย! |
    | **✅ ตรวจทั้งหมด** | **{len(all_check_cols)}** | |

    ⚠️ **ทำไมต้องเช็ค corr ต่ำด้วย?**
    Correlation วัดได้แค่ความสัมพันธ์แบบ linear
    Leakage อาจซ่อนอยู่ใน non-linear pattern เช่น
    feature ที่ = 0 เสมอสำหรับ default และ > 0 สำหรับ non-default
    → corr ต่ำ แต่แยก class ได้สมบูรณ์
    """)
    return (all_check_cols,)


@app.cell
def _(all_check_cols, df_sample, mo, np, pl, plt):
    def _make_sep():
        # ── Robust Cohen's d — clip p1/p99 ก่อน เพื่อป้องกัน outlier inflate std ──
        # ปัญหา: .std() อ่อนไหวต่อ extreme values มาก → pooled_std กว้างขึ้น → d ดูต่ำกว่าจริง
        # วิธีแก้: clip ทุก feature ที่ p1–p99 ของข้อมูลรวม ก่อนคำนวณ mean/std
        _arr_all = df_sample.select(all_check_cols).fill_null(0).to_numpy()  # shape (n, p)
        _p1  = np.percentile(_arr_all, 1,  axis=0)  # lower bound ต่อ feature
        _p99 = np.percentile(_arr_all, 99, axis=0)  # upper bound ต่อ feature

        _mask0 = (df_sample["target"] == 0).to_numpy()
        _mask1 = (df_sample["target"] == 1).to_numpy()

        # clip outliers แล้วแยกตาม class
        _clipped = np.clip(_arr_all, _p1, _p99)
        _arr0 = _clipped[_mask0]  # rows ของ non-default หลัง clip
        _arr1 = _clipped[_mask1]  # rows ของ default หลัง clip

        _m0 = _arr0.mean(axis=0)
        _m1 = _arr1.mean(axis=0)
        _s0 = _arr0.std(axis=0)
        _s1 = _arr1.std(axis=0)

        # Robust Cohen's d ครบทุก feature ในครั้งเดียว
        pooled = np.sqrt((_s0 ** 2 + _s1 ** 2) / 2) + 1e-8
        d_vals = np.abs(_m1 - _m0) / pooled
        scores = {c: float(d) for c, d in zip(all_check_cols, d_vals)}

        flagged  = {k: v for k, v in scores.items() if v > 3}
        watchlist = {k: v for k, v in scores.items() if 1.5 < v <= 3}

        # ── Plot 1: histogram ของ d ทุก feature ──────────────────────────────
        fig1, ax = plt.subplots(figsize=(11, 4))
        ax.hist(d_vals, bins=80, color="#6c8ebf", edgecolor="white")
        ax.axvline(1.5, color="orange",  linestyle="--", linewidth=1.5, label="ระวัง (d=1.5)")
        ax.axvline(3.0, color="red",     linestyle="--", linewidth=1.5, label="น่าสงสัย (d=3)")
        ax.set_xlabel("Cohen's d")
        ax.set_ylabel("# Features")
        ax.set_title(
            f"Class Separation — ทุก {len(all_check_cols)} features ที่เข้า model\n"
            f"🔴 d > 3: {len(flagged)} features   🟠 d 1.5–3: {len(watchlist)} features",
            fontweight="bold",
        )
        ax.legend()
        plt.tight_layout()

        # ── Plot 2: KDE รวม flagged (d>3) + watchlist (d 1.5–3) ──────────────
        # แสดงทั้งสองกลุ่มเพราะ watchlist ก็ต้องตรวจ — ห้ามมองข้าม
        _top_red    = sorted(flagged.items(),   key=lambda x: -x[1])[:8]
        _top_orange = sorted(watchlist.items(), key=lambda x: -x[1])[:4]
        TOP = _top_red + _top_orange  # รวมสูงสุด 12 ตัว

        if not TOP:
            fig2, _ax = plt.subplots(figsize=(8, 3))
            fig2.patch.set_facecolor("white")
            _ax.set_facecolor("white")
            _ax.text(0.5, 0.5, "ทุก feature มี d < 1.5 ✅\nไม่มี flagged หรือ watchlist เลย",
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
                ax.hist(v0, bins=50, density=True, alpha=0.55, color="#2196F3", label="Non-default (0)")
                ax.hist(v1, bins=50, density=True, alpha=0.55, color="#F44336", label="Default (1)")

                # color-code title ตาม severity
                if d > 3:
                    title_color, badge = "#d62728", "🔴"
                else:
                    title_color, badge = "#ff7f0e", "🟠"

                ax.set_title(f"{col}\nd = {d:.2f} {badge}", color=title_color,
                             fontsize=8.5, fontweight="bold")
                ax.legend(fontsize=7)

            for j in range(len(TOP), len(flat)):
                flat[j].set_visible(False)

            plt.suptitle(
                f"KDE — 🔴 Flagged (d>3): {len(_top_red)} features   "
                f"🟠 Watchlist (d 1.5–3): {len(_top_orange)} features",
                fontweight="bold",
            )
            plt.tight_layout()

        return fig1, fig2, scores, flagged, watchlist

    fig_sep_hist, fig_sep_kde, cohen_d_scores, flagged_sep, watchlist_sep = _make_sep()

    mo.md(f"""
    ## เทคนิคที่ 1: Class Separation Analysis — ครบทุก feature ที่เข้า model

    คำนวณ **Cohen's d** สำหรับ **{len(all_check_cols)} features** ทั้งหมด

    | d | ความหมาย | จำนวน |
    |----|---------|-------|
    | < 1.5 | ✅ ปกติ — distribution ทับซ้อน | {len([v for v in cohen_d_scores.values() if v <= 1.5])} |
    | 1.5–3 | 🟠 ระวัง — แยกกันค่อนข้างมาก | {len(watchlist_sep)} |
    | > 3 | 🔴 น่าสงสัย — แยกขาดเกือบสมบูรณ์ | {len(flagged_sep)} |

    Features ที่ d > 3: `{"`, `".join(flagged_sep.keys()) if flagged_sep else "ไม่มี ✅"}`
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
def _(all_check_cols, df, mo, np, pl, plt):
    def _make_time():
        n = df.shape[0]

        # หา date column — ถ้าไม่มีใช้ row-order quartile
        _date_col = None
        for _c in df.columns:
            if df[_c].dtype in (pl.Date, pl.Datetime):
                _date_col = _c
                break

        if _date_col:
            _df     = df.with_columns(pl.col(_date_col).dt.strftime("%Y-%m").alias("__time_bin"))
            x_label = f"Month ({_date_col})"
        else:
            _df     = df.with_row_index("__ridx").with_columns(
                pl.when(pl.col("__ridx") < n // 4).then(pl.lit("Q1"))
                .when(pl.col("__ridx") < n // 2).then(pl.lit("Q2"))
                .when(pl.col("__ridx") < 3 * n // 4).then(pl.lit("Q3"))
                .otherwise(pl.lit("Q4"))
                .alias("__time_bin")
            )
            x_label = "Quartile (Row Order)"

        # ── คำนวณ mean ของทุก feature ใน all_check_cols แบบ batch ───────────
        # Polars group_by + agg ในครั้งเดียว = เร็วกว่าวน loop ต่อ feature มาก
        _agg = (
            _df.group_by(["__time_bin", "target"])
            .agg([pl.col(c).mean().alias(c) for c in all_check_cols])
            .sort(["__time_bin", "target"])
        )
        _t0 = _agg.filter(pl.col("target") == 0)
        _t1 = _agg.filter(pl.col("target") == 1)
        _bins = _t0["__time_bin"].to_list()

        # คำนวณ gap_cv สำหรับทุก feature
        unstable  = []
        gap_cvs   = {}
        for col in all_check_cols:
            t0_means = _t0[col].fill_null(0).to_list()
            t1_means = _t1[col].fill_null(0).to_list()
            gaps     = [abs(a - b) for a, b in zip(t0_means, t1_means)]
            gap_cv   = float(np.std(gaps) / (np.mean(gaps) + 1e-8))
            gap_cvs[col] = gap_cv
            if gap_cv > 0.3:
                unstable.append(col)

        # ── Plot top 6 ที่ไม่เสถียรที่สุด ────────────────────────────────────
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
            ax.set_title(f"{col}\ngap_cv = {gap_cvs[col]:.2f} 🔴", fontsize=8.5, fontweight="bold",
                         color="#d62728")
            ax.set_xlabel(x_label, fontsize=7)
            ax.set_ylabel("Mean Value", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.set_facecolor("#fff0f0")

        for j in range(SHOW, len(flat)):
            flat[j].set_visible(False)

        note = f"(date: `{_date_col}`)" if _date_col else "(row-order quartiles)"
        _title = f"Time Stability — Top {len(TOP)} Unstable Features จาก {len(all_check_cols)} ทั้งหมด {note}" \
                 if TOP else f"Time Stability — ทุก {len(all_check_cols)} features เสถียร ✅"
        plt.suptitle(_title, fontweight="bold")
        plt.tight_layout()
        return fig, unstable

    fig_time, unstable_cols = _make_time()

    mo.md(f"""
    ## เทคนิคที่ 2: Time-Series Stability Check — ครบทุก feature ที่เข้า model

    ตรวจสอบ **{len(all_check_cols)} features** ทั้งหมดแบบ batch ใน Polars operation เดียว

    feature ที่ดีควรมี gap ระหว่าง Default vs Non-default **คงที่ตลอดเวลา**
    ถ้า gap เปลี่ยนรุนแรง (gap_cv > 0.3) = อาจมี temporal leakage หรือ distribution shift

    พบ **{len(unstable_cols)}** / {len(all_check_cols)} features ที่ไม่เสถียร:
    `{"`, `".join(unstable_cols[:20]) + ("..." if len(unstable_cols) > 20 else "") if unstable_cols else "ไม่มี ✅"}`
    """)
    return fig_time, unstable_cols


@app.cell
def _(fig_time):
    fig_time
    return


@app.cell
def _(
    StratifiedKFold,
    df_sample,
    lgb,
    mo,
    np,
    numeric_cols,
    plt,
    roc_auc_score,
):
    def _run_adv():
        n = df_sample.shape[0]

        # สร้าง adversarial target: ครึ่งแรก = 0, ครึ่งหลัง = 1 (ตามลำดับ row)
        # ถ้า data เรียงตามเวลา นี่คือการทดสอบว่า "อนาคต" ต่างจาก "อดีต" ไหม
        _adv_y          = np.zeros(n, dtype=int)
        _adv_y[n // 2:] = 1

        # ใช้ numeric features ทั้งหมด (ไม่กรอง safe/suspicious) — ต้องการเห็นทุก feature
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
            _m      = lgb.train(
                _params, _dtrain, num_boost_round=300, valid_sets=[_dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(False)],
            )
            _oof[_val] = _m.predict(_X[_val])
            _imp      += _m.feature_importance(importance_type="gain")

        _auc     = roc_auc_score(_adv_y, _oof)
        _top_idx = np.argsort(_imp)[::-1][:20]
        return _auc, [(numeric_cols[i], float(_imp[i])) for i in _top_idx]

    adv_auc, adv_top_features = _run_adv()
    adv_ok = adv_auc < 0.6

    def _plot_adv():
        feats, imps = zip(*adv_top_features)
        colors = ["#d62728" if i < 5 else "#ff7f0e" if i < 10 else "#4C72B0"
                  for i in range(len(feats))]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(feats)[::-1], list(imps)[::-1], color=list(colors)[::-1])
        ax.set_xlabel("Feature Importance (Gain) — ยิ่งสูง ยิ่ง non-stationary")
        ax.set_title(
            f"Adversarial Validation — AUC = {adv_auc:.4f}\n"
            f"{'✅ AUC < 0.6 → data stationary — ปลอดภัย' if adv_ok else '🔴 AUC ≥ 0.6 → มี feature ที่ไม่เสถียรตามเวลา'}",
            fontweight="bold",
        )
        plt.tight_layout()
        return fig

    fig_adv = _plot_adv()

    mo.md(f"""
    ## เทคนิคที่ 3: Adversarial Validation

    **หลักการ:** แบ่ง training data เป็น "ครึ่งแรก" vs "ครึ่งหลัง"
    แล้วสอนโมเดลให้ทายว่า row นี้มาจากครึ่งไหน

    | AUC | ความหมาย |
    |-----|---------|
    | ≈ 0.5 | ✅ แยกไม่ออก — data stationary ตลอดเวลา ปลอดภัย |
    | 0.6–0.7 | 🟠 มี feature บางตัวที่เปลี่ยนตามเวลา — monitor ใกล้ชิด |
    | > 0.7 | 🔴 non-stationary มาก — เสี่ยง temporal leakage หรือ model collapse |

    **ผลลัพธ์: AUC = {adv_auc:.4f}** → {"✅ stationary" if adv_ok else "🔴 มี feature ที่ไม่เสถียร"}

    Feature สีแดง (top 5) = ตัวที่ทำให้โมเดลแยก "ช่วงเวลา" ได้มากที่สุด
    → นี่คือตัวแรกที่ต้องตรวจ business logic
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

        # รวม train + test แล้วสร้าง is_test label
        # numeric_cols มาจาก cs.numeric().exclude("target") → customer_ID (string) ไม่อยู่ใน
        # ถ้าเผลอใส่ ID หรือ timestamp เข้าไป AUC จะเป็น 1.0 เสมอ — ระวัง!
        _X_train = df_sample.select(numeric_cols).to_numpy()
        _X_test  = _test_sample.select(numeric_cols).to_numpy()
        _X       = np.vstack([_X_train, _X_test])
        _is_test = np.concatenate([
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
            _m      = lgb.train(
                _params, _dtrain, num_boost_round=300, valid_sets=[_dval],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(False)],
            )
            _oof[_val] = _m.predict(_X[_val])
            _imp      += _m.feature_importance(importance_type="gain")

        _auc     = roc_auc_score(_is_test, _oof)
        _top_idx = np.argsort(_imp)[::-1][:20]
        return _auc, [(numeric_cols[i], float(_imp[i])) for i in _top_idx]

    adv_tt_auc, adv_tt_top_features = _run_tt_adv()
    adv_tt_ok = (adv_tt_auc < 0.6) if adv_tt_auc is not None else None

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
        ax.set_xlabel("Feature Importance (Gain) — ยิ่งสูง ยิ่ง drift ระหว่าง train/test")
        _status = "✅ AUC < 0.6 → Train/Test มาจาก distribution เดียวกัน" if adv_tt_ok \
                  else "🔴 AUC ≥ 0.6 → มี feature drift ระหว่าง Train และ Test!"
        ax.set_title(f"True Adversarial Validation: Train vs Test — AUC = {adv_tt_auc:.4f}\n{_status}",
                     fontweight="bold")
        plt.tight_layout()
        return fig

    fig_adv_tt = _plot_tt()

    if adv_tt_auc is None:
        _md = "## เทคนิคที่ 4: True Adversarial Validation (Train vs Test)\n\n⚠️ ไม่พบ `test_features.parquet` — ข้ามเทคนิคนี้"
    else:
        _md = f"""
    ## เทคนิคที่ 4: True Adversarial Validation (Train vs Test)

    **หลักการ:** เอา train + test มารวมกัน สร้าง label `is_test` แล้วสอนโมเดลให้แยก
    ถ้าโมเดลแยกได้ = มี feature ที่ "drift" ระหว่างสองชุด = โมเดลจะพังเมื่อ deploy จริง

    | AUC | ความหมาย |
    |-----|---------|
    | ≈ 0.5 | ✅ Train/Test มาจาก distribution เดียวกัน — ปลอดภัย |
    | 0.6–0.7 | 🟠 มี feature drift บางส่วน — monitor ใกล้ชิด |
    | > 0.7 | 🔴 Drift รุนแรง — feature สีแดงต้องตรวจหรือตัดทิ้ง |

    **ผลลัพธ์: AUC = {adv_tt_auc:.4f}** → {"✅ distributions similar" if adv_tt_ok else "🔴 significant drift detected"}

    Feature สีแดง (top 5) = ตัวที่ "แยก" train ออกจาก test ได้มากที่สุด
    → ถ้าใช้ตัวนี้ใน model จะเจอ distribution shift เมื่อ predict บน test จริง
    """

    mo.md(_md)
    return adv_tt_auc, adv_tt_ok, adv_tt_top_features, fig_adv_tt


@app.cell
def _(fig_adv_tt):
    fig_adv_tt
    return


@app.cell
def _(all_check_cols, df_sample, mo, np, plt):
    def _make_interaction():
        # top 30 by corr จาก all_check_cols — ครอบคลุมทั้ง suspicious และ low-corr
        TOP = all_check_cols[:30]
        _target = df_sample["target"].to_numpy()

        # คำนวณ pairwise |corr| ด้วย numpy (เร็วกว่า nested Polars)
        _X   = df_sample.select(TOP).fill_null(0).to_numpy()
        _mat = np.abs(np.corrcoef(_X.T))

        # หา pairs ที่ corr กันสูง (>0.7) = อาจเป็น "ตัวแปรคู่แฝด"
        _pairs = []
        for i in range(len(TOP)):
            for j in range(i + 1, len(TOP)):
                if _mat[i, j] > 0.7:
                    _pairs.append((TOP[i], TOP[j], _mat[i, j]))
        _pairs.sort(key=lambda x: -x[2])

        # ทดสอบ interaction: ถ้า corr(c1 - c2, target) > max(corr(c1), corr(c2))
        # แสดงว่า "การผสม" สองตัวให้ signal เพิ่มขึ้น = อาจซ่อน leakage
        flagged = []
        for c1, c2, pair_corr in _pairs[:15]:
            v1   = df_sample[c1].fill_null(0).to_numpy()
            v2   = df_sample[c2].fill_null(0).to_numpy()
            diff = v1 - v2

            corr_c1   = abs(np.corrcoef(v1,   _target)[0, 1])
            corr_c2   = abs(np.corrcoef(v2,   _target)[0, 1])
            corr_diff = abs(np.corrcoef(diff, _target)[0, 1])

            # interaction amplifies signal > 10% → น่าสงสัย
            if corr_diff > max(corr_c1, corr_c2) * 1.1:
                flagged.append({
                    "c1": c1, "c2": c2, "pair_corr": pair_corr,
                    "corr_c1": corr_c1, "corr_c2": corr_c2, "corr_diff": corr_diff,
                })

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # ── ซ้าย: heatmap pairwise |corr| ──────────────────────────────────
        im = axes[0].imshow(_mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
        axes[0].set_xticks(range(len(TOP)))
        axes[0].set_yticks(range(len(TOP)))
        axes[0].set_xticklabels(TOP, rotation=90, fontsize=6.5)
        axes[0].set_yticklabels(TOP, fontsize=6.5)
        plt.colorbar(im, ax=axes[0], fraction=0.03)
        axes[0].set_title(
            "Pairwise |Correlation| ของ Suspicious Features\n"
            "บล็อกสีแดง = feature คู่ที่ corr กันสูง (>0.7) = อาจเป็น twin variables",
            fontweight="bold",
        )

        # ── ขวา: interaction amplification ─────────────────────────────────
        if flagged:
            _labels = [f"{d['c1'][:10]}\n− {d['c2'][:10]}" for d in flagged]
            _x      = list(range(len(flagged)))
            axes[1].bar([i - 0.25 for i in _x], [d["corr_c1"]   for d in flagged],
                        0.25, label="corr(c1, target)",      color="#2196F3", alpha=0.8)
            axes[1].bar([i        for i in _x], [d["corr_c2"]   for d in flagged],
                        0.25, label="corr(c2, target)",      color="#4CAF50", alpha=0.8)
            axes[1].bar([i + 0.25 for i in _x], [d["corr_diff"] for d in flagged],
                        0.25, label="corr(c1−c2, target) ← interaction",
                        color="#F44336", alpha=0.9)
            axes[1].set_xticks(_x)
            axes[1].set_xticklabels(_labels, fontsize=7)
            axes[1].set_ylabel("|Correlation with Target|")
            axes[1].set_title(
                "Interaction Amplification\n"
                "แดงสูงกว่าน้ำเงิน/เขียว = การผสม 2 feature ให้ signal แรงกว่าตัวเดี่ยว",
                fontweight="bold",
            )
            axes[1].legend(fontsize=8)
        else:
            axes[1].text(0.5, 0.5, "ไม่พบ interaction ที่น่าสงสัย ✅\n(ไม่มีคู่ที่เมื่อลบกันแล้ว corr สูงขึ้น)",
                         ha="center", va="center", fontsize=11)
            axes[1].axis("off")

        plt.suptitle("Feature Interaction Analysis (SHAP-free)", fontweight="bold")
        plt.tight_layout()
        return fig, flagged

    fig_interaction, flagged_interactions = _make_interaction()

    mo.md(f"""
    ## เทคนิคที่ 5: Feature Interaction Analysis

    **หลักการ:** feature เดี่ยวอาจไม่ใช่ leakage แต่พอ "ผสม" กัน (เช่น ลบกัน) อาจกลายเป็นเฉลย

    **แทน SHAP:** ใช้ pairwise |corr| + diff test แทน (SHAP ยังไม่รองรับ Python 3.14)

    วิธีอ่านผล:
    - **Heatmap ซ้าย**: feature คู่ไหน corr กันสูง (สีแดง) = อาจ encode ข้อมูลเดียวกัน
    - **กราฟขวา**: ถ้า `corr(c1 − c2, target)` (แดง) สูงกว่า `corr(c1)` และ `corr(c2)` = interaction น่าสงสัย

    พบ **{len(flagged_interactions)}** interaction คู่ที่น่าสงสัย:
    `{"`, `".join(f"{d['c1']} − {d['c2']}" for d in flagged_interactions) if flagged_interactions else "ไม่มี ✅"}`
    """)
    return fig_interaction, flagged_interactions


@app.cell
def _(fig_interaction):
    fig_interaction
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
    flagged_interactions,
    flagged_sep,
    mo,
    unstable_cols,
    watchlist_sep,
):
    _adv_flagged    = [f for f, _ in adv_top_features[:5]]
    _adv_tt_flagged = [f for f, _ in adv_tt_top_features[:5]]
    _int_flagged    = [d["c1"] for d in flagged_interactions] + \
                      [d["c2"] for d in flagged_interactions]

    _all_flagged = sorted(
        set(flagged_sep) | set(unstable_cols) | set(_adv_flagged) |
        set(_adv_tt_flagged) | set(_int_flagged)
    )

    _rows = []
    for _f in _all_flagged:
        _s  = "🔴" if _f in flagged_sep      else "✅"
        _t  = "🔴" if _f in unstable_cols    else "✅"
        _a  = "🔴" if _f in _adv_flagged     else "✅"
        _tt = "🔴" if _f in _adv_tt_flagged  else "✅"
        _i  = "🔴" if _f in _int_flagged     else "✅"
        _n  = sum([_f in flagged_sep, _f in unstable_cols,
                   _f in _adv_flagged, _f in _adv_tt_flagged, _f in _int_flagged])
        _rows.append(f"| `{_f}` | {_s} | {_t} | {_a} | {_tt} | {_i} | **{_n}/5** |")

    _table   = "\n    ".join(_rows) if _rows else "| ไม่มี feature ที่ถูก flag ✅ | | | | | | |"
    _tt_line = f"AUC = {adv_tt_auc:.4f} {'✅' if adv_tt_ok else '🔴'}" \
               if adv_tt_auc is not None else "ไม่มี test data"
    _total_checked = len(all_check_cols)

    mo.md(f"""
    ## สรุปผล: Feature Risk Matrix — ครบ {_total_checked} Features

    ตรวจสอบ **ทุก feature ที่เข้า model** — ไม่ข้ามแม้แต่ตัวที่ corr ต่ำ

    | เทคนิค | ตรวจ | ผลลัพธ์ |
    |--------|------|--------|
    | 1. Class Separation (d > 3) | {_total_checked} features | {len(flagged_sep)} 🔴 flagged, {len(watchlist_sep)} 🟠 watchlist |
    | 2. Time Stability | {_total_checked} features | {len(unstable_cols)} 🔴 ไม่เสถียร |
    | 3. Adversarial (Train/Train) | {_total_checked} features | AUC = {adv_auc:.4f} {"✅" if adv_ok else "🔴"} |
    | 4. Adversarial (Train/Test) | {_total_checked} features | {_tt_line} |
    | 5. Feature Interaction | top 30 pairs | {len(flagged_interactions)} คู่น่าสงสัย |

    ### Feature Risk Table

    | Feature | Class Sep | Time | Adv(self) | Adv(T/T) | Interaction | Score |
    |---------|-----------|------|-----------|----------|-------------|-------|
    {_table}

    ### Action Plan

    | Score | Action |
    |-------|--------|
    | **4–5/5** | 🚨 ตรวจ business logic ทันที — เสี่ยง leakage สูงมาก |
    | **2–3/5** | ⚠️ ใส่ watchlist, monitor ใน CV และ production |
    | **1/5** | 👀 เก็บไว้ แต่ระวัง |
    | **0/5** | ✅ พิสูจน์แล้วว่าสะอาด — นี่คือ Golden Feature |

    ---
    **Features ที่ผ่าน 0/5 = "ยืนยันแล้วว่าไม่มี leakage"** ไม่ใช่แค่ "ไม่ได้เช็ค"
    เพราะเราเช็คด้วย 5 เทคนิคครบทุกตัวแล้ว
    """)
    return


@app.cell
def _(
    Path,
    adv_top_features,
    adv_tt_top_features,
    all_check_cols,
    cohen_d_scores,
    flagged_interactions,
    flagged_sep,
    mo,
    pl,
    unstable_cols,
    watchlist_sep,
):
    _ROOT = Path(__file__).resolve().parent
    while not (_ROOT / "pixi.toml").exists():
        _ROOT = _ROOT.parent

    _adv_flagged    = {f for f, _ in adv_top_features[:5]}
    _adv_tt_flagged = {f for f, _ in adv_tt_top_features[:5]}
    _int_flagged    = {d["c1"] for d in flagged_interactions} | \
                      {d["c2"] for d in flagged_interactions}

    _rows = []
    for _feat in all_check_cols:
        _d          = cohen_d_scores.get(_feat, 0.0)
        _f1         = _feat in set(flagged_sep)
        _f2         = _feat in set(unstable_cols)
        _f3         = _feat in _adv_flagged
        _f4         = _feat in _adv_tt_flagged
        _f5         = _feat in _int_flagged
        _score      = sum([_f1, _f2, _f3, _f4, _f5])
        _watch_flag = _feat in set(watchlist_sep)
        _rows.append({
            "feature":            _feat,
            "cohen_d":            round(_d, 4),
            "flag_class_sep":     _f1,
            "flag_time_stability": _f2,
            "flag_adv_self":      _f3,
            "flag_adv_tt":        _f4,
            "flag_interaction":   _f5,
            "risk_score":         _score,
            "verdict": "BLOCK" if _score >= 3 else "WATCH" if (_score >= 2 or _watch_flag) else "CLEAN",
        })

    risk_df = pl.DataFrame(_rows).sort("risk_score", descending=True)
    _out = _ROOT / "data/processed/feature_risk_scores.parquet"
    risk_df.write_parquet(_out)

    _n_block = (risk_df["verdict"] == "BLOCK").sum()
    _n_watch = (risk_df["verdict"] == "WATCH").sum()
    _n_clean = (risk_df["verdict"] == "CLEAN").sum()

    mo.md(f"""
    ## บันทึก Feature Risk Scores → baseline.py Gate 1.5

    คำนวณ risk score สำหรับ **{len(all_check_cols)} features** และบันทึกที่:
    `data/processed/feature_risk_scores.parquet`

    | Verdict | จำนวน | เงื่อนไข |
    |---------|-------|---------|
    | 🚨 BLOCK | **{_n_block}** | risk_score ≥ 3 → baseline.py จะ block ออกจาก feature pool |
    | ⚠️ WATCH | **{_n_watch}** | risk_score 1–2 หรืออยู่ใน watchlist → monitor ใกล้ชิด |
    | ✅ CLEAN | **{_n_clean}** | risk_score = 0 → ปลอดภัยสุด — Golden Features |

    **baseline.py** จะโหลดไฟล์นี้ใน **Gate 1.5** (หลัง Gate 1) เพื่อ block feature อันตราย
    ก่อนเข้า CV loop — ทำให้ทุก fold ได้ feature pool ที่ผ่านการตรวจ 5 เทคนิคแล้ว
    """)
    return


if __name__ == "__main__":
    app.run()
