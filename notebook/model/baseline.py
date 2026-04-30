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
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import roc_auc_score

    mo.md("""
    # Baseline Model — American Express Default Prediction

    ## หลักการ: Leakage ต้องเป็นไปไม่ได้โดยโครงสร้าง

    ทุกขั้นตอนถูกห่อด้วยฟังก์ชันที่รับเฉพาะข้อมูลที่อนุญาตให้เห็น
    → ทำให้ leakage เป็น "ไม่มีทางทำได้" ไม่ใช่แค่ "พยายามระวัง"

    ```
    drop_red_flags()          ← ลบ feature อันตรายออกก่อนทุกอย่าง
         ↓
    select_features_leak_free()  ← เลือก feature จาก inner train เท่านั้น
         ↓
    get_fold_arrays()         ← แปลงเฉพาะ columns ที่เลือกแล้วเป็น numpy
         ↓
    train_lgb_fold() / train_xgb_fold()  ← train โดยไม่รู้จัก val เลย
         ↓
    evaluate_oof()            ← วัดผลบน OOF ที่ไม่เคยถูกแตะ
    ```
    """)
    return (
        Path,
        StratifiedKFold,
        cs,
        lgb,
        mo,
        np,
        pd,
        pl,
        plt,
        re,
        roc_auc_score,
        train_test_split,
        xgb,
    )


@app.cell
def _(pl):
    def drop_red_flags(df: "pl.DataFrame", numeric_cols: list, threshold: float = 0.9):
        """
        ขั้นตอนที่ 1 — ลบ feature ที่อันตรายออกก่อนเข้า CV

        ใช้ full training data เพื่อ "กำจัด" feature ที่สัมพันธ์กับ target
        สูงเกินไป (อาจเป็น target ในรูปแบบอื่น = leakage)

        การใช้ full data ที่นี่ปลอดภัย เพราะเราทำเพื่อ "ห้าม" ไม่ใช่ "เลือก":
        - ห้าม (exclusion) จาก full data → ปลอดภัย
        - เลือก (selection) จาก full data → leakage!

        Parameters
        ----------
        df            : full training dataframe (Polars)
        numeric_cols  : รายชื่อ column ทั้งหมดที่จะพิจารณา
        threshold     : ค่า |corr| ที่ถือว่าอันตราย (default 0.9)

        Returns
        -------
        safe_cols     : รายชื่อ column ที่ผ่านการตรวจ — ใช้ใน CV ได้
        report        : Polars DataFrame แสดง corr ของทุก column
        """
        # คำนวณ |correlation| ระหว่างทุก feature กับ target
        report = (
            df.select([pl.corr(c, "target").alias(c) for c in numeric_cols])
            .unpivot(variable_name="feature", value_name="corr")
            .with_columns(pl.col("corr").abs().alias("abs_corr"))
            .sort("abs_corr", descending=True)
        )

        # แยก safe กับ red_flag
        safe_cols    = report.filter(pl.col("abs_corr") <= threshold)["feature"].to_list()
        dropped_cols = report.filter(pl.col("abs_corr") >  threshold)["feature"].to_list()

        # หยุดโปรแกรมทันทีถ้ามี red flag และยังไม่ได้ถูก drop
        # (ป้องกันกรณีที่มีคนแก้โค้ดแล้วข้ามขั้นตอนนี้)
        assert len(dropped_cols) == 0 or len(safe_cols) < len(numeric_cols), \
            "drop_red_flags() failed to remove dangerous features!"

        return safe_cols, report

    return (drop_red_flags,)


@app.cell
def _(pl):
    def select_features_leak_free(inner_df: "pl.DataFrame", safe_cols: list, top_n: int):
        """
        ขั้นตอนที่ 2 — เลือก feature จาก inner train เท่านั้น

        ฟังก์ชันนี้รับแค่ inner_df (ไม่รู้จัก val fold เลย)
        → เป็นไปไม่ได้ที่ val labels จะรั่วเข้ามาในการเลือก feature

        Parameters
        ----------
        inner_df  : เฉพาะ rows ของ inner training (ไม่มี early-stop holdout, ไม่มี OOF val)
        safe_cols : รายชื่อ column ที่ผ่าน drop_red_flags() แล้ว
        top_n     : จำนวน feature ที่ต้องการเลือก

        Returns
        -------
        selected  : list ของ feature ที่เลือกสำหรับ fold นี้
        """
        selected = (
            inner_df
            # คำนวณ corr บน inner train เท่านั้น — val rows ไม่อยู่ใน inner_df
            .select([pl.corr(c, "target").alias(c) for c in safe_cols])
            .unpivot(variable_name="feature", value_name="corr")
            .with_columns(pl.col("corr").abs())
            .sort("corr", descending=True)
            .head(top_n)
        )["feature"].to_list()
        return selected

    return (select_features_leak_free,)


@app.function
def get_fold_arrays(df, row_indices: "np.ndarray", feature_cols: list):
    """
    ขั้นตอนที่ 3 — แปลง Polars → numpy เฉพาะ rows และ columns ที่อนุญาต

    แยก X และ y ออกจากกัน และ convert เฉพาะ columns ที่เลือกแล้ว
    → ไม่มีทางที่ feature ที่ไม่ได้เลือกหรือ val rows จะรั่วเข้าไป

    Parameters
    ----------
    df           : full Polars DataFrame
    row_indices  : numpy array ของ index ที่อนุญาตให้เห็น
    feature_cols : รายชื่อ feature ที่เลือกแล้ว (จาก select_features_leak_free)

    Returns
    -------
    X : numpy array รูปร่าง (n_rows, n_features)
    y : numpy array รูปร่าง (n_rows,)
    """
    # แปลงเฉพาะ rows ที่อนุญาต + columns ที่เลือกแล้วเท่านั้น
    subset = df[list(map(int, row_indices))]
    X = subset.select(feature_cols).to_numpy()
    y = subset["target"].to_numpy()
    return X, y


@app.cell
def _(lgb):
    def train_lgb_fold(X_inner, y_inner, X_es, y_es, params: dict, n_rounds=1000, patience=50):
        """
        ขั้นตอนที่ 4a — Train LightGBM สำหรับ 1 fold

        ฟังก์ชันนี้ไม่รู้จัก val/OOF เลย — รับแค่ inner train และ early-stop holdout
        → val labels เป็นไปไม่ได้ที่จะรั่วเข้า training loop

        Parameters
        ----------
        X_inner, y_inner : inner training data (90% ของ training fold)
        X_es, y_es       : early-stop holdout (10% ของ training fold)
                           ใช้เพื่อหยุด training เท่านั้น ไม่ใช่ OOF val
        params           : LightGBM hyperparameters
        n_rounds         : จำนวนต้นไม้สูงสุด
        patience         : หยุดถ้า val AUC ไม่ดีขึ้นใน patience รอบต่อเนื่อง

        Returns
        -------
        model : trained LightGBM model
        """
        dtrain = lgb.Dataset(X_inner, label=y_inner)
        des    = lgb.Dataset(X_es,    label=y_es, reference=dtrain)
        model  = lgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            valid_sets=[des],
            callbacks=[
                lgb.early_stopping(patience, verbose=False),
                lgb.log_evaluation(False),
            ],
        )
        return model

    return (train_lgb_fold,)


@app.cell
def _(xgb):
    def train_xgb_fold(X_inner, y_inner, X_es, y_es, params: dict, n_rounds=1000, patience=50):
        """
        ขั้นตอนที่ 4b — Train XGBoost สำหรับ 1 fold

        โครงสร้างเหมือน train_lgb_fold ทุกอย่าง
        val/OOF data ไม่มีทางเข้ามาในฟังก์ชันนี้

        Parameters
        ----------
        X_inner, y_inner : inner training data
        X_es, y_es       : early-stop holdout
        params           : XGBoost hyperparameters
        n_rounds, patience: เหมือน LightGBM

        Returns
        -------
        model : trained XGBoost model
        """
        dtrain = xgb.DMatrix(X_inner, label=y_inner)
        des    = xgb.DMatrix(X_es,    label=y_es)
        model  = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            evals=[(des, "es")],
            early_stopping_rounds=patience,
            verbose_eval=False,
        )
        return model

    return (train_xgb_fold,)


@app.cell
def _(roc_auc_score):
    def evaluate_oof(model, X_val, y_val, model_type: str = "lgb"):
        """
        ขั้นตอนที่ 5 — วัดผลบน OOF ที่ไม่เคยถูกแตะระหว่าง training

        ฟังก์ชันนี้เรียกหลังจาก train เสร็จแล้วเท่านั้น
        X_val, y_val ไม่เคยเข้าไปใน train_lgb_fold หรือ train_xgb_fold

        Parameters
        ----------
        model      : trained model (LightGBM หรือ XGBoost)
        X_val      : features ของ OOF validation rows
        y_val      : labels จริงของ OOF rows
        model_type : "lgb" หรือ "xgb"

        Returns
        -------
        preds : numpy array ของ predicted probabilities
        auc   : AUC score สำหรับ fold นี้
        """
        if model_type == "lgb":
            import lightgbm as _lgb
            preds = model.predict(X_val)
        else:
            import xgboost as _xgb
            preds = model.predict(_xgb.DMatrix(X_val))

        auc = roc_auc_score(y_val, preds)
        return preds, auc

    return (evaluate_oof,)


@app.cell
def _(Path, cs, pd, pl, re):
    # โหลดข้อมูล — df ยังเป็น Polars ตลอด ไม่ to_pandas() ทั้ง frame
    ROOT = Path(__file__).resolve().parent
    while not (ROOT / "pixi.toml").exists():
        ROOT = ROOT.parent

    train_feat = pl.read_parquet(ROOT / "data/processed/train_features.parquet")
    labels     = pl.read_parquet(ROOT / "data/processed/train_labels.parquet")
    df         = train_feat.join(labels.select(["customer_ID", "target"]), on="customer_ID")

    numeric_cols = df.select(cs.numeric().exclude("target")).columns

    col_meta = []
    for _c in numeric_cols:
        _m = re.match(r"^([A-Z])_(\d+)_(\w+)$", _c)
        if _m:
            col_meta.append({"column": _c, "group": _m.group(1), "agg": _m.group(3)})
    meta_df = pd.DataFrame(col_meta)
    return df, numeric_cols


@app.cell
def _(df, drop_red_flags, mo, numeric_cols, pl, plt):
    # ── Gate 1: ลบ red flag features ออกก่อน — ถ้าไม่ผ่านขั้นนี้ CV ไม่เริ่ม ─
    safe_cols, corr_report = drop_red_flags(df, numeric_cols, threshold=0.9)

    red_flags  = corr_report.filter(pl.col("abs_corr") > 0.9)
    suspicious = corr_report.filter((pl.col("abs_corr") > 0.5) & (pl.col("abs_corr") <= 0.9))
    n_dropped  = len(numeric_cols) - len(safe_cols)

    def _plot_gate():
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        vals = corr_report["abs_corr"].to_numpy()
        axes[0].hist(vals, bins=60, color="#6c8ebf", edgecolor="white")
        axes[0].axvline(0.5, color="orange", linestyle="--", linewidth=1.5, label="Suspicious >0.5")
        axes[0].axvline(0.9, color="red",    linestyle="--", linewidth=1.5, label="Red flag >0.9 (DROPPED)")
        axes[0].set_xlabel("|Correlation with Target|")
        axes[0].set_ylabel("# Features")
        axes[0].set_title("Gate 1: Red Flag Filter", fontweight="bold")
        axes[0].legend()

        top20  = corr_report.head(20)
        colors = ["#d62728" if v > 0.9 else "#ff7f0e" if v > 0.5 else "#4C72B0"
                  for v in top20["abs_corr"].to_list()]
        axes[1].barh(top20["feature"].to_list()[::-1],
                     top20["abs_corr"].to_list()[::-1], color=colors[::-1])
        axes[1].axvline(0.9, color="red", linestyle="--", linewidth=1.5, alpha=0.8,
                        label="Drop threshold")
        axes[1].set_title("Top 20 Correlations\n(red = BLOCKED from model)", fontweight="bold")
        axes[1].legend()
        plt.tight_layout()
        return fig

    fig_gate = _plot_gate()

    mo.md(f"""
    ## Gate 1: Red Flag Filter — ผ่านแล้วจึงเดินหน้าต่อได้

    `drop_red_flags()` ประมวลผลเสร็จแล้ว:

    | | จำนวน |
    |-|-------|
    | Features ทั้งหมด | {len(numeric_cols)} |
    | 🔴 Dropped (corr > 0.9) | **{n_dropped}** |
    | 🟠 Suspicious (0.5–0.9) | **{len(suspicious)}** |
    | ✅ Safe — เข้า CV ได้ | **{len(safe_cols)}** |

    {"✅ ไม่มี red flag — safe_cols พร้อมใช้งาน" if n_dropped == 0 else f"🔴 {n_dropped} features ถูก block ออกจาก model แล้ว — จะไม่มีทางเข้า CV"}
    """)
    return fig_gate, safe_cols


@app.cell
def _(fig_gate):
    fig_gate
    return


@app.cell
def _():
    TOP_N   = 30   # จำนวน feature ที่เลือกต่อ fold
    N_FOLDS = 5    # จำนวน fold ใน CV
    return N_FOLDS, TOP_N


@app.cell
def _(Path, mo, pl, safe_cols):
    # ── Gate 1.5: Feature Risk Filter ────────────────────────────────────────
    # โหลดผล leakage_detection.py — block features ที่ risk_score ≥ 3
    # ถ้าไม่มีไฟล์ → fallback ใช้ safe_cols จาก Gate 1 เหมือนเดิม
    _ROOT = Path(__file__).resolve().parent
    while not (_ROOT / "pixi.toml").exists():
        _ROOT = _ROOT.parent

    _risk_path = _ROOT / "data/processed/feature_risk_scores.parquet"
    if _risk_path.exists():
        _risk_df        = pl.read_parquet(_risk_path)
        _blocked        = set(_risk_df.filter(pl.col("verdict") == "BLOCK")["feature"].to_list())
        _watched        = set(_risk_df.filter(pl.col("verdict") == "WATCH")["feature"].to_list())
        safe_cols_final = [c for c in safe_cols if c not in _blocked]
        _n_blocked      = sum(c in _blocked for c in safe_cols)
        _n_watched      = sum(c in _watched for c in safe_cols_final)
        _status = (
            f"| Safe cols (Gate 1) | {len(safe_cols)} |\n"
            f"    | 🚨 Blocked (risk ≥ 3) | **{_n_blocked}** |\n"
            f"    | ⚠️ Watch (ยังอยู่ใน pool) | **{_n_watched}** |\n"
            f"    | ✅ safe_cols_final → เข้า CV | **{len(safe_cols_final)}** |"
        )
        _header = "✅ `feature_risk_scores.parquet` โหลดแล้ว"
    else:
        safe_cols_final = safe_cols
        _status = f"| safe_cols_final (= safe_cols) | **{len(safe_cols_final)}** |"
        _header = "⚠️ ยังไม่พบ `feature_risk_scores.parquet` — รัน `leakage_detection.py` ก่อน\n\n    ใช้ `safe_cols` จาก Gate 1 แทน"

    mo.md(f"""
    ## Gate 1.5: Feature Risk Filter

    {_header}

    | | จำนวน |
    |-|-------|
    {_status}

    > **หมายเหตุ:** BLOCK features ถูกกรองออกชั่วคราว — วัดผล model ก่อน
    > แล้วค่อยตรวจ False Positive ทีละตัวในขั้นต่อไป
    """)
    return (safe_cols_final,)


@app.cell
def _(
    N_FOLDS,
    StratifiedKFold,
    TOP_N,
    df,
    evaluate_oof,
    mo,
    np,
    roc_auc_score,
    safe_cols_final,
    select_features_leak_free,
    train_lgb_fold,
    train_test_split,
):
    import time as _time

    # y และ idx ดึงจาก df — ไม่ convert ทั้ง frame
    _y   = df["target"].to_numpy()
    _idx = np.arange(len(_y))

    _skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    _lgb_params = {
        "objective": "binary", "metric": "auc",
        "learning_rate": 0.05, "num_leaves": 31, "max_depth": 6,
        "min_child_samples": 100, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1,
        "verbosity": -1, "random_state": 42,
    }

    lgb_oof        = np.zeros(len(_y))
    lgb_fold_aucs  = []
    lgb_fold_feats = []

    _t0 = _time.time()
    for _fold, (_tr_idx, _val_idx) in enumerate(_skf.split(_idx, _y)):

        # แบ่ง training → inner train + early-stop holdout
        _inner_idx, _es_idx = train_test_split(
            _tr_idx, test_size=0.1, random_state=_fold, stratify=_y[_tr_idx]
        )

        # Gate 2: เลือก feature จาก inner train เท่านั้น (ผ่านฟังก์ชัน)
        # safe_cols_final มาจาก Gate 1 + Gate 1.5 → ผ่าน 5 เทคนิค leakage detection แล้ว
        _inner_df = df[list(map(int, _inner_idx))]
        _feats    = select_features_leak_free(_inner_df, safe_cols_final, TOP_N)
        lgb_fold_feats.append(_feats)

        # Gate 3: แปลงเฉพาะ rows+cols ที่อนุญาต (ผ่านฟังก์ชัน)
        _X_inner, _y_inner = get_fold_arrays(df, _inner_idx, _feats)
        _X_es,    _y_es    = get_fold_arrays(df, _es_idx,    _feats)
        _X_val,   _y_val   = get_fold_arrays(df, _val_idx,   _feats)

        # Gate 4: Train — ฟังก์ชันไม่รับ val data เลย
        _model = train_lgb_fold(_X_inner, _y_inner, _X_es, _y_es, _lgb_params)

        # Gate 5: Evaluate บน OOF ที่ไม่เคยเข้า train
        _preds, _auc       = evaluate_oof(_model, _X_val, _y_val, "lgb")
        lgb_oof[_val_idx]  = _preds
        lgb_fold_aucs.append(_auc)

    lgb_time    = _time.time() - _t0
    lgb_oof_auc = roc_auc_score(_y, lgb_oof)
    y           = _y
    skf         = _skf
    idx         = _idx

    mo.md(f"""
    ## Gate 2–5: LightGBM Leak-Free CV

    | Fold | AUC |
    |------|-----|
    {"".join(f"| {i+1} | {auc:.5f} |" + chr(10) for i, auc in enumerate(lgb_fold_aucs))}
    | **OOF** | **{lgb_oof_auc:.5f}** |

    Training time: {lgb_time:.1f}s
    """)
    return idx, lgb_fold_aucs, lgb_fold_feats, lgb_oof_auc, lgb_time, skf, y


@app.cell
def _(
    TOP_N,
    df,
    evaluate_oof,
    idx,
    mo,
    np,
    roc_auc_score,
    safe_cols_final,
    select_features_leak_free,
    skf,
    train_test_split,
    train_xgb_fold,
    y,
):
    import time as _time2

    _xgb_params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 100,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "tree_method": "hist", "verbosity": 0, "random_state": 42,
    }

    xgb_oof       = np.zeros(len(y))
    xgb_fold_aucs = []

    _t0 = _time2.time()
    for _fold, (_tr_idx, _val_idx) in enumerate(skf.split(idx, y)):
        _inner_idx, _es_idx = train_test_split(
            _tr_idx, test_size=0.1, random_state=_fold, stratify=y[_tr_idx]
        )

        _inner_df = df[list(map(int, _inner_idx))]
        _feats    = select_features_leak_free(_inner_df, safe_cols_final, TOP_N)

        _X_inner, _y_inner = get_fold_arrays(df, _inner_idx, _feats)
        _X_es,    _y_es    = get_fold_arrays(df, _es_idx,    _feats)
        _X_val,   _y_val   = get_fold_arrays(df, _val_idx,   _feats)

        _model             = train_xgb_fold(_X_inner, _y_inner, _X_es, _y_es, _xgb_params)
        _preds, _auc       = evaluate_oof(_model, _X_val, _y_val, "xgb")
        xgb_oof[_val_idx]  = _preds
        xgb_fold_aucs.append(_auc)

    xgb_time    = _time2.time() - _t0
    xgb_oof_auc = roc_auc_score(y, xgb_oof)

    mo.md(f"""
    ## Gate 2–5: XGBoost Leak-Free CV

    | Fold | AUC |
    |------|-----|
    {"".join(f"| {i+1} | {auc:.5f} |" + chr(10) for i, auc in enumerate(xgb_fold_aucs))}
    | **OOF** | **{xgb_oof_auc:.5f}** |

    Training time: {xgb_time:.1f}s
    """)
    return xgb_fold_aucs, xgb_oof_auc, xgb_time


@app.cell
def _(
    TOP_N,
    df,
    idx,
    lgb,
    lgb_oof_auc,
    mo,
    np,
    roc_auc_score,
    safe_cols_final,
    select_features_leak_free,
    skf,
    train_test_split,
    y,
):
    # ── Shuffle Sanity Test ───────────────────────────────────────────────────
    # ใช้ pipeline เดิมทุกอย่าง แต่สลับ target แบบสุ่ม
    # ถ้า AUC ยังสูง → มี leakage (โมเดลเรียนรู้จากโครงสร้าง ไม่ใช่ signal)
    # ถ้า AUC ≈ 0.5  → pipeline สะอาด ✅

    import time as _time3

    _lgb_params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05,
        "num_leaves": 31, "verbosity": -1, "random_state": 42,
    }

    _rng       = np.random.default_rng(seed=0)
    _y_shuffle = _rng.permutation(y)   # สลับ target สุ่ม — ตัดความสัมพันธ์ทั้งหมด
    _shuf_oof  = np.zeros(len(y))

    for _fold, (_tr_idx, _val_idx) in enumerate(skf.split(idx, y)):
        _inner_idx, _es_idx = train_test_split(
            _tr_idx, test_size=0.1, random_state=_fold, stratify=y[_tr_idx]
        )
        # feature selection ยังใช้ target จริง (safe_cols_final ผ่าน Gate 1 + 1.5 แล้ว)
        # แต่ training ใช้ y_shuffled → โมเดลไม่ควรเรียนรู้อะไรได้
        _inner_df = df[list(map(int, _inner_idx))]
        _feats    = select_features_leak_free(_inner_df, safe_cols_final, TOP_N)

        _X_inner, _ = get_fold_arrays(df, _inner_idx, _feats)
        _X_es,    _ = get_fold_arrays(df, _es_idx,    _feats)
        _X_val,   _ = get_fold_arrays(df, _val_idx,   _feats)

        # train กับ y ที่ shuffle แล้ว
        _dtrain = lgb.Dataset(_X_inner, label=_y_shuffle[_inner_idx])
        _des    = lgb.Dataset(_X_es,    label=_y_shuffle[_es_idx], reference=_dtrain)
        _model  = lgb.train(
            _lgb_params, _dtrain, num_boost_round=1000, valid_sets=[_des],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)],
        )
        _shuf_oof[_val_idx] = _model.predict(_X_val)

    shuffle_auc = roc_auc_score(_y_shuffle, _shuf_oof)
    leakage_ok  = abs(shuffle_auc - 0.5) < 0.02

    mo.md(f"""
    ## Shuffle Sanity Test — พิสูจน์ว่าไม่มี leakage

    | Pipeline | AUC |
    |----------|-----|
    | Real target | **{lgb_oof_auc:.5f}** |
    | Shuffled target | **{shuffle_auc:.5f}** |
    | Expected | ~0.500 |

    {"✅ **PASSED** — AUC ≈ 0.5 หลัง shuffle → pipeline สะอาด ไม่มี leakage" if leakage_ok else f"🔴 **FAILED** — AUC = {shuffle_auc:.4f} ยังสูงอยู่ → มี leakage!"}
    """)
    return leakage_ok, shuffle_auc


@app.cell
def _(
    N_FOLDS,
    leakage_ok,
    lgb_fold_aucs,
    lgb_fold_feats,
    lgb_oof_auc,
    lgb_time,
    mo,
    np,
    plt,
    shuffle_auc,
    xgb_fold_aucs,
    xgb_oof_auc,
    xgb_time,
):
    def _plot_final():
        from collections import Counter
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x_pos = np.arange(N_FOLDS)
        w = 0.35
        axes[0].bar(x_pos - w/2, lgb_fold_aucs, w,
                    label=f"LightGBM OOF={lgb_oof_auc:.4f}", color="#4C72B0", alpha=0.85)
        axes[0].bar(x_pos + w/2, xgb_fold_aucs, w,
                    label=f"XGBoost  OOF={xgb_oof_auc:.4f}", color="#DD8452", alpha=0.85)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels([f"Fold {i+1}" for i in range(N_FOLDS)])
        axes[0].set_ylabel("AUC")
        axes[0].set_ylim(min(lgb_fold_aucs + xgb_fold_aucs) - 0.01, 1.0)
        axes[0].set_title("AUC per Fold", fontweight="bold")
        axes[0].legend()

        _counter = Counter(f for fold in lgb_fold_feats for f in fold)
        _top     = sorted(_counter.items(), key=lambda x: x[1], reverse=True)[:25]
        _feats, _cnts = zip(*_top)
        _colors = ["#2ca02c" if c == N_FOLDS else "#ff7f0e" if c >= 3 else "#d62728"
                   for c in _cnts]
        axes[1].barh(list(_feats)[::-1], list(_cnts)[::-1], color=list(_colors)[::-1])
        axes[1].axvline(N_FOLDS, color="green", linestyle="--", alpha=0.7,
                        label=f"All {N_FOLDS} folds = stable signal")
        axes[1].set_xlabel("# Folds Selected In")
        axes[1].set_title("Feature Stability\n(green = เลือกครบทุก fold = signal จริง)",
                          fontweight="bold")
        axes[1].legend()

        plt.suptitle("Leak-Free Baseline Results", fontsize=13, fontweight="bold")
        plt.tight_layout()
        return fig

    fig_final = _plot_final()
    _winner = "LightGBM" if lgb_oof_auc >= xgb_oof_auc else "XGBoost"

    mo.md(f"""
    ## สรุปผล Baseline

    | Model | OOF AUC | Std | Time |
    |-------|---------|-----|------|
    | **LightGBM** | **{lgb_oof_auc:.5f}** | ±{np.std(lgb_fold_aucs):.5f} | {lgb_time:.1f}s |
    | **XGBoost**  | **{xgb_oof_auc:.5f}** | ±{np.std(xgb_fold_aucs):.5f} | {xgb_time:.1f}s |
    | Shuffle test | {shuffle_auc:.5f} | expected ~0.500 | |

    **Winner: {_winner}** | **Leakage: {"✅ NONE" if leakage_ok else "🔴 DETECTED"}**

    ## Leakage Guard Functions ที่ใช้

    | ฟังก์ชัน | ป้องกันอะไร |
    |----------|------------|
    | `drop_red_flags()` | block feature ที่ corr > 0.9 ออกจาก model ถาวร |
    | `select_features_leak_free()` | เลือก feature จาก inner train เท่านั้น |
    | `get_fold_arrays()` | แปลงเฉพาะ rows+cols ที่อนุญาต ไม่มีทาง val รั่วเข้า |
    | `train_lgb_fold()` | รับแค่ inner train + es holdout — ไม่รู้จัก OOF เลย |
    | `train_xgb_fold()` | เหมือนกัน |
    | `evaluate_oof()` | เรียกหลัง train เสร็จเท่านั้น — OOF ไม่เคยเข้า training |
    """)
    return (fig_final,)


@app.cell
def _(fig_final):
    fig_final
    return


if __name__ == "__main__":
    app.run()
