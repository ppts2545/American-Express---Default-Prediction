#!/usr/bin/env python3
"""
Baseline Training Pipeline — LightGBM + XGBoost 5-fold Leak-Free CV

Usage:
    pixi run train
    python scripts/train_baseline.py

Gates:
    Gate 1  : drop features with |corr| > 0.9 (red flags)
    Gate 1.5: drop BLOCK features from leakage_detection pipeline
    Gate 2  : select top-N features per fold from inner train only
    Gate 3-5: train with no OOF data leakage
"""
from pathlib import Path
import time
import sys

import numpy as np
import polars as pl
import polars.selectors as cs
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

# ── ANSI ──────────────────────────────────────────────────────────────────────
R = "\033[91m"
Y = "\033[93m"
G = "\033[92m"
B = "\033[94m"
W = "\033[1m"
X = "\033[0m"

def _section(title: str) -> None:
    print(f"\n{W}{'─'*60}{X}")
    print(f"{W}  {title}{X}")
    print(f"{W}{'─'*60}{X}")


# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
while not (ROOT / "pixi.toml").exists():
    ROOT = ROOT.parent

TOP_N   = 30
N_FOLDS = 5

# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────
_section("Loading Data")
_t0 = time.time()

train_feat = pl.read_parquet(ROOT / "data/processed/train_features.parquet")
labels     = pl.read_parquet(ROOT / "data/processed/train_labels.parquet")
df         = train_feat.join(labels.select(["customer_ID", "target"]), on="customer_ID")

numeric_cols = df.select(cs.numeric().exclude("target")).columns
print(f"  rows={df.shape[0]:,}  numeric_cols={len(numeric_cols)}")
print(f"  target_rate={df['target'].mean():.3f}  loaded in {time.time()-_t0:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 1: Drop Red Flags (|corr| > 0.9)
# ─────────────────────────────────────────────────────────────────────────────
_section("Gate 1: Red Flag Filter  (|corr| > 0.9)")
_t = time.time()
_corr = (
    df.select([pl.corr(c, "target").alias(c) for c in numeric_cols])
    .unpivot(variable_name="feature", value_name="corr")
    .with_columns(pl.col("corr").abs().alias("abs_corr"))
    .sort("abs_corr", descending=True)
)
safe_cols    = _corr.filter(pl.col("abs_corr") <= 0.9)["feature"].to_list()
n_dropped_g1 = len(numeric_cols) - len(safe_cols)
print(f"  dropped={n_dropped_g1}  safe={len(safe_cols)}  ({time.time()-_t:.1f}s)")
if n_dropped_g1:
    for _r in _corr.filter(pl.col("abs_corr") > 0.9).iter_rows(named=True):
        print(f"    {R}DROP{X} {_r['feature']}  corr={_r['abs_corr']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 1.5: Feature Risk Filter (from leakage_detection pipeline)
# ─────────────────────────────────────────────────────────────────────────────
_section("Gate 1.5: Feature Risk Filter")
_risk_path = ROOT / "data/processed/feature_risk_scores.parquet"
if _risk_path.exists():
    _risk_df        = pl.read_parquet(_risk_path)
    _blocked        = set(_risk_df.filter(pl.col("verdict") == "BLOCK")["feature"].to_list())
    _watched        = set(_risk_df.filter(pl.col("verdict") == "WATCH")["feature"].to_list())
    safe_cols_final = [c for c in safe_cols if c not in _blocked]
    n_blocked       = sum(c in _blocked for c in safe_cols)
    n_watched       = sum(c in _watched for c in safe_cols_final)
    print(f"  {G}✅ feature_risk_scores.parquet loaded{X}")
    print(f"  safe (Gate 1)={len(safe_cols)}  blocked={n_blocked}  watched={n_watched}  "
          f"final={len(safe_cols_final)}")
    if n_blocked:
        for _f in sorted(_blocked & set(safe_cols)):
            _row = _risk_df.filter(pl.col("feature") == _f).to_dicts()[0]
            print(f"    {R}BLOCK{X} {_f:<30} score={_row['risk_score']}/9")
else:
    safe_cols_final = safe_cols
    print(f"  {Y}⚠️  feature_risk_scores.parquet not found{X}")
    print(f"  Run `pixi run leakage` first to enable Gate 1.5")
    print(f"  Using Gate 1 output: {len(safe_cols_final)} features")

# ─────────────────────────────────────────────────────────────────────────────
# Gate 2-5: LightGBM Leak-Free CV
# ─────────────────────────────────────────────────────────────────────────────
_section(f"LightGBM {N_FOLDS}-Fold Leak-Free CV  (top {TOP_N} features/fold)")

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

lgb_oof       = np.zeros(len(_y))
lgb_fold_aucs = []
_t0 = time.time()

for _fold, (_tr_idx, _val_idx) in enumerate(_skf.split(_idx, _y)):
    _ft = time.time()

    # Gate 2: เลือก feature จาก inner train เท่านั้น
    _inner_idx, _es_idx = train_test_split(
        _tr_idx, test_size=0.1, random_state=_fold, stratify=_y[_tr_idx]
    )
    _inner_df = df[list(map(int, _inner_idx))]
    _feats = (
        _inner_df
        .select([pl.corr(c, "target").alias(c) for c in safe_cols_final])
        .unpivot(variable_name="feature", value_name="corr")
        .with_columns(pl.col("corr").abs())
        .sort("corr", descending=True)
        .head(TOP_N)
    )["feature"].to_list()

    # Gate 3: แปลงเฉพาะ rows+cols ที่อนุญาต
    def _get(idx, cols):
        sub = df[list(map(int, idx))]
        return sub.select(cols).to_numpy(), sub["target"].to_numpy()

    X_inner, y_inner = _get(_inner_idx, _feats)
    X_es,    y_es    = _get(_es_idx,    _feats)
    X_val,   y_val   = _get(_val_idx,   _feats)

    # Gate 4: Train — ไม่รู้จัก val เลย
    dtrain  = lgb.Dataset(X_inner, label=y_inner)
    des     = lgb.Dataset(X_es,    label=y_es, reference=dtrain)
    model   = lgb.train(
        _lgb_params, dtrain, num_boost_round=1000,
        valid_sets=[des],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(False)],
    )

    # Gate 5: Evaluate บน OOF
    _preds             = model.predict(X_val)
    _auc               = roc_auc_score(y_val, _preds)
    lgb_oof[_val_idx]  = _preds
    lgb_fold_aucs.append(_auc)

    _status = G if _auc >= 0.80 else Y if _auc >= 0.75 else R
    print(f"  Fold {_fold+1}/{N_FOLDS}  AUC={_status}{_auc:.5f}{X}  ({time.time()-_ft:.1f}s)")

lgb_oof_auc = roc_auc_score(_y, lgb_oof)
lgb_time    = time.time() - _t0

_status = G if lgb_oof_auc >= 0.80 else Y if lgb_oof_auc >= 0.75 else R
print(f"\n  {W}OOF AUC = {_status}{lgb_oof_auc:.5f}{X}  total={lgb_time:.1f}s")
print(f"  std = {np.std(lgb_fold_aucs):.5f}  min={min(lgb_fold_aucs):.5f}  max={max(lgb_fold_aucs):.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
_section("Summary")
print(f"  Features (total)   : {len(numeric_cols)}")
print(f"  After Gate 1       : {len(safe_cols)}  (dropped {n_dropped_g1} red flags)")
print(f"  After Gate 1.5     : {len(safe_cols_final)}  (blocked {len(numeric_cols)-len(safe_cols_final)-n_dropped_g1} leakage suspects)")
print(f"  Per-fold feature   : top {TOP_N}")
print(f"  LGB OOF AUC       : {_status}{lgb_oof_auc:.5f}{X}")
print()
