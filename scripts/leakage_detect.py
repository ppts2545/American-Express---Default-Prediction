#!/usr/bin/env python3
"""
Leakage Detection Pipeline — 9 techniques (headless, no UI)

Usage:
    pixi run leakage
    python scripts/leakage_detect.py

Output:
    data/processed/feature_risk_scores.parquet   ← Gate 1.5 ใน baseline อ่านตัวนี้
    data/processed/risk_runs/risk_<ts>.parquet   ← history สำหรับ compare runs

Technique order (Fast → Temporal → Medium → Slow):
    T1: Null Pattern          ⚡ O(n)
    T2: Class Separation      ⚡ O(n) vectorized
    T3: Mutual Information    ⚡ O(n log n)
    T4: Variance Ratio        ⚡ O(n) vectorized
    T5: PSI                   🕐 full data
    T6: Time Stability        🕐 full data
    T7: Single-Feature AUC    🐢 O(n log n) + LGB
    T8: Adversarial T/T       🔥 LGB training
    T9: Adversarial T/Test    🔥 LGB training
"""
from pathlib import Path
import tomllib
import time
import datetime as dt

import numpy as np
import polars as pl
import polars.selectors as cs
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif

# ── ANSI colors ───────────────────────────────────────────────────────────────
R  = "\033[91m"
Y  = "\033[93m"
G  = "\033[92m"
B  = "\033[94m"
W  = "\033[1m"
X  = "\033[0m"


def _log(step: int, name: str, elapsed: float = 0.0, detail: str = "") -> None:
    badge = f"{B}[{step}/9]{X}"
    t     = f"{Y}{elapsed:.1f}s{X}" if elapsed else ""
    sep   = f" | {t}" if elapsed else ""
    print(f"{badge} {W}{name}{X}{sep}  {detail}")


def _section(title: str) -> None:
    print(f"\n{W}{'─'*60}{X}")
    print(f"{W}  {title}{X}")
    print(f"{W}{'─'*60}{X}")


# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
while not (ROOT / "pixi.toml").exists():
    ROOT = ROOT.parent

# ── Config ────────────────────────────────────────────────────────────────────
with open(ROOT / "config" / "leakage_thresholds.toml", "rb") as _f:
    cfg = tomllib.load(_f)

_BLOCK = cfg["risk_score"]["block_threshold"]
_WATCH = cfg["risk_score"]["watch_threshold"]

# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────
_section("Loading Data")
_t0 = time.time()

train_feat = pl.read_parquet(ROOT / "data/processed/train_features.parquet")
labels     = pl.read_parquet(ROOT / "data/processed/train_labels.parquet")
df         = train_feat.join(labels.select(["customer_ID", "target"]), on="customer_ID")

numeric_cols = df.select(cs.numeric().exclude("target")).columns

SAMPLE_N  = min(50_000, df.shape[0])
df_sample = df.sample(n=SAMPLE_N, seed=42)

print(f"  rows={df.shape[0]:,}  cols={df.shape[1]}  numeric={len(numeric_cols)}")
print(f"  sample={SAMPLE_N:,}  target_rate={df['target'].mean():.3f}")

# ── Gate 0: drop corr > 0.9 ──────────────────────────────────────────────────
_corr = (
    df_sample
    .select([pl.corr(c, "target").alias(c) for c in numeric_cols])
    .unpivot(variable_name="feature", value_name="corr")
    .with_columns(pl.col("corr").abs().alias("abs_corr"))
    .sort("abs_corr", descending=True)
)
red_flag_set   = set(_corr.filter(pl.col("abs_corr") > 0.9)["feature"].to_list())
all_check_cols = [c for c in numeric_cols if c not in red_flag_set]
print(f"  red_flag (corr>0.9)={len(red_flag_set)}  checking={len(all_check_cols)}")

_df0      = df_sample.filter(pl.col("target") == 0)
_df1      = df_sample.filter(pl.col("target") == 1)
_y_sample = df_sample["target"].to_numpy()
_X_sample = df_sample.select(all_check_cols).fill_null(0).to_numpy()

print(f"  loaded in {time.time()-_t0:.1f}s\n")

# ─────────────────────────────────────────────────────────────────────────────
# T1: Null Pattern Analysis
# ─────────────────────────────────────────────────────────────────────────────
_t = time.time()
_null0 = np.array([_df0[c].is_null().mean() for c in all_check_cols])
_null1 = np.array([_df1[c].is_null().mean() for c in all_check_cols])
_diff  = np.abs(_null1 - _null0)
null_scores     = dict(zip(all_check_cols, _diff))
null_flagged    = {k: v for k, v in null_scores.items() if v > cfg["null_pattern"]["flagged_diff"]}
null_suspicious = {k: v for k, v in null_scores.items()
                   if cfg["null_pattern"]["suspicious_diff"] < v <= cfg["null_pattern"]["flagged_diff"]}
_log(1, "Null Pattern Analysis", time.time()-_t,
     f"🔴 {len(null_flagged)} flagged  🟠 {len(null_suspicious)} suspicious")

# ─────────────────────────────────────────────────────────────────────────────
# T2: Class Separation (Cohen's d)
# ─────────────────────────────────────────────────────────────────────────────
_t   = time.time()
_arr = _X_sample.copy()
_p1  = np.percentile(_arr, 1,  axis=0)
_p99 = np.percentile(_arr, 99, axis=0)
_clipped = np.clip(_arr, _p1, _p99)
_mask0   = (_y_sample == 0)
_mask1   = (_y_sample == 1)
_arr0, _arr1 = _clipped[_mask0], _clipped[_mask1]
_m0, _m1 = _arr0.mean(0), _arr1.mean(0)
_s0, _s1 = _arr0.std(0),  _arr1.std(0)
_pooled  = np.sqrt((_s0**2 + _s1**2) / 2) + 1e-8
d_vals   = np.abs(_m1 - _m0) / _pooled
cohen_d_scores = dict(zip(all_check_cols, d_vals))
flagged_sep    = {k: v for k, v in cohen_d_scores.items() if v > cfg["class_separation"]["flagged_d"]}
watchlist_sep  = {k: v for k, v in cohen_d_scores.items()
                  if cfg["class_separation"]["watchlist_d"] < v <= cfg["class_separation"]["flagged_d"]}
_log(2, "Class Separation (Cohen's d)", time.time()-_t,
     f"🔴 {len(flagged_sep)} flagged  🟠 {len(watchlist_sep)} watchlist")

# ─────────────────────────────────────────────────────────────────────────────
# T3: Mutual Information
# ─────────────────────────────────────────────────────────────────────────────
_t = time.time()
_mi_arr   = mutual_info_classif(_X_sample, _y_sample, discrete_features=False, random_state=42)
mi_scores = dict(zip(all_check_cols, _mi_arr))
mi_flagged    = {k: v for k, v in mi_scores.items() if v > cfg["mutual_information"]["flagged"]}
mi_suspicious = {k: v for k, v in mi_scores.items()
                 if cfg["mutual_information"]["suspicious"] < v <= cfg["mutual_information"]["flagged"]}
_log(3, "Mutual Information", time.time()-_t,
     f"🔴 {len(mi_flagged)} flagged  🟠 {len(mi_suspicious)} suspicious")

# ─────────────────────────────────────────────────────────────────────────────
# T4: Variance Ratio
# ─────────────────────────────────────────────────────────────────────────────
_t       = time.time()
_arr_all = _X_sample.copy()
_p1v     = np.percentile(_arr_all, 1,  axis=0)
_p99v    = np.percentile(_arr_all, 99, axis=0)
_clippedv = np.clip(_arr_all, _p1v, _p99v)
_arr0v, _arr1v = _clippedv[_mask0], _clippedv[_mask1]
_std0 = _arr0v.std(axis=0) + 1e-8
_std1 = _arr1v.std(axis=0) + 1e-8
_ratio = np.maximum(_std1 / _std0, _std0 / _std1)
var_scores     = dict(zip(all_check_cols, _ratio))
var_flagged    = {k: v for k, v in var_scores.items() if v > cfg["variance_ratio"]["flagged"]}
var_suspicious = {k: v for k, v in var_scores.items()
                  if cfg["variance_ratio"]["suspicious"] < v <= cfg["variance_ratio"]["flagged"]}
_log(4, "Variance Ratio", time.time()-_t,
     f"🔴 {len(var_flagged)} flagged  🟠 {len(var_suspicious)} suspicious")

# ─────────────────────────────────────────────────────────────────────────────
# T5: PSI (Population Stability Index)
# ─────────────────────────────────────────────────────────────────────────────
_t        = time.time()
N_BUCKETS = cfg["psi"]["buckets"]
_n_half   = df.shape[0] // 2
_df_early = df.head(_n_half)
_df_late  = df.tail(df.shape[0] - _n_half)


def _psi_one(col: str) -> float:
    _exp = _df_early[col].drop_nulls().to_numpy()
    _act = _df_late[col].drop_nulls().to_numpy()
    if len(_exp) < 2 or len(_act) < 2:
        return 0.0
    _breaks = np.unique(np.percentile(_exp, np.linspace(0, 100, N_BUCKETS + 1)))
    if len(_breaks) < 2:
        return 0.0
    _e_cnt = np.histogram(_exp, bins=_breaks)[0].astype(float)
    _a_cnt = np.histogram(_act, bins=_breaks)[0].astype(float)
    _e_pct = np.clip(_e_cnt / _e_cnt.sum(), 1e-6, None)
    _a_pct = np.clip(_a_cnt / _a_cnt.sum(), 1e-6, None)
    return float(np.sum((_a_pct - _e_pct) * np.log(_a_pct / _e_pct)))


psi_scores    = {c: _psi_one(c) for c in all_check_cols}
psi_flagged   = {k: v for k, v in psi_scores.items() if v > cfg["psi"]["unstable"]}
psi_suspicious = {k: v for k, v in psi_scores.items()
                  if cfg["psi"]["monitor"] < v <= cfg["psi"]["unstable"]}
_log(5, "PSI (Population Stability Index)", time.time()-_t,
     f"🔴 {len(psi_flagged)} unstable  🟠 {len(psi_suspicious)} monitor")

# ─────────────────────────────────────────────────────────────────────────────
# T6: Time Stability (gap_cv)
# ─────────────────────────────────────────────────────────────────────────────
_t = time.time()
n = df.shape[0]
_df_t = df.with_row_index("__ridx").with_columns(
    pl.when(pl.col("__ridx") < n // 4).then(pl.lit("Q1"))
    .when(pl.col("__ridx") < n // 2).then(pl.lit("Q2"))
    .when(pl.col("__ridx") < 3 * n // 4).then(pl.lit("Q3"))
    .otherwise(pl.lit("Q4"))
    .alias("__time_bin")
)
_agg = (
    _df_t.group_by(["__time_bin", "target"])
    .agg([pl.col(c).mean().alias(c) for c in all_check_cols])
    .sort(["__time_bin", "target"])
)
_t0_agg      = _agg.filter(pl.col("target") == 0)
_t1_agg      = _agg.filter(pl.col("target") == 1)
unstable_cols = []
gap_cvs       = {}
for _c in all_check_cols:
    _gaps   = [abs(a - b) for a, b in zip(_t0_agg[_c].fill_null(0), _t1_agg[_c].fill_null(0))]
    _gap_cv = float(np.std(_gaps) / (np.mean(_gaps) + 1e-8))
    gap_cvs[_c] = _gap_cv
    if _gap_cv > cfg["time_stability"]["unstable_gap_cv"]:
        unstable_cols.append(_c)
_log(6, "Time Stability (gap_cv)", time.time()-_t,
     f"🔴 {len(unstable_cols)} unstable")

# ─────────────────────────────────────────────────────────────────────────────
# T7: Single-Feature AUC (2-pass: raw → LGB)
# ─────────────────────────────────────────────────────────────────────────────
_t   = time.time()
_raw = np.array([roc_auc_score(_y_sample, _X_sample[:, i]) for i in range(_X_sample.shape[1])])
_raw = np.maximum(_raw, 1 - _raw)
raw_auc_scores = dict(zip(all_check_cols, _raw))

_sus_idx   = [i for i, v in enumerate(_raw) if v > cfg["single_feat_auc"]["screen_at"]]
_sf_params = {
    "objective": "binary", "metric": "auc",
    "num_leaves": 7, "max_depth": 3,
    "learning_rate": 0.1, "min_child_samples": 50,
    "verbosity": -1, "random_state": 42,
}
_skf3b     = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
lgb_auc_scores = {}
for _i in _sus_idx:
    _feat = all_check_cols[_i]
    _x1   = _X_sample[:, _i: _i + 1]
    _oof  = np.zeros(len(_y_sample))
    for _tr, _val in _skf3b.split(_x1, _y_sample):
        _dtrain = lgb.Dataset(_x1[_tr], label=_y_sample[_tr])
        _m = lgb.train(
            _sf_params, _dtrain, num_boost_round=100,
            callbacks=[lgb.log_evaluation(False)],
        )
        _oof[_val] = _m.predict(_x1[_val])
    lgb_auc_scores[_feat] = roc_auc_score(_y_sample, _oof)

sf_scores     = {f: max(raw_auc_scores[f], lgb_auc_scores.get(f, 0)) for f in all_check_cols}
sf_flagged    = {k: v for k, v in sf_scores.items() if v > cfg["single_feat_auc"]["flagged"]}
sf_suspicious = {k: v for k, v in sf_scores.items()
                 if cfg["single_feat_auc"]["suspicious"] < v <= cfg["single_feat_auc"]["flagged"]}
_log(7, "Single-Feature AUC", time.time()-_t,
     f"🔴 {len(sf_flagged)} flagged  🟠 {len(sf_suspicious)} suspicious  "
     f"(LGB on {len(_sus_idx)} candidates)")

# ─────────────────────────────────────────────────────────────────────────────
# T8: Adversarial Validation (Train/Train)
# ─────────────────────────────────────────────────────────────────────────────
_t = time.time()
_adv_y              = np.zeros(SAMPLE_N, dtype=int)
_adv_y[SAMPLE_N // 2:] = 1
_adv_params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 31,
    "verbosity": -1, "random_state": 42,
}
_skf3    = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
_adv_oof = np.zeros(SAMPLE_N)
_adv_imp = np.zeros(len(all_check_cols))
for _tr, _val in _skf3.split(_X_sample, _adv_y):
    _dtrain = lgb.Dataset(_X_sample[_tr], label=_adv_y[_tr])
    _dval   = lgb.Dataset(_X_sample[_val], label=_adv_y[_val], reference=_dtrain)
    _m = lgb.train(
        _adv_params, _dtrain, num_boost_round=300, valid_sets=[_dval],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(False)],
    )
    _adv_oof[_val] = _m.predict(_X_sample[_val])
    _adv_imp      += _m.feature_importance(importance_type="gain")
adv_auc  = roc_auc_score(_adv_y, _adv_oof)
adv_ok   = adv_auc < cfg["adversarial"]["suspicious_auc"]
_top_idx = np.argsort(_adv_imp)[::-1][:5]
adv_top5 = [all_check_cols[i] for i in _top_idx]
_log(8, "Adversarial (Train/Train)", time.time()-_t,
     f"AUC={adv_auc:.4f} {'✅ stationary' if adv_ok else '🔴 non-stationary'}")

# ─────────────────────────────────────────────────────────────────────────────
# T9: Adversarial Validation (Train/Test)
# ─────────────────────────────────────────────────────────────────────────────
_t = time.time()
_test_path = ROOT / "data/processed/test_features.parquet"
if _test_path.exists():
    _test        = pl.read_parquet(_test_path)
    _test_sample = _test.sample(n=min(SAMPLE_N, _test.shape[0]), seed=42)
    _X_test      = _test_sample.select(all_check_cols).to_numpy()
    _X_tt        = np.vstack([_X_sample, _X_test])
    _is_test     = np.concatenate([np.zeros(len(_X_sample)), np.ones(len(_X_test))]).astype(int)
    _tt_oof      = np.zeros(len(_X_tt))
    _tt_imp      = np.zeros(len(all_check_cols))
    for _tr, _val in _skf3.split(_X_tt, _is_test):
        _dtrain = lgb.Dataset(_X_tt[_tr], label=_is_test[_tr])
        _dval   = lgb.Dataset(_X_tt[_val], label=_is_test[_val], reference=_dtrain)
        _m = lgb.train(
            _adv_params, _dtrain, num_boost_round=300, valid_sets=[_dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(False)],
        )
        _tt_oof[_val] = _m.predict(_X_tt[_val])
        _tt_imp      += _m.feature_importance(importance_type="gain")
    adv_tt_auc  = roc_auc_score(_is_test, _tt_oof)
    adv_tt_ok   = adv_tt_auc < cfg["adversarial"]["suspicious_auc"]
    _tt_top_idx = np.argsort(_tt_imp)[::-1][:5]
    adv_tt_top5 = [all_check_cols[i] for i in _tt_top_idx]
    _log(9, "Adversarial (Train/Test)", time.time()-_t,
         f"AUC={adv_tt_auc:.4f} {'✅ no drift' if adv_tt_ok else '🔴 drift detected'}")
else:
    adv_tt_auc, adv_tt_ok, adv_tt_top5 = None, None, []
    _log(9, "Adversarial (Train/Test)", time.time()-_t,
         "⚠️ test_features.parquet not found — skipped")

# ─────────────────────────────────────────────────────────────────────────────
# Build Risk Score & Save
# ─────────────────────────────────────────────────────────────────────────────
_section("Building Risk Scores & Saving")

_null_set    = set(null_flagged) | set(null_suspicious)
_sep_set     = set(flagged_sep)
_mi_set      = set(mi_flagged) | set(mi_suspicious)
_var_set     = set(var_flagged) | set(var_suspicious)
_psi_set     = set(psi_flagged) | set(psi_suspicious)
_ts_set      = set(unstable_cols)
_sf_set      = set(sf_flagged) | set(sf_suspicious)
_adv_self_set = set(adv_top5)
_adv_tt_set  = set(adv_tt_top5)

_rows = []
for _feat in all_check_cols:
    _f1 = _feat in _null_set        # T1 Null Pattern
    _f2 = _feat in _sep_set         # T2 Class Separation
    _f3 = _feat in _mi_set          # T3 Mutual Information
    _f4 = _feat in _var_set         # T4 Variance Ratio
    _f5 = _feat in _psi_set         # T5 PSI
    _f6 = _feat in _ts_set          # T6 Time Stability
    _f7 = _feat in _sf_set          # T7 Single-Feature AUC
    _f8 = _feat in _adv_self_set    # T8 Adversarial T/T
    _f9 = _feat in _adv_tt_set      # T9 Adversarial T/Test
    _score      = sum([_f1, _f2, _f3, _f4, _f5, _f6, _f7, _f8, _f9])
    _watch_flag = _feat in set(watchlist_sep)
    _verdict    = "BLOCK" if _score >= _BLOCK else "WATCH" if (_score >= _WATCH or _watch_flag) else "CLEAN"
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
        "verdict":              _verdict,
    })

_ts     = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
risk_df = pl.DataFrame(_rows).sort("risk_score", descending=True)

_out      = ROOT / "data/processed/feature_risk_scores.parquet"
_runs_dir = ROOT / "data/processed/risk_runs"
_runs_dir.mkdir(exist_ok=True)

risk_df.write_parquet(_out)
risk_df.with_columns(pl.lit(_ts).alias("run_id")).write_parquet(_runs_dir / f"risk_{_ts}.parquet")

# compare กับ run ก่อนหน้า
_prev_runs = sorted(r for r in _runs_dir.glob("risk_*.parquet") if r.stem != f"risk_{_ts}")
if _prev_runs:
    _prev    = pl.read_parquet(_prev_runs[-1])
    _prev_lk = dict(zip(_prev["feature"].to_list(), _prev["verdict"].to_list()))
    _changes = [
        (r["feature"], _prev_lk[r["feature"]], r["verdict"])
        for r in risk_df.iter_rows(named=True)
        if r["feature"] in _prev_lk and _prev_lk[r["feature"]] != r["verdict"]
    ]
    if _changes:
        print(f"\n  {W}∆ Changes from previous run:{X}")
        for _feat, _old, _new in _changes:
            _arrow = "⬆️" if _new == "BLOCK" else "⬇️"
            print(f"    {_feat}: {_old} → {_new} {_arrow}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
n_block = (risk_df["verdict"] == "BLOCK").sum()
n_watch = (risk_df["verdict"] == "WATCH").sum()
n_clean = (risk_df["verdict"] == "CLEAN").sum()

_section("Summary")
print(f"  Run:  {_ts}  |  Techniques: 9  |  Features checked: {len(all_check_cols)}")
print(f"\n  {'Verdict':<10} {'Count':>6}  {'Threshold'}")
print(f"  {'─'*40}")
print(f"  {R}BLOCK{X}      {n_block:>6}   risk_score ≥ {_BLOCK}/9")
print(f"  {Y}WATCH{X}      {n_watch:>6}   risk_score ≥ {_WATCH}/9 or in watchlist")
print(f"  {G}CLEAN{X}      {n_clean:>6}   passed all 9 techniques")

if n_block > 0:
    _top_block = risk_df.filter(pl.col("verdict") == "BLOCK").head(10)
    print(f"\n  Top BLOCK features (score/9):")
    for _r in _top_block.iter_rows(named=True):
        print(f"    {R}{_r['feature']:<30}{X}  score={_r['risk_score']}/9")

print(f"\n  {G}✅ Saved:{X}")
print(f"     {_out}")
print(f"     {_runs_dir / f'risk_{_ts}.parquet'}")
print(f"\n  Gate 1.5 in baseline.py will auto-block {n_block} features\n")
