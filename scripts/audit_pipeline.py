"""
Pipeline Audit Script
=====================
ตรวจสอบสุขภาพของ data pipeline ทั้งหมดใน 5 หมวด:

  Section 1 — File Existence   : ไฟล์ที่จำเป็นมีครบไหม?
  Section 2 — Row Counts       : จำนวน customers ถูกต้องไหม?
  Section 3 — Feature Counts   : features ถูก engineer ครบไหม?
  Section 4 — Data Quality     : duplicates, nulls, column alignment
  Section 5 — Leakage Gate     : feature_risk_scores ครอบคลุม features ปัจจุบันไหม?

รันได้ใน terminal:
  pixi run audit

Exit code:
  0 = ผ่านทุกอย่าง (pipeline พร้อม train)
  1 = มี critical issue (ต้องแก้ก่อน)
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import polars.selectors as cs

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
RAW  = ROOT / "data/raw"

# ── Report Helpers ────────────────────────────────────────────────────────────

FAILURES  = []   # critical — pipeline จะทำงานผิดพลาดถ้าไม่แก้
WARNINGS  = []   # non-critical — pipeline ยังรันได้ แต่ควรตรวจ

def ok(msg: str):
    print(f"  ✅  {msg}")

def warn(msg: str):
    print(f"  ⚠️   {msg}")
    WARNINGS.append(msg)

def fail(msg: str):
    print(f"  ❌  {msg}")
    FAILURES.append(msg)

def section(title: str):
    print(f"\n{'═'*54}")
    print(f"  {title}")
    print(f"{'═'*54}")

# ── Section 1: File Existence ─────────────────────────────────────────────────
# ตรวจว่าไฟล์ output ของแต่ละ pipeline step มีอยู่จริง
# ถ้าขาดไฟล์ใด step ถัดไปจะ crash ทันที → critical failure

section("SECTION 1 — File Existence")

# map: relative path → (description, is_critical)
REQUIRED_FILES = {
    "data/raw/train_data.csv":                    ("raw train CSV — จาก Kaggle",           True),
    "data/raw/test_data.csv":                     ("raw test CSV — จาก Kaggle",            True),
    "data/raw/train_labels.csv":                  ("raw labels — จาก Kaggle",              True),
    "data/processed/train_features.parquet":      ("train features — Step 01+02",          True),
    "data/processed/test_features.parquet":       ("test features — Step 01+02",           True),
    "data/processed/train_labels.parquet":        ("train labels parquet — Step 01",       True),
    "data/processed/feature_risk_scores.parquet": ("leakage gate scores — feature_gate.py", False),
}

file_status: dict[str, bool] = {}  # ใช้ check ว่าควร skip section ถัดไปไหม

for rel, (desc, critical) in REQUIRED_FILES.items():
    p = ROOT / rel
    exists = p.exists()
    file_status[rel] = exists
    if exists:
        mb = p.stat().st_size / (1024 ** 2)
        ok(f"{rel}  ({mb:,.0f} MB)")
    elif critical:
        fail(f"MISSING: {rel}  [{desc}]")
    else:
        warn(f"MISSING: {rel}  [{desc}]  — Gate 1.5 จะถูกข้าม")

# ── Section 2: Row Counts ─────────────────────────────────────────────────────
# ตรวจจำนวน customer ใน parquet ว่าตรงกับที่ Kaggle กำหนดไว้
# ถ้าไม่ตรง → preprocessing อาจ drop หรือ duplicate rows ไปโดยไม่รู้ตัว

section("SECTION 2 — Row Counts")

# ค่าเหล่านี้มาจากหน้า Kaggle competition
EXPECTED = {
    "train_features": 458_913,
    "test_features":  924_621,
    "train_labels":   458_913,
}

loaded: dict[str, pl.DataFrame] = {}  # เก็บไว้ใช้ใน section ถัดๆ ไป

for key, expected_n in EXPECTED.items():
    rel = f"data/processed/{key}.parquet"
    if not file_status.get(rel, False):
        warn(f"{key}: ข้ามเพราะไฟล์ไม่มี")
        continue
    df = pl.read_parquet(PROC / f"{key}.parquet")
    loaded[key] = df
    n = len(df)
    diff = abs(n - expected_n)
    if diff == 0:
        ok(f"{key}: {n:,} rows — ตรงทุกตัว")
    elif diff < 50:
        # tolerance เล็กน้อย เผื่อ Kaggle update dataset
        warn(f"{key}: {n:,} rows (expected {expected_n:,}, diff={diff}) — ตรวจสอบด้วย")
    else:
        fail(f"{key}: {n:,} rows แต่ expected {expected_n:,} — preprocessing มีปัญหา")

# ── Section 3: Feature Counts ─────────────────────────────────────────────────
# ตรวจว่า feature engineering แต่ละ phase ถูก run ครบหรือยัง
# แต่ละกลุ่มมี naming convention เฉพาะ → ตรวจจาก suffix ได้

section("SECTION 3 — Feature Counts")

if "train_features" in loaded:
    tf    = loaded["train_features"]
    cols  = tf.columns
    c_set = set(cols)

    # Phase 1 (01_preprocess.py): 7 numeric agg + 2 categorical agg
    orig = [c for c in cols if c.endswith((
        "_mean", "_std", "_min", "_max", "_last",
        "_last_minus_mean", "_last_minus_first", "_nunique",
    ))]
    if len(orig) >= 1_200:
        ok(f"Phase 1 — original agg features : {len(orig):,}  (expected ~1,261)")
    else:
        warn(f"Phase 1 — original agg features : {len(orig):,}  (expected ~1,261) — re-run 01_preprocess.py?")

    # Phase 2a (02_feature_engineering.py): ratio features
    ratio = [c for c in cols if c.endswith(("_last_div_mean", "_last_div_max", "_min_div_max"))]
    if len(ratio) >= 400:
        ok(f"Phase 2a — ratio features        : {len(ratio):,}")
    else:
        warn(f"Phase 2a — ratio features        : {len(ratio):,}  (expected ~530) — re-run 02_feature_engineering.py")

    # Phase 2b: lag features (เดือนที่ 2nd + 3rd จาก raw CSV)
    lag = [c for c in cols if c.endswith(("_lag2", "_lag3"))]
    if len(lag) >= 300:
        ok(f"Phase 2b — lag features           : {len(lag):,}")
    else:
        warn(f"Phase 2b — lag features           : {len(lag):,}  (expected ~354) — re-run 02_feature_engineering.py")

    # Phase 2c: slope features (finite difference approximation)
    slope = [c for c in cols if c.endswith("_slope")]
    if len(slope) >= 150:
        ok(f"Phase 2c — slope features         : {len(slope):,}")
    else:
        warn(f"Phase 2c — slope features         : {len(slope):,}  (expected ~177) — re-run 02_feature_engineering.py")

    # Phase 2d: count features (temporal depth per customer)
    count = [c for c in ("n_months", "n_delinquent_months") if c in c_set]
    if len(count) == 2:
        ok(f"Phase 2d — count features         : {count}")
    else:
        missing = [c for c in ("n_months", "n_delinquent_months") if c not in c_set]
        warn(f"Phase 2d — count features missing : {missing} — re-run 02_feature_engineering.py")

    # รวมทั้งหมด
    total = len([c for c in cols if c != "customer_ID"])
    print(f"\n  Total features in train_features : {total:,}")
else:
    warn("ข้ามเพราะ train_features.parquet ไม่มี")

# ── Section 4: Data Quality ───────────────────────────────────────────────────
# ตรวจ 4 เรื่องหลัก:
#   4a) Duplicate customer_ID → aggregate_customer() มีบั๊กไหม?
#   4b) Null rates → lag features จะ null ตาม design แต่ไม่ควรเกิน 50%
#   4c) Target distribution → default rate ควรอยู่ที่ ~26%
#   4d) Train/test column alignment → ถ้าต่างกันจะ predict ไม่ได้

section("SECTION 4 — Data Quality")

# 4a: Duplicate check
if "train_features" in loaded:
    tf     = loaded["train_features"]
    n_uniq = tf["customer_ID"].n_unique()
    if n_uniq == len(tf):
        ok(f"4a) No duplicate customer_IDs ({n_uniq:,} unique)")
    else:
        fail(f"4a) Duplicate customer_IDs: {len(tf):,} rows แต่ {n_uniq:,} unique — ตรวจ aggregate_customer()")

# 4b: Null rates (แยก original vs lag เพราะ lag null ตาม design)
if "train_features" in loaded:
    tf       = loaded["train_features"]
    num_cols = tf.select(cs.numeric()).columns
    null_ct  = tf.select(num_cols).null_count().row(0)
    null_arr = np.array(null_ct) / len(tf)

    # original features (no lag suffix) ไม่ควร null สูง
    orig_idx  = [i for i, c in enumerate(num_cols)
                 if not c.endswith(("_lag2", "_lag3", "_slope"))]
    lag_idx   = [i for i, c in enumerate(num_cols)
                 if c.endswith(("_lag2", "_lag3", "_slope"))]

    orig_null = null_arr[orig_idx].mean() if orig_idx else 0
    lag_null  = null_arr[lag_idx].mean()  if lag_idx  else 0

    if orig_null < 0.30:
        ok(f"4b) Original feature null rate : {orig_null:.1%} avg")
    else:
        warn(f"4b) Original feature null rate : {orig_null:.1%} avg — สูงกว่าปกติ")

    # lag null คาดว่า 5-15% (customers ที่มี < 2-3 months history)
    if lag_null < 0.50:
        ok(f"4b) Lag/slope feature null rate: {lag_null:.1%} avg  (ปกติ เพราะ short-history customers)")
    else:
        warn(f"4b) Lag/slope feature null rate: {lag_null:.1%} avg — สูงผิดปกติ ตรวจ stream_raw_for_lag()")

# 4c: Target distribution
if "train_labels" in loaded:
    default_rate = loaded["train_labels"]["target"].mean()
    if 0.24 <= default_rate <= 0.28:
        ok(f"4c) Default rate: {default_rate:.2%}  (expected ~26%)")
    else:
        warn(f"4c) Default rate: {default_rate:.2%}  — ผิดปกติ ตรวจ train_labels.csv")

# 4d: Column alignment between train and test
if "train_features" in loaded and "test_features" in loaded:
    train_cols = set(loaded["train_features"].columns)
    test_cols  = set(loaded["test_features"].columns)
    only_train = train_cols - test_cols - {"customer_ID"}
    only_test  = test_cols  - train_cols - {"customer_ID"}
    if not only_train and not only_test:
        ok(f"4d) Train/test columns match perfectly ({len(train_cols):,} cols)")
    else:
        if only_train:
            fail(f"4d) Columns in train ไม่มีใน test: {sorted(only_train)[:5]}...")
        if only_test:
            fail(f"4d) Columns in test ไม่มีใน train: {sorted(only_test)[:5]}...")

# ── Section 5: Leakage Gate Coverage ─────────────────────────────────────────
# ตรวจว่า feature_risk_scores.parquet ครอบคลุม features ปัจจุบันทั้งหมด
# ถ้า features ใหม่ (lag/ratio/slope) ยังไม่ถูก gate → Gate 1.5 จะข้ามไป
# ซึ่งหมายความว่า features ใหม่เหล่านั้นเข้า model โดยไม่ผ่านการตรวจ leakage

section("SECTION 5 — Leakage Gate Coverage")

gate_path = PROC / "feature_risk_scores.parquet"
if not gate_path.exists():
    warn("feature_risk_scores.parquet ไม่มี — รัน 'pixi run leakage' ก่อน")
else:
    risk        = pl.read_parquet(gate_path)
    gate_feats  = set(risk["feature"].to_list())

    # สรุป verdict distribution
    verdicts = risk["verdict"].value_counts().sort("verdict")
    for row in verdicts.iter_rows(named=True):
        v    = row["verdict"]
        cnt  = row["count"]
        icon = {"PASS": "✅", "WATCH": "⚠️ ", "BLOCK": "🚨"}.get(v, "  ")
        print(f"  {icon}  {v:5s}: {cnt:,} features")

    # ตรวจ coverage กับ features ที่มีอยู่จริงใน parquet
    if "train_features" in loaded:
        tf           = loaded["train_features"]
        current_num  = set(tf.select(cs.numeric()).columns)
        not_gated    = current_num - gate_feats

        if not not_gated:
            ok(f"Gate covers all {len(current_num):,} numeric features ✓")
        else:
            warn(
                f"{len(not_gated):,} features ไม่มีใน gate scores "
                f"(เช่น {sorted(not_gated)[:3]}) — "
                f"รัน 'pixi run leakage' เพื่อ re-evaluate"
            )

# ── Section 6: Feature Registry Sync ─────────────────────────────────────────
# ตรวจว่า feature_registry.json sync กับ train_features.parquet ไหม
# ถ้าไม่ sync → มี features ที่ยังไม่ถูก register หรือถูกลบออกไปแล้ว

section("SECTION 6 — Feature Registry Sync")

sys.path.insert(0, str(ROOT))
from scripts.feature_registry import (
    load_registry, check_sync, print_summary, REGISTRY_PATH
)

if not REGISTRY_PATH.exists():
    warn("feature_registry.json ยังไม่มี — รัน 01_preprocess.py และ 02_feature_engineering.py ก่อน")
else:
    # สรุปสถานะ registry
    print_summary()
    print()

    # ตรวจ sync กับ train_features.parquet
    if "train_features" in loaded:
        not_reg, ghosts = check_sync(PROC / "train_features.parquet")

        if not not_reg and not ghosts:
            ok(f"Registry sync กับ train_features.parquet สมบูรณ์ ✓")
        else:
            if not_reg:
                warn(
                    f"{len(not_reg):,} columns ใน parquet แต่ไม่มีใน registry — "
                    f"รัน 02_feature_engineering.py เพื่อ auto-register\n"
                    f"       ตัวอย่าง: {sorted(not_reg)[:3]}"
                )
            if ghosts:
                warn(
                    f"{len(ghosts):,} features อยู่ใน registry (active) แต่ไม่มีใน parquet — "
                    f"ถูกลบออกไปหรือยัง? ถ้าใช่ให้ deprecate_feature()\n"
                    f"       ตัวอย่าง: {sorted(ghosts)[:3]}"
                )
    else:
        warn("ข้าม registry sync — train_features.parquet ไม่มี")

# ── Final Summary ─────────────────────────────────────────────────────────────

section("AUDIT SUMMARY")

if FAILURES:
    print(f"\n  ❌  {len(FAILURES)} critical failure(s):")
    for msg in FAILURES:
        print(f"       • {msg}")
    print()
    if WARNINGS:
        print(f"  ⚠️   {len(WARNINGS)} warning(s):")
        for msg in WARNINGS:
            print(f"       • {msg}")
    print("\n  Pipeline is NOT ready.  แก้ failures ก่อน แล้วรัน audit อีกครั้ง\n")
    sys.exit(1)
else:
    if WARNINGS:
        print(f"\n  ⚠️   {len(WARNINGS)} warning(s) — pipeline ยังรันได้ แต่ควรตรวจ:")
        for msg in WARNINGS:
            print(f"       • {msg}")
    print("\n  ✅  All critical checks passed — pipeline is healthy.\n")
    sys.exit(0)
