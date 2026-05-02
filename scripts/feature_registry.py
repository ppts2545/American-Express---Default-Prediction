"""
Feature Registry Helper
=======================
ติดตามว่า feature ไหนถูกเพิ่มเมื่อไหร่ มาจาก step ไหน และสถานะปัจจุบันคืออะไร

Registry เก็บอยู่ที่ feature_registry.json ในโฟลเดอร์ project root
รูปแบบ:
{
  "P_2_mean": {
    "added":  "2026-05-03",   ← วันที่ register อัตโนมัติ
    "phase":  "original",     ← original / engineered / special
    "reason": "01_preprocess.py — aggregate monthly rows",
    "status": "active"        ← active / deprecated
  },
  ...
}

ใช้งาน:
  from scripts.feature_registry import update_registry, deprecate_feature

  # หลังเขียน parquet เสร็จ
  update_registry(parquet_path, phase="engineered", reason="02_feature_engineering.py")

  # ถ้าจะเลิกใช้ feature (ไม่ลบ เก็บ history ไว้)
  deprecate_feature("P_2_lag2", reason="noise มากกว่า signal จาก leakage gate")
"""

import json
import sys
from datetime import date
from pathlib import Path

import polars as pl
import polars.selectors as cs

# ── Paths ─────────────────────────────────────────────────────────────────────

def _find_root() -> Path:
    """หา project root โดยมองหา pixi.toml ขึ้นไปเรื่อยๆ"""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "pixi.toml").exists():
            return p
        p = p.parent
    raise FileNotFoundError("ไม่พบ pixi.toml — รัน script จาก project directory")

ROOT          = _find_root()
REGISTRY_PATH = ROOT / "feature_registry.json"

# ── Core Functions ────────────────────────────────────────────────────────────

def load_registry() -> dict:
    """โหลด registry จาก JSON — ถ้ายังไม่มีไฟล์ return dict ว่าง"""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_registry(registry: dict) -> None:
    """บันทึก registry กลับเป็น JSON (sorted key เพื่ออ่านง่าย)"""
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False,
                  sort_keys=True)


def update_registry(
    parquet_path: str | Path,
    phase: str,
    reason: str,
    exclude_cols: set[str] | None = None,
) -> int:
    """
    เปรียบเทียบ columns ใน parquet กับ registry ที่มีอยู่
    columns ที่ยังไม่มีใน registry จะถูก register อัตโนมัติพร้อมวันที่วันนี้

    Parameters
    ----------
    parquet_path : path ของ parquet file ที่ต้องการ register
    phase        : "original" | "engineered" | "special"
    reason       : อธิบายว่า feature นี้มาจากไหน เพื่ออะไร
    exclude_cols : columns ที่ไม่ต้องการ register (default: customer_ID, target)

    Returns
    -------
    n_new : จำนวน features ใหม่ที่ถูก register รอบนี้
    """
    if exclude_cols is None:
        exclude_cols = {"customer_ID", "target"}

    # อ่าน columns จาก parquet (ไม่โหลดข้อมูล — เร็วมาก)
    df      = pl.read_parquet(parquet_path, n_rows=0)
    cols    = [c for c in df.columns if c not in exclude_cols]

    registry = load_registry()
    today    = str(date.today())
    n_new    = 0

    for col in cols:
        if col not in registry:
            # column ใหม่ → register อัตโนมัติ
            registry[col] = {
                "added":  today,
                "phase":  phase,
                "reason": reason,
                "status": "active",
            }
            n_new += 1

    save_registry(registry)
    return n_new


def deprecate_feature(feature_name: str, reason: str = "") -> bool:
    """
    Mark feature ว่าไม่ใช้แล้ว (deprecated) — ไม่ลบออก เพื่อเก็บ history

    Parameters
    ----------
    feature_name : ชื่อ column ที่ต้องการ deprecate
    reason       : เหตุผลที่ deprecate

    Returns
    -------
    True ถ้าสำเร็จ, False ถ้าไม่พบ feature
    """
    registry = load_registry()
    if feature_name not in registry:
        print(f"ไม่พบ '{feature_name}' ใน registry")
        return False

    registry[feature_name]["status"]            = "deprecated"
    registry[feature_name]["deprecated_date"]   = str(date.today())
    if reason:
        registry[feature_name]["deprecated_reason"] = reason

    save_registry(registry)
    print(f"Deprecated: {feature_name}")
    return True


def check_sync(parquet_path: str | Path) -> tuple[set, set]:
    """
    ตรวจสอบว่า registry sync กับ parquet ไหม

    Returns
    -------
    not_registered : columns ใน parquet แต่ไม่มีใน registry
    ghost_features : columns ใน registry แต่ไม่มีใน parquet (อาจถูกลบไป)
    """
    df           = pl.read_parquet(parquet_path, n_rows=0)
    parquet_cols = set(df.columns) - {"customer_ID", "target"}
    registry     = load_registry()
    reg_active   = {k for k, v in registry.items() if v.get("status") == "active"}

    not_registered = parquet_cols - set(registry.keys())
    ghost_features = reg_active   - parquet_cols

    return not_registered, ghost_features


def print_summary() -> None:
    """Print สรุปสถานะ registry ทั้งหมด"""
    registry = load_registry()
    if not registry:
        print("  Registry ว่างเปล่า — รัน 01_preprocess.py และ 02_feature_engineering.py ก่อน")
        return

    by_phase  = {}
    by_status = {}
    for v in registry.values():
        by_phase[v["phase"]]   = by_phase.get(v["phase"],   0) + 1
        by_status[v["status"]] = by_status.get(v["status"], 0) + 1

    print(f"  Total registered : {len(registry):,} features")
    for phase, cnt in sorted(by_phase.items()):
        print(f"    {phase:12s}: {cnt:,}")
    print()
    for status, cnt in sorted(by_status.items()):
        icon = "✅" if status == "active" else "🗄️ "
        print(f"  {icon}  {status:12s}: {cnt:,}")


# ── CLI ───────────────────────────────────────────────────────────────────────
# รันตรงๆ ได้: python scripts/feature_registry.py

if __name__ == "__main__":
    print("\n── Feature Registry Summary ──────────────────────")
    print_summary()
    print()
