import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import re
    import marimo as mo
    import polars as pl
    import polars.selectors as cs
    import numpy as np
    from pathlib import Path

    ROOT = Path(__file__).resolve().parent
    while not (ROOT / "pixi.toml").exists():
        ROOT = ROOT.parent

    mo.md("""
    # คู่มืออธิบาย Column ทั้งหมด
    ## American Express Default Prediction

    ตารางนี้อธิบายความหมายของทุก column ใน `train_features.parquet`
    ทั้ง features ดั้งเดิมและที่ engineer เพิ่มเติม

    ใช้ช่องค้นหาหรือ filter ด้านล่างเพื่อหา column ที่ต้องการ
    """)
    return ROOT, mo, np, pl, re, cs


@app.cell
def _(mo):
    # ── คำอธิบาย Group ────────────────────────────────────────────────────────
    mo.md("""
    ## กลุ่ม Feature (Feature Groups)

    ชื่อ column เริ่มต้นด้วยตัวอักษรที่บอก "ประเภทข้อมูล":

    | กลุ่ม | ย่อมาจาก | ความหมาย | ตัวอย่าง |
    |-------|----------|---------|---------|
    | **P** | Payment | ข้อมูลการชำระเงิน — ชำระเท่าไหร่ ชำระตรงเวลาไหม | `P_2`, `P_3` |
    | **D** | Delinquency | ข้อมูลการค้างชำระ — ค้างนานแค่ไหน ค้างบ่อยไหม | `D_39`, `D_41` |
    | **B** | Balance | ยอดคงค้างและวงเงิน — ใช้วงเงินไปเท่าไหร่ | `B_1`, `B_2` |
    | **S** | Spend | ข้อมูลการใช้จ่าย — ซื้ออะไร ใช้จ่ายเท่าไหร่ | `S_3`, `S_5` |
    | **R** | Risk | ตัวชี้วัดความเสี่ยงภายใน — credit bureau score ฯลฯ | `R_1`, `R_2` |

    ---

    ## ประเภท Suffix (Aggregation Types)

    แต่ละ feature ดั้งเดิมถูก aggregate จาก monthly rows → 1 row ต่อ customer
    ด้วย suffix ต่อไปนี้:

    ### Suffix จากการ Aggregate (01_preprocess.py)

    | Suffix | สูตร | ความหมาย | ตีความ |
    |--------|------|---------|--------|
    | `_mean` | avg ทุกเดือน | ค่าเฉลี่ยตลอดประวัติ | สถานะโดยรวม |
    | `_std` | std ทุกเดือน | ความผันผวน | ไม่นิ่ง = เสี่ยงกว่า |
    | `_min` | min ทุกเดือน | ค่าต่ำสุดที่เคยเป็น | worst case ที่เคยเห็น |
    | `_max` | max ทุกเดือน | ค่าสูงสุดที่เคยเป็น | peak ที่เคยเป็น |
    | `_last` | เดือนล่าสุด | สถานะปัจจุบัน | สำคัญที่สุด |
    | `_last_minus_mean` | last − mean | เทียบปัจจุบัน vs ค่าเฉลี่ย | + = แย่ลง, − = ดีขึ้น |
    | `_last_minus_first` | last − first | เปลี่ยนแปลงตั้งแต่ต้น | + = เพิ่มขึ้น, − = ลดลง |
    | `_last` (cat) | เดือนล่าสุด | category ปัจจุบัน | สถานะล่าสุด |
    | `_nunique` (cat) | count distinct | เคยเปลี่ยน category กี่ครั้ง | สูง = ไม่นิ่ง |

    ### Suffix จาก Feature Engineering (02_feature_engineering.py)

    | Suffix | สูตร | ความหมาย | ตีความ |
    |--------|------|---------|--------|
    | `_lag2` | เดือนที่ 2nd-last | ค่า 2 เดือนก่อน | ดู sequence กับ `_last` |
    | `_lag3` | เดือนที่ 3rd-last | ค่า 3 เดือนก่อน | ดู sequence กับ `_last` |
    | `_slope` | (last − lag3) / 2 | อัตราเปลี่ยนแปลงต่อเดือน | + = เพิ่มขึ้น, − = ลดลง |
    | `_last_div_mean` | last ÷ mean | สถานะปัจจุบัน vs ค่าเฉลี่ย | > 1 = แย่ลง |
    | `_last_div_max` | last ÷ max | ใกล้ peak แค่ไหน | → 1.0 = อยู่ที่จุดสูงสุด |
    | `_min_div_max` | min ÷ max | ช่วง compression | → 1.0 = ไม่ผันผวน |

    ### Features พิเศษ

    | Column | ความหมาย |
    |--------|---------|
    | `n_months` | จำนวนเดือนที่มีประวัติ (1–13) |
    | `n_delinquent_months` | จำนวนเดือนที่มี D column > 0 |
    | `delinquent_rate` | n_delinquent_months ÷ n_months (สัดส่วนค้างชำระ) |
    """)
    return


@app.cell
def _(ROOT, mo, np, pl, re, cs):
    # ── โหลดข้อมูลและคำนวณ stats ─────────────────────────────────────────────
    _df = pl.read_parquet(ROOT / "data/processed/train_features.parquet")
    _n  = len(_df)

    # กลุ่ม categorical (จาก preprocess)
    _CAT_BASES = {'B_30','B_38','D_114','D_116','D_117','D_120','D_126','D_63','D_64','D_66','D_68'}

    # mapping: suffix → ความหมายภาษาไทย
    SUFFIX_TH = {
        "_mean":            "ค่าเฉลี่ยทุกเดือน",
        "_std":             "ความผันผวน (standard deviation)",
        "_min":             "ค่าต่ำสุดตลอดประวัติ",
        "_max":             "ค่าสูงสุดตลอดประวัติ",
        "_last":            "ค่าเดือนล่าสุด (สถานะปัจจุบัน)",
        "_last_minus_mean": "เดือนล่าสุด ลบ ค่าเฉลี่ย (trend)",
        "_last_minus_first":"เดือนล่าสุด ลบ เดือนแรก (การเปลี่ยนแปลงทั้งหมด)",
        "_nunique":         "จำนวน category ที่เคยเห็น (categorical)",
        "_lag2":            "ค่า 2 เดือนก่อน",
        "_lag3":            "ค่า 3 เดือนก่อน",
        "_slope":           "อัตราเปลี่ยนแปลงต่อเดือน = (last − lag3) ÷ 2",
        "_last_div_mean":   "last ÷ mean — เทียบปัจจุบัน vs ค่าเฉลี่ยตัวเอง",
        "_last_div_max":    "last ÷ max — ใกล้จุดสูงสุดแค่ไหน",
        "_min_div_max":     "min ÷ max — ช่วงความผันผวน (→ 1 = ไม่ผันผวน)",
    }

    GROUP_TH = {
        "P": "Payment — การชำระเงิน",
        "D": "Delinquency — การค้างชำระ",
        "B": "Balance — ยอดคงค้าง/วงเงิน",
        "S": "Spend — การใช้จ่าย",
        "R": "Risk — ตัวชี้วัดความเสี่ยง",
    }

    # suffixes เรียงจากยาวสุดก่อน เพื่อ match ถูก
    _SUFFIXES = sorted(SUFFIX_TH.keys(), key=len, reverse=True)

    def parse_column(col: str):
        # special columns
        if col == "customer_ID":
            return ("—", "—", "—", "รหัสลูกค้า (unique identifier)", "special")
        if col == "n_months":
            return ("—", "—", "—", "จำนวนเดือนที่มีประวัติ (1–13)", "special")
        if col == "n_delinquent_months":
            return ("—", "—", "—", "จำนวนเดือนที่ค้างชำระ (D column > 0)", "special")
        if col == "delinquent_rate":
            return ("—", "—", "—", "สัดส่วนเดือนที่ค้างชำระ = n_delinquent_months ÷ n_months", "special")

        # standard: GROUP_NUMBER_SUFFIX
        for suf in _SUFFIXES:
            if col.endswith(suf):
                base = col[: -len(suf)]            # e.g. "P_2"
                m    = re.match(r"^([A-Z])_(\d+)$", base)
                if m:
                    grp  = m.group(1)
                    num  = m.group(2)
                    grp_th = GROUP_TH.get(grp, grp)
                    suf_th = SUFFIX_TH[suf]
                    phase  = "engineered" if suf in (
                        "_lag2","_lag3","_slope",
                        "_last_div_mean","_last_div_max","_min_div_max"
                    ) else "original"
                    return (grp, base, suf, f"{grp_th}  |  {suf_th}", phase)
        # fallback
        return ("?", col, "", col, "unknown")

    # คำนวณ stats: null%, mean, std สำหรับ numeric columns
    _num_cols = _df.select(cs.numeric()).columns
    _null_pct  = (_df.select(_num_cols).null_count() / _n * 100).row(0)
    _means     = _df.select([pl.col(c).mean() for c in _num_cols]).row(0)
    _stds      = _df.select([pl.col(c).std()  for c in _num_cols]).row(0)
    _num_stats = {c: (np, mn, sd) for c, np, mn, sd in
                  zip(_num_cols, _null_pct, _means, _stds)}

    # สร้าง rows
    rows = []
    for _col in _df.columns:
        if _col == "customer_ID":
            continue
        grp, base, suf, meaning, phase = parse_column(_col)
        np_val = _num_stats.get(_col, (None, None, None))
        rows.append({
            "column":    _col,
            "กลุ่ม":     grp,
            "base":      base,
            "suffix":    suf,
            "ความหมาย":  meaning,
            "null %":    f"{np_val[0]:.1f}%" if np_val[0] is not None else "—",
            "mean":      f"{np_val[1]:.4f}"  if np_val[1] is not None else "—",
            "std":       f"{np_val[2]:.4f}"  if np_val[2] is not None else "—",
            "phase":     phase,
        })

    import pandas as _pd
    table_df = _pd.DataFrame(rows)

    mo.md(f"""
    ## ตาราง Column ทั้งหมด ({len(rows):,} columns)

    ค้นหาได้ที่ช่อง search ด้านบนตาราง | กด header เพื่อ sort
    """)
    return (rows, table_df, SUFFIX_TH, GROUP_TH)


@app.cell
def _(mo, table_df):
    # ── Interactive Table ─────────────────────────────────────────────────────
    mo.ui.table(
        table_df,
        pagination=True,
        page_size=50,
        selection=None,
        label="ค้นหา column...",
    )
    return


@app.cell
def _(mo, table_df):
    # ── สรุปจำนวน column แต่ละกลุ่ม ──────────────────────────────────────────
    _summary = (
        table_df
        .groupby(["กลุ่ม", "phase"])
        .size()
        .reset_index(name="จำนวน")
        .sort_values(["กลุ่ม", "phase"])
    )

    _by_phase = table_df.groupby("phase").size().reset_index(name="จำนวน")

    mo.md(f"""
    ## สรุปจำนวน Features

    | Phase | จำนวน | มาจากไหน |
    |-------|-------|---------|
    | original | {_by_phase[_by_phase['phase']=='original']['จำนวน'].values[0]:,} | 01_preprocess.py |
    | engineered | {_by_phase[_by_phase['phase']=='engineered']['จำนวน'].values[0]:,} | 02_feature_engineering.py |
    | special | {_by_phase[_by_phase['phase']=='special']['จำนวน'].values[0]:,} | count/rate features |

    **รวมทั้งหมด: {len(table_df):,} columns**
    """)
    return


if __name__ == "__main__":
    app.run()
