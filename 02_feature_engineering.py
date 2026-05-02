import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import gc
    import marimo as mo
    import polars as pl
    import numpy as np
    import pandas as pd
    from pathlib import Path

    PROC_DIR = Path('data/processed')
    RAW_DIR  = Path('data/raw')

    CAT_COLS = {'B_30','B_38','D_114','D_116','D_117','D_120','D_126','D_63','D_64','D_66','D_68'}
    SKIP_COLS = {'customer_ID', 'S_2'} | CAT_COLS

    mo.md("""
    # 02 — Feature Engineering

    เพิ่ม **4 กลุ่ม feature** ให้กับ processed features เพื่อให้ถึงระดับ Top Kaggle:

    | กลุ่ม | จำนวน | วิธี | เวลา |
    |-------|-------|------|------|
    | **Ratio** | ~500 | last/mean, last/max, min/max | < 1 นาที |
    | **Lag** | ~354 | lag2, lag3 (เดือนก่อนหน้า) | 5-10 นาที |
    | **Count** | 2 | n_months, n_delinquent_months | (รวมกับ lag pass) |
    | **Slope** | ~177 | (lag1-lag3)/2 finite difference | (รวมกับ lag pass) |

    ```
    data/processed/train_features.parquet  (1,261 features)
         ↓  ratio features  (polars, fast)
         ↓  streaming raw CSV pass  (lag + count + slope)
    data/processed/train_features.parquet  (2,100+ features, overwrite)
    ```

    ## ทำไม Lag Features ถึงสำคัญ?

    Processed features รวม 13 เดือนเป็น mean/last/std แล้ว — สูญเสีย **temporal sequence**
    Top Kaggle solutions เก็บ 2-3 เดือนย้อนหลังแยกกัน:

    | Feature | ความหมาย |
    |---------|---------|
    | `P_2_last` | payment เดือนล่าสุด (มีอยู่แล้ว) |
    | `P_2_lag2` | payment 2 เดือนก่อน (ใหม่) |
    | `P_2_lag3` | payment 3 เดือนก่อน (ใหม่) |
    | `P_2_slope` | `(lag1-lag3)/2` — trend direction (ใหม่) |

    Customer ที่ payment ลดลงเรื่อยๆ มีความเสี่ยง default สูงกว่า customer ที่ค่าเฉลี่ยเหมือนกัน
    แต่ปัจจุบัน improving → Lag features จับ temporal pattern นี้ได้
    """)
    return CAT_COLS, PROC_DIR, RAW_DIR, SKIP_COLS, Path, gc, mo, np, pd, pl


@app.cell
def _(PROC_DIR, mo, pl):
    mo.md("""
    ## Step 1 — Ratio Features (Fast, Polars)

    สร้างจาก `train_features.parquet` โดยตรง — ไม่ต้องแตะ raw CSV

    **3 กลุ่ม ratio:**

    | Feature | สูตร | ความหมาย |
    |---------|------|---------|
    | `X_last_div_mean` | last / mean | สถานะปัจจุบัน vs ค่าเฉลี่ย: > 1 = แย่ลง |
    | `X_last_div_max` | last / max | ใกล้ peak หรือไม่: → 1.0 = อยู่ที่ peak |
    | `X_min_div_max` | min / max | ช่วง compression: → 1.0 = ไม่มีความผันผวน |

    ทุก ratio ถูก clip ที่ [-10, 10] และ handle division-by-zero → null
    """)

    def make_ratio_features(df: pl.DataFrame) -> pl.DataFrame:
        exprs = []
        cols = df.columns
        col_set = set(cols)
        added: set[str] = set()  # guard against duplicate aliases from multiple suffixes sharing same base

        for base_col in cols:
            if not any(base_col.endswith(s) for s in ('_last', '_mean', '_min', '_max')):
                continue
            for suffix in ('_last', '_mean', '_min', '_max'):
                if base_col.endswith(suffix):
                    base = base_col[: -len(suffix)]
                    break

            last_col = f"{base}_last"
            mean_col = f"{base}_mean"
            max_col  = f"{base}_max"
            min_col  = f"{base}_min"

            alias = f"{base}_last_div_mean"
            if last_col in col_set and mean_col in col_set and alias not in col_set and alias not in added:
                exprs.append(
                    pl.when(pl.col(mean_col).abs() < 1e-6)
                    .then(None)
                    .otherwise((pl.col(last_col) / pl.col(mean_col)).clip(-10, 10))
                    .cast(pl.Float32)
                    .alias(alias)
                )
                added.add(alias)

            alias = f"{base}_last_div_max"
            if last_col in col_set and max_col in col_set and alias not in col_set and alias not in added:
                exprs.append(
                    pl.when(pl.col(max_col).abs() < 1e-6)
                    .then(None)
                    .otherwise((pl.col(last_col) / pl.col(max_col)).clip(-10, 10))
                    .cast(pl.Float32)
                    .alias(alias)
                )
                added.add(alias)

            alias = f"{base}_min_div_max"
            if min_col in col_set and max_col in col_set and alias not in col_set and alias not in added:
                exprs.append(
                    pl.when(pl.col(max_col).abs() < 1e-6)
                    .then(None)
                    .otherwise((pl.col(min_col) / pl.col(max_col)).clip(-10, 10))
                    .cast(pl.Float32)
                    .alias(alias)
                )
                added.add(alias)

        if not exprs:
            return df
        return df.with_columns(exprs)

    # quick test on shape
    _sample = pl.read_parquet(PROC_DIR / 'train_features.parquet', n_rows=100)
    _ratio_sample = make_ratio_features(_sample)
    _new_cols = _ratio_sample.shape[1] - _sample.shape[1]
    print(f"Ratio features to add: {_new_cols}")
    return (make_ratio_features,)


@app.cell
def _(CAT_COLS, RAW_DIR, SKIP_COLS, gc, mo, np, pd):
    mo.md("""
    ## Step 2 — Raw CSV Streaming (Lag + Count + Slope)

    ### เหตุผลที่ต้องกลับไปดู Raw CSV

    Processed features ทำ `groupby().last()` แล้ว — เดือนที่ 2nd-last และ 3rd-last **หายไปแล้ว**
    ต้องอ่าน raw CSV อีกครั้งเพื่อเก็บ:
    - **Lag buffer**: 3 monthly rows ล่าสุดต่อ customer (เรียงตาม S_2)
    - **Count buffer**: กี่เดือนทั้งหมด + กี่เดือนที่มี D-column > 0 (delinquency)

    ### Memory Budget

    ```
    460K customers × 3 rows × 177 cols × 4 bytes ≈ 1.0 GB
    ```
    เก็บทั้งหมดใน dict — manageable บน 16 GB RAM

    ### Streaming Logic

    ```
    for chunk in pd.read_csv(raw_csv, chunksize=50_000):
        sort by S_2 within chunk
        for cid, group in chunk.groupby('customer_ID'):
            lag_buffer[cid] = last 3 rows (numeric only)
            count_buffer[cid] += len(group)
            delinq_buffer[cid] += months with any D_col > 0
    ```

    Customer ที่ถูกตัดข้าม chunk → buffer จะถูก update ใน chunk ถัดไปโดยอัตโนมัติ
    เนื่องจาก raw CSV เรียงตาม customer_ID + S_2 แล้ว → 3 rows สุดท้ายถูกเสมอ
    """)

    # D-columns for delinquency counting
    _ALL_D_COLS_SET = None  # จะ detect จาก header

    CHUNK_SIZE_RAW = 50_000

    def stream_raw_for_lag(csv_path: str) -> tuple[dict, dict, dict, list]:
        """
        Single streaming pass through raw CSV.
        Returns:
          lag_buffer   : {cid -> np.ndarray shape (≤3, n_num_cols)}
          count_buffer : {cid -> int}  total months seen
          delinq_buffer: {cid -> int}  months with any D_col > 0
          num_cols     : list of numeric column names (177 cols)
        """
        lag_buffer: dict   = {}
        count_buffer: dict = {}
        delinq_buffer: dict = {}
        num_cols: list = []
        d_col_indices: list = []

        print(f"Streaming {csv_path}...")
        reader = pd.read_csv(csv_path, chunksize=CHUNK_SIZE_RAW)

        for chunk_i, chunk in enumerate(reader):
            # detect columns on first chunk
            if not num_cols:
                num_cols = [c for c in chunk.columns if c not in SKIP_COLS]
                d_col_indices = [i for i, c in enumerate(num_cols) if c.startswith('D_') and c not in CAT_COLS]

            # sort within chunk so rows are time-ordered
            chunk = chunk.sort_values('S_2')

            # downcast to float32 to save memory
            for c in num_cols:
                if chunk[c].dtype == np.float64:
                    chunk[c] = chunk[c].astype(np.float32)

            num_arr = chunk[num_cols].to_numpy(dtype=np.float32, na_value=np.nan)
            cids    = chunk['customer_ID'].to_numpy()

            # group by customer_ID (chunk is sorted by S_2, may not be sorted by cid)
            prev_cid = None
            row_start = 0
            for row_i in range(len(cids) + 1):
                cid = cids[row_i] if row_i < len(cids) else None
                if cid != prev_cid:
                    if prev_cid is not None:
                        rows = num_arr[row_start:row_i]  # shape (k, n_cols)
                        n    = len(rows)

                        # update count
                        count_buffer[prev_cid] = count_buffer.get(prev_cid, 0) + n

                        # update delinquency: months where any D_col > 0
                        if d_col_indices:
                            d_vals = rows[:, d_col_indices]
                            delinq = int(np.sum(np.any(d_vals > 0, axis=1)))
                        else:
                            delinq = 0
                        delinq_buffer[prev_cid] = delinq_buffer.get(prev_cid, 0) + delinq

                        # update lag buffer: keep last 3 rows
                        prev_buf = lag_buffer.get(prev_cid)
                        if prev_buf is None:
                            lag_buffer[prev_cid] = rows[-3:]
                        else:
                            combined = np.vstack([prev_buf, rows])
                            lag_buffer[prev_cid] = combined[-3:]

                    prev_cid  = cid
                    row_start = row_i

            if (chunk_i + 1) % 10 == 0:
                n_done = (chunk_i + 1) * CHUNK_SIZE_RAW
                print(f"  {n_done:,} rows processed, {len(lag_buffer):,} customers buffered...")
            del chunk
            gc.collect()

        print(f"Done. {len(lag_buffer):,} customers, {len(num_cols)} numeric cols.")
        return lag_buffer, count_buffer, delinq_buffer, num_cols

    print("stream_raw_for_lag() helper ready")
    return CHUNK_SIZE_RAW, stream_raw_for_lag


@app.cell
def _(mo, np, pd, stream_raw_for_lag):
    mo.md("""
    ## Step 3 — Buffer → DataFrame

    แปลง lag_buffer dict → DataFrame ที่มี columns:
    - `{col}_lag2` — เดือนที่ 2nd-last (index -2 จาก buffer)
    - `{col}_lag3` — เดือนที่ 3rd-last (index -3 จาก buffer)
    - `{col}_slope` — `(lag1 - lag3) / 2` finite difference approximation
    - `n_months` — จำนวนเดือนทั้งหมดใน history
    - `n_delinquent_months` — จำนวนเดือนที่มี D column > 0

    ### ทำไม slope = (lag1 - lag3) / 2?

    Finite difference บน 3 points: ถ้า lag1=เดือนล่าสุด, lag3=3 เดือนก่อน:
    ```
    slope ≈ (y_t - y_{t-2}) / 2  → rate of change per month
    ```
    เป็น approximation ของ derivative แบบ simple — เร็วและ interpretable
    """)

    def buffers_to_dataframe(
        lag_buffer: dict,
        count_buffer: dict,
        delinq_buffer: dict,
        num_cols: list,
    ) -> pd.DataFrame:
        cids   = list(lag_buffer.keys())
        n_cust = len(cids)
        n_cols = len(num_cols)

        lag2_arr  = np.full((n_cust, n_cols), np.nan, dtype=np.float32)
        lag3_arr  = np.full((n_cust, n_cols), np.nan, dtype=np.float32)
        slope_arr = np.full((n_cust, n_cols), np.nan, dtype=np.float32)

        for i, cid in enumerate(cids):
            buf = lag_buffer[cid]  # shape (≤3, n_cols)
            k   = len(buf)
            lag1 = buf[-1]          # most recent (= _last in processed)
            if k >= 2:
                lag2_arr[i] = buf[-2]
            if k >= 3:
                lag3_arr[i] = buf[-3]
                slope_arr[i] = (lag1 - buf[-3]) / 2.0

        lag2_cols  = [f"{c}_lag2"  for c in num_cols]
        lag3_cols  = [f"{c}_lag3"  for c in num_cols]
        slope_cols = [f"{c}_slope" for c in num_cols]

        df = pd.DataFrame(
            np.hstack([lag2_arr, lag3_arr, slope_arr]),
            index=cids,
            columns=lag2_cols + lag3_cols + slope_cols,
        )
        df.index.name = 'customer_ID'

        df['n_months']           = pd.array([count_buffer.get(c, 0)  for c in cids], dtype='int16')
        df['n_delinquent_months'] = pd.array([delinq_buffer.get(c, 0) for c in cids], dtype='int16')

        return df

    print("buffers_to_dataframe() helper ready")
    return (buffers_to_dataframe,)


@app.cell
def _(PROC_DIR, RAW_DIR, buffers_to_dataframe, make_ratio_features, mo, pl, stream_raw_for_lag):
    mo.md("""
    ## Step 4 — Process Train Features

    **Pipeline:**
    1. อ่าน `train_features.parquet` → เพิ่ม ratio features
    2. Stream `train_data.csv` → lag/count/slope buffers
    3. Convert buffers → DataFrame
    4. Join ทุกอย่างเข้าด้วยกัน → overwrite `train_features.parquet`

    **Idempotent check:** ถ้า `P_2_lag2` column มีอยู่แล้ว → skip
    """)

    _train_path = PROC_DIR / 'train_features.parquet'

    _existing = pl.read_parquet(_train_path, n_rows=1)
    if 'P_2_lag2' in _existing.columns:
        print("train_features.parquet already has lag features — skipping.")
    else:
        print("Loading train_features.parquet...")
        _train = pl.read_parquet(_train_path)
        print(f"  Loaded: {_train.shape[0]:,} customers × {_train.shape[1]:,} features")

        # Phase 1: ratio features
        print("Adding ratio features...")
        _train = make_ratio_features(_train)
        print(f"  After ratio: {_train.shape[1]:,} features")

        # Phase 2: stream raw CSV
        _lag_buf, _cnt_buf, _delinq_buf, _num_cols = stream_raw_for_lag(
            str(RAW_DIR / 'train_data.csv')
        )

        # Phase 3: convert buffers → pandas df, then polars
        import pandas as _pd
        _lag_df = buffers_to_dataframe(_lag_buf, _cnt_buf, _delinq_buf, _num_cols)
        del _lag_buf, _cnt_buf, _delinq_buf
        _lag_pl = pl.from_pandas(_lag_df.reset_index())
        del _lag_df

        # Phase 4: join
        _train = _train.join(_lag_pl, on='customer_ID', how='left')
        print(f"  After lag/slope/count: {_train.shape[1]:,} features")

        # Save
        _train.write_parquet(_train_path)
        print(f"Saved train_features.parquet: {_train.shape[0]:,} rows × {_train.shape[1]:,} cols")
        del _train, _lag_pl
    return


@app.cell
def _(PROC_DIR, RAW_DIR, buffers_to_dataframe, make_ratio_features, mo, pl, stream_raw_for_lag):
    mo.md("""
    ## Step 5 — Process Test Features

    เหมือน Step 4 ทุกอย่าง แต่ใช้ `test_features.parquet` + `test_data.csv`
    """)

    _test_path = PROC_DIR / 'test_features.parquet'

    _existing_t = pl.read_parquet(_test_path, n_rows=1)
    if 'P_2_lag2' in _existing_t.columns:
        print("test_features.parquet already has lag features — skipping.")
    else:
        print("Loading test_features.parquet...")
        _test = pl.read_parquet(_test_path)
        print(f"  Loaded: {_test.shape[0]:,} customers × {_test.shape[1]:,} features")

        # Phase 1: ratio features
        print("Adding ratio features...")
        _test = make_ratio_features(_test)
        print(f"  After ratio: {_test.shape[1]:,} features")

        # Phase 2: stream raw test CSV
        _lag_buf, _cnt_buf, _delinq_buf, _num_cols = stream_raw_for_lag(
            str(RAW_DIR / 'test_data.csv')
        )

        # Phase 3: convert buffers → polars
        import pandas as _pd
        _lag_df = buffers_to_dataframe(_lag_buf, _cnt_buf, _delinq_buf, _num_cols)
        del _lag_buf, _cnt_buf, _delinq_buf
        _lag_pl = pl.from_pandas(_lag_df.reset_index())
        del _lag_df

        # Phase 4: join
        _test = _test.join(_lag_pl, on='customer_ID', how='left')
        print(f"  After lag/slope/count: {_test.shape[1]:,} features")

        # Save
        _test.write_parquet(_test_path)
        print(f"Saved test_features.parquet: {_test.shape[0]:,} rows × {_test.shape[1]:,} cols")
        del _test, _lag_pl
    return


@app.cell
def _(PROC_DIR, mo, pl):
    mo.md("""
    ## Step 6 — Verify & Summary

    ตรวจสอบ feature count และ sample ของ features ใหม่
    """)

    _train_final = pl.read_parquet(PROC_DIR / 'train_features.parquet', n_rows=5)
    _all_cols = _train_final.columns

    _lag_cols   = [c for c in _all_cols if c.endswith(('_lag2', '_lag3'))]
    _slope_cols = [c for c in _all_cols if c.endswith('_slope')]
    _ratio_cols = [c for c in _all_cols if c.endswith(('_last_div_mean', '_last_div_max', '_min_div_max'))]
    _count_cols = [c for c in _all_cols if c in ('n_months', 'n_delinquent_months')]

    mo.md(f"""
    ### Feature Count Summary

    | กลุ่ม | จำนวน |
    |-------|-------|
    | Lag (lag2 + lag3) | {len(_lag_cols):,} |
    | Slope | {len(_slope_cols):,} |
    | Ratio | {len(_ratio_cols):,} |
    | Count | {len(_count_cols)} |
    | **รวมทั้งหมด** | **{len(_all_cols):,}** |

    ### ตัวอย่าง features ใหม่
    """)
    return


@app.cell
def _(PROC_DIR, mo, pl):
    mo.md("""
    ## Step 7 — Delinquency Rate Feature

    เพิ่ม `delinquent_rate = n_delinquent_months / n_months`

    **ทำไม relative ดีกว่า absolute?**

    | Feature | ปัญหา | แก้ด้วย |
    |---------|-------|---------|
    | `n_delinquent_months` (absolute) | drift ตามช่วงเศรษฐกิจ | → |
    | `delinquent_rate` (relative) | stable ข้ามช่วงเวลา เปรียบเทียบคนได้ยุติธรรม | ✅ |

    Customer ที่ค้างชำระ 3/13 เดือน vs 3/3 เดือน — absolute เหมือนกัน แต่ relative ต่างกันมาก
    """)

    for _name, _path in [("train", PROC_DIR / "train_features.parquet"),
                          ("test",  PROC_DIR / "test_features.parquet")]:
        _df = pl.read_parquet(_path)
        if "delinquent_rate" in _df.columns:
            print(f"{_name}: delinquent_rate มีอยู่แล้ว — skipping")
            continue
        if "n_months" not in _df.columns or "n_delinquent_months" not in _df.columns:
            print(f"{_name}: ไม่มี n_months/n_delinquent_months — ข้าม (รัน Step 4/5 ก่อน)")
            continue

        _df = _df.with_columns(
            pl.when(pl.col("n_months") == 0)
            .then(None)
            .otherwise(pl.col("n_delinquent_months") / pl.col("n_months"))
            .cast(pl.Float32)
            .alias("delinquent_rate")
        )
        _df.write_parquet(_path)
        _rate = _df["delinquent_rate"]
        print(f"{_name}: delinquent_rate added  mean={_rate.mean():.3f}  max={_rate.max():.3f}")
    return


@app.cell
def _(PROC_DIR, mo):
    mo.md("""
    ## Step 8 — Auto-Register Engineered Features

    บันทึก columns ใหม่ทั้งหมด (lag/ratio/slope/count/delinquent_rate)
    ลงใน `feature_registry.json` อัตโนมัติ
    """)

    import sys as _sys
    _sys.path.insert(0, str(PROC_DIR.parent.parent))  # project root
    from scripts.feature_registry import update_registry as _update_registry

    _n = _update_registry(
        PROC_DIR / "train_features.parquet",
        phase="engineered",
        reason="02_feature_engineering.py — lag/ratio/slope/count/delinquent_rate",
    )
    print(f"feature_registry.json: +{_n} new engineered features registered")
    return


if __name__ == "__main__":
    app.run()
