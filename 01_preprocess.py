import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import os, zipfile, gc
    import marimo as mo
    import pandas as pd
    import numpy as np
    from pathlib import Path

    RAW_DIR = Path('data/raw')
    OUT_DIR = Path('data/processed')
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mo.md("""
    # 01 — Preprocessing Pipeline

    Script นี้แปลง raw monthly statement data → per-customer feature matrix
    ที่พร้อมส่งเข้า model

    ```
    data/raw/train_data.csv     (~16 GB, monthly rows)
         ↓  aggregate_customer()
    data/processed/train_features.parquet   (per-customer, 1 row / customer)
    data/processed/test_features.parquet
    data/processed/train_labels.parquet
    ```

    ## ทำไม Raw Data ถึงต้อง Preprocess ก่อน?

    Raw data มี **1 customer = หลาย rows** (หนึ่ง row ต่อหนึ่งเดือน):

    | customer_ID | P_2 | D_39 | B_1 | ... | (เดือน) |
    |-------------|-----|------|-----|-----|---------|
    | cust_001    | 0.8 | 0.0  | 1.2 | ... | Jan     |
    | cust_001    | 0.6 | 0.1  | 1.1 | ... | Feb     |
    | cust_001    | 0.4 | 0.3  | 0.9 | ... | Mar     |

    แต่ ML model ต้องการ **1 customer = 1 row** → ต้อง aggregate ก่อน

    ## Libraries ที่ใช้
    | Library | ทำอะไร |
    |---------|--------|
    | `pandas` | อ่าน CSV, groupby, aggregate |
    | `numpy` | downcast dtype เพื่อประหยัด memory |
    | `zipfile` | แตกไฟล์ .zip จาก Kaggle |
    | `gc` | garbage collect หลัง process แต่ละ chunk เพื่อคืน RAM |
    | `pathlib.Path` | จัดการ file path แบบ cross-platform |
    """)
    return OUT_DIR, Path, RAW_DIR, gc, mo, np, os, pd, zipfile


@app.cell
def _(RAW_DIR, mo, os):
    mo.md("""
    ## Step 1 — Download Data จาก Kaggle (~16 GB)

    ใช้ Kaggle CLI (`kaggle competitions download`) ดาวน์โหลดข้อมูล

    **Idempotent check:** ถ้าไฟล์มีอยู่แล้วและใหญ่กว่า 15 GB → ข้ามขั้นตอนนี้ทันที
    ป้องกันการ download ซ้ำโดยไม่จำเป็น

    > **ต้องการ:** `kaggle.json` API token อยู่ที่ `~/.kaggle/kaggle.json`
    > ได้มาจาก kaggle.com → Account → API → Create New Token
    """)

    ZIP_PATH = RAW_DIR / 'amex-default-prediction.zip'

    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 15 * (1024**3):
        print(f'Already downloaded ({ZIP_PATH.stat().st_size/(1024**3):.1f} GB) — skipping.')
    else:
        print('Downloading (~16 GB)...')
        os.system(f'kaggle competitions download -c amex-default-prediction -p {RAW_DIR}')
        print('Done!')
    return (ZIP_PATH,)


@app.cell
def _(RAW_DIR, ZIP_PATH, mo, zipfile):
    mo.md("""
    ## Step 2 — Unzip (~50 GB พื้นที่ disk ที่ต้องการ)

    แตก 3 ไฟล์จาก zip:
    - `train_data.csv` — monthly statement ของ training customers
    - `test_data.csv` — monthly statement ของ test customers
    - `train_labels.csv` — ว่า customer แต่ละคน default หรือไม่ (0/1)

    **Idempotent check:** ถ้าทั้ง 3 ไฟล์มีอยู่แล้ว → ข้ามทันที
    """)

    needed = ['train_data.csv', 'test_data.csv', 'train_labels.csv']
    if all((RAW_DIR / f).exists() for f in needed):
        print('Already unzipped — skipping.')
    else:
        with zipfile.ZipFile(ZIP_PATH) as z:
            for m in z.namelist():
                print(f'  Extracting {m}...')
                z.extract(m, RAW_DIR)
        print('Unzip done!')

    for f in sorted(RAW_DIR.iterdir()):
        print(f'  {f.name}  {f.stat().st_size/(1024**3):.1f} GB')

    return


@app.cell
def _(Path, gc, mo, np, pd):
    mo.md("""
    ## Step 3 — Feature Engineering Helpers

    ### `downcast(df)` — ลด Memory ด้วยการ Downcast dtype

    Raw data ใช้ `float64` (8 bytes) และ `int64` (8 bytes) ต่อค่า
    แต่ข้อมูล financial ทั่วไปไม่ต้องการ precision สูงขนาดนั้น

    ```
    float64  (8 bytes) → float32 (4 bytes)  ลด 50%
    int64    (8 bytes) → int32   (4 bytes)  ลด 50%
    ```

    ข้อมูล 16 GB จะลดเหลือ ~8 GB ใน RAM — สำคัญมากเมื่อ process บน laptop

    ---

    ### `aggregate_customer(df)` — หัวใจของ Preprocessing

    แปลง monthly rows → 1 row ต่อ customer ด้วย 2 กลุ่ม:

    **Numeric columns** (ตัวเลขจริง เช่น balance, payment amount):

    | Aggregation | Suffix | ความหมาย | ตัวอย่าง |
    |-------------|--------|----------|---------|
    | `groupby.mean()` | `_mean` | ค่าเฉลี่ยทุกเดือน | payment เฉลี่ยตลอดประวัติ |
    | `groupby.std()` | `_std` | ความผันผวน | payment ขึ้นๆ ลงๆ แค่ไหน |
    | `groupby.min()` | `_min` | ต่ำสุดตลอดประวัติ | balance ต่ำที่สุดเคยเป็น |
    | `groupby.max()` | `_max` | สูงสุดตลอดประวัติ | ยอดหนี้สูงสุดเคยเป็น |
    | `groupby.last()` | `_last` | เดือนล่าสุด | สถานะปัจจุบัน |
    | `last - mean` | `_last_minus_mean` | เดือนล่าสุด vs ค่าเฉลี่ย | trend ขึ้นหรือลง? |
    | `last - first` | `_last_minus_first` | เดือนล่าสุด vs เดือนแรก | เปลี่ยนไปเท่าไหร่ตลอดช่วง |

    **Categorical columns** (codes/flags เช่น product type, risk grade):
    ไม่สามารถหา mean/std ได้ → ใช้แค่:
    - `_last` = category ล่าสุด (สถานะปัจจุบัน)
    - `_nunique` = มีกี่ category ที่เคยเห็น (เปลี่ยนบ่อยไหม?)

    ```python
    CAT_COLS = ['B_30','B_38','D_114', ...]  # 11 categorical features
    ```

    ผลลัพธ์: feature เดิม 1 column → 7 columns (numeric) หรือ 2 columns (categorical)
    จึงได้ **1,261 features** จาก 190 original columns

    ---

    ### `process_file()` — Chunked Processing สำหรับ File ขนาดใหญ่

    ทำไมต้อง chunk? เพราะ `train_data.csv` มีขนาด ~16 GB
    ใหญ่กว่า RAM ของคอมพิวเตอร์ส่วนใหญ่

    ```
    อ่านทีละ 50,000 rows (CHUNK_SIZE)
         ↓ aggregate_customer() → customer-level rows
         ↓ เก็บผลไว้ใน list
         ↓ del chunk + gc.collect()  ← คืน RAM ทันทีหลังใช้เสร็จ

    เมื่ออ่านครบทุก chunk:
         ↓ pd.concat(chunks)  → รวมทุก chunk
         ↓ groupby().mean()   ← re-aggregate: customer ที่กระจายใน 2 chunk
                                จะได้ถูก average รวมกันตรงนี้
         ↓ to_parquet()       → บันทึกเป็น parquet (อ่านเร็วกว่า CSV มาก)
    ```

    > **ทำไม re-aggregate รอบสอง?** เพราะ customer 1 คนอาจถูกตัดคร่อมระหว่าง chunk
    > เช่น เดือน 1-2 อยู่ใน chunk 1, เดือน 3-6 อยู่ใน chunk 2
    > ต้อง merge ผลของทั้งสอง chunk เข้าด้วยกัน
    """)

    CHUNK_SIZE = 50_000
    CAT_COLS = ['B_30','B_38','D_114','D_116','D_117','D_120','D_126','D_63','D_64','D_66','D_68']

    def downcast(df):
        for col in df.select_dtypes('float64').columns:
            df[col] = df[col].astype(np.float32)
        for col in df.select_dtypes('int64').columns:
            df[col] = df[col].astype(np.int32)
        return df

    def aggregate_customer(df):
        cid = df['customer_ID']
        df  = df.drop(columns=['customer_ID', 'S_2'])  # S_2 = statement date (ไม่ใช่ feature)
        num = [c for c in df.columns if c not in CAT_COLS]
        cat = [c for c in df.columns if c in CAT_COLS]
        parts = []
        if num:
            g = df[num].groupby(cid)
            last, mean, first = g.last(), g.mean(), g.first()
            parts.append(pd.concat([
                mean.add_suffix('_mean'),
                g.std().add_suffix('_std'),
                g.min().add_suffix('_min'),
                g.max().add_suffix('_max'),
                last.add_suffix('_last'),
                (last - mean).add_suffix('_last_minus_mean'),
                (last - first).add_suffix('_last_minus_first'),
            ], axis=1))
        if cat:
            g = df[cat].groupby(cid)
            parts.append(pd.concat([
                g.last().add_suffix('_last'),
                g.nunique().add_suffix('_nunique'),
            ], axis=1))
        return pd.concat(parts, axis=1)

    def process_file(csv_path, out_path, label):
        if Path(out_path).exists():
            print(f'{label} already done — skipping.')
            return
        print(f'Processing {label}...')
        chunks = []
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):
            chunks.append(aggregate_customer(downcast(chunk)))
            if (i+1) % 5 == 0:
                print(f'  {(i+1)*CHUNK_SIZE:,} rows...')
            del chunk
            gc.collect()  # คืน RAM ทันทีหลัง process แต่ละ chunk
        combined = pd.concat(chunks)
        # re-aggregate: รวม customer ที่กระจายอยู่ใน 2+ chunks เข้าด้วยกัน
        num_cols = combined.select_dtypes('number').columns
        str_cols = combined.select_dtypes(exclude='number').columns
        result = [combined[num_cols].groupby(combined.index).mean()]
        if len(str_cols):
            result.append(combined[str_cols].groupby(combined.index).last())
        final = downcast(pd.concat(result, axis=1))
        final.to_parquet(out_path)
        print(f'  Done: {final.shape[0]:,} customers x {final.shape[1]:,} features')
        del chunks, combined, final
        gc.collect()

    print('Helpers ready')
    return (process_file,)


@app.cell
def _(OUT_DIR, RAW_DIR, mo, process_file):
    mo.md("""
    ## Step 4 — Process Train Data (~20–40 นาที)

    อ่าน `train_data.csv` → aggregate → บันทึกเป็น `train_features.parquet`

    **Idempotent:** ถ้า parquet มีอยู่แล้ว → ข้ามทันที (รัน notebook ซ้ำได้ปลอดภัย)

    > **Parquet ดีกว่า CSV อย่างไร?**
    > - อ่านเร็วกว่า 10-50x (columnar format)
    > - บีบอัดอัตโนมัติ (ไฟล์เล็กกว่า CSV)
    > - เก็บ dtype ไว้ด้วย (ไม่ต้อง downcast ซ้ำทุกครั้งที่อ่าน)
    """)
    process_file(RAW_DIR / 'train_data.csv', OUT_DIR / 'train_features.parquet', 'train data')
    return


@app.cell
def _(OUT_DIR, RAW_DIR, mo, process_file):
    mo.md("""
    ## Step 5 — Process Test Data

    เหมือน Step 4 ทุกอย่าง แต่ใช้ `test_data.csv`
    ผลลัพธ์ถูกบันทึกเป็น `test_features.parquet`

    > **หมายเหตุ:** test data ไม่มี labels — ใช้สำหรับ predict และ submit ไปยัง Kaggle
    """)
    process_file(RAW_DIR / 'test_data.csv', OUT_DIR / 'test_features.parquet', 'test data')
    return


@app.cell
def _(OUT_DIR, RAW_DIR, mo, pd):
    mo.md("""
    ## Step 6 — Save Labels

    `train_labels.csv` เป็นไฟล์เล็ก (แค่ 2 columns):
    - `customer_ID` — รหัส customer
    - `target` — 0 = ไม่ default, 1 = default

    แปลงเป็น parquet เพื่อความสม่ำเสมอกับ features file

    Pipeline notebook อื่นๆ ใช้ไฟล์นี้ด้วย `join` กับ `train_features.parquet`:
    ```python
    df = train_features.join(train_labels, on='customer_ID')
    ```
    """)

    out = OUT_DIR / 'train_labels.parquet'
    if out.exists():
        print('Already saved — skipping.')
    else:
        labels = pd.read_csv(RAW_DIR / 'train_labels.csv')
        labels.to_parquet(out, index=False)
        print(f'{len(labels):,} labels saved. Default rate: {labels["target"].mean():.2%}')
    return


@app.cell
def _(OUT_DIR, mo):
    mo.md("""
    ## Step 7 — Auto-Register Features

    บันทึก columns ทั้งหมดที่ preprocessing สร้างลงใน `feature_registry.json`
    ทำงานอัตโนมัติ — columns ที่มีอยู่แล้วใน registry จะไม่ถูกแตะ
    """)

    import sys as _sys
    _sys.path.insert(0, str(OUT_DIR.parent.parent))  # project root
    from scripts.feature_registry import update_registry as _update_registry

    _n_train = _update_registry(
        OUT_DIR / "train_features.parquet",
        phase="original",
        reason="01_preprocess.py — aggregate monthly CSV → per-customer features",
    )
    print(f"feature_registry.json: +{_n_train} new features registered (train)")
    return


if __name__ == "__main__":
    app.run()
