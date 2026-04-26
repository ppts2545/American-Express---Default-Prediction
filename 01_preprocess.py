import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import os, zipfile, gc
    import pandas as pd
    import numpy as np
    from pathlib import Path

    RAW_DIR = Path('data/raw')
    OUT_DIR = Path('data/processed')
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Paths ready')

    return OUT_DIR, Path, RAW_DIR, gc, np, os, pd, zipfile


@app.cell
def _(RAW_DIR, os):
    # Step1 - Downloads Data
    ZIP_PATH = RAW_DIR / 'amex-default-prediction.zip'

    if ZIP_PATH.exists() and ZIP_PATH.stat().st_size > 15 * (1024**3):
        print(f'Already downloaded ({ZIP_PATH.stat().st_size/(1024**3):.1f} GB) — skipping.')
    else:
        print('Downloading (~16 GB)...')
        os.system(f'kaggle competitions download -c amex-default-prediction -p {RAW_DIR}')
        print('Done!')
    return (ZIP_PATH,)


@app.cell
def _(RAW_DIR, ZIP_PATH, zipfile):
    # Step 2 — Unzip (~50 GB needed on disk)
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
def _(Path, gc, np, pd):
    # Step 3 — Feature engineering helpers
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
        df  = df.drop(columns=['customer_ID', 'S_2'])
        num = [c for c in df.columns if c not in CAT_COLS]
        cat = [c for c in df.columns if c in CAT_COLS]
        parts = []
        if num:
            g = df[num].groupby(cid)
            last, mean, first = g.last(), g.mean(), g.first()
            parts.append(pd.concat([
                mean.add_suffix('_mean'), g.std().add_suffix('_std'),
                g.min().add_suffix('_min'), g.max().add_suffix('_max'),
                last.add_suffix('_last'),
                (last-mean).add_suffix('_last_minus_mean'),
                (last-first).add_suffix('_last_minus_first'),
            ], axis=1))
        if cat:
            g = df[cat].groupby(cid)
            parts.append(pd.concat([g.last().add_suffix('_last'), g.nunique().add_suffix('_nunique')], axis=1))
        return pd.concat(parts, axis=1)

    def process_file(csv_path, out_path, label):
        if Path(out_path).exists():
            print(f'{label} already done — skipping.')
            return
        print(f'Processing {label}...')
        chunks = []
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):
            chunks.append(aggregate_customer(downcast(chunk)))
            if (i+1) % 5 == 0: print(f'  {(i+1)*CHUNK_SIZE:,} rows...')
            del chunk; gc.collect()
        combined = pd.concat(chunks)
        num_cols = combined.select_dtypes('number').columns
        str_cols = combined.select_dtypes(exclude='number').columns
        result = [combined[num_cols].groupby(combined.index).mean()]
        if len(str_cols): result.append(combined[str_cols].groupby(combined.index).last())
        final = downcast(pd.concat(result, axis=1))
        final.to_parquet(out_path)
        print(f'  Done: {final.shape[0]:,} customers x {final.shape[1]:,} features')
        del chunks, combined, final; gc.collect()

    print('Helpers ready')

    return (process_file,)


@app.cell
def _(OUT_DIR, RAW_DIR, process_file):
    #Step 4 — Process train (~20–40 min)
    process_file(RAW_DIR/'train_data.csv', OUT_DIR/'train_features.parquet', 'train data')
    return


@app.cell
def _(OUT_DIR, RAW_DIR, process_file):
    #Step 5 — Process test
    process_file(RAW_DIR/'test_data.csv', OUT_DIR/'test_features.parquet', 'test data')

    return


@app.cell
def _(OUT_DIR, RAW_DIR, pd):
    # Step 6 — Save labels
    out = OUT_DIR / 'train_labels.parquet'
    if out.exists():
        print('Already saved — skipping.')
    else:
        labels = pd.read_csv(RAW_DIR / 'train_labels.csv')
        labels.to_parquet(out, index=False)
        print(f'{len(labels):,} labels saved. Default rate: {labels["target"].mean():.2%}')

    return


if __name__ == "__main__":
    app.run()
