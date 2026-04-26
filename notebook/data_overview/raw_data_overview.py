import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import polars as pl
    import seaborn

    return (Path,)


@app.cell
def _(Path):

    # Go up parent directory
    ROOT = next(
        p
        for p in [Path.cwd(), *Path.cwd().parents]
        if (p / "data" / "raw" / "train.csv").exists()
    )

    RAW_PATH = ROOT / "data" / "raw" / "train.csv"
    CLEAN_PATH = ROOT / "data" / "processed" / "quality" / "df_train_cleaned.parquet"
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    return


if __name__ == "__main__":
    app.run()
