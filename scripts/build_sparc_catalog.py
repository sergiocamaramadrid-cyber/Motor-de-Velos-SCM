from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/SPARC/metadata")
OUT_DIR = Path("data/SPARC")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = {
        "meta": DATA_DIR / "SPARC_Lelli2016c.mrt",
        "cdr": DATA_DIR / "CDR_Lelli2016b.mrt",
        "btfr": DATA_DIR / "BTFR_Lelli2019.mrt",
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing SPARC table: {path}")

    meta = pd.read_csv(files["meta"])
    cdr = pd.read_csv(files["cdr"])
    btfr = pd.read_csv(files["btfr"])

    for df in (meta, cdr, btfr):
        df["Galaxy"] = df["Galaxy"].astype(str).str.strip()
    return meta, cdr, btfr


def build_catalog() -> pd.DataFrame:
    meta, cdr, btfr = load_tables()

    catalog = meta.merge(cdr, on="Galaxy", how="left")
    catalog = catalog.merge(btfr, on="Galaxy", how="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "sparc_master_catalog.csv"
    catalog.to_csv(out_file, index=False)
    print(f"Catalog created: {len(catalog)} galaxies")
    print(f"Output file: {out_file}")
    return catalog


if __name__ == "__main__":
    build_catalog()
