from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/SPARC/metadata")
OUT_DIR = Path("data/SPARC")


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta = pd.read_csv(DATA_DIR / "SPARC_Lelli2016c.mrt")
    cdr = pd.read_csv(DATA_DIR / "CDR_Lelli2016b.mrt")
    btfr = pd.read_csv(DATA_DIR / "BTFR_Lelli2019.mrt")
    return meta, cdr, btfr


def build_catalog() -> pd.DataFrame:
    meta, cdr, btfr = load_tables()

    catalog = meta.merge(cdr, on="Galaxy", how="left")
    catalog = catalog.merge(btfr, on="Galaxy", how="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(OUT_DIR / "sparc_master_catalog.csv", index=False)
    print("Catalog created:", len(catalog), "galaxies")
    return catalog


if __name__ == "__main__":
    build_catalog()
