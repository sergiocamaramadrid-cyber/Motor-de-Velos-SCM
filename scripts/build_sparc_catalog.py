from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/SPARC/metadata")
OUT_DIR = Path("data/SPARC")
REQUIRED_METADATA_FILES = (
    "SPARC_Lelli2016c.mrt",
    "CDR_Lelli2016b.mrt",
    "BTFR_Lelli2019.mrt",
    "MassModels_Lelli2016c.mrt",
)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [name for name in REQUIRED_METADATA_FILES if not (DATA_DIR / name).exists()]
    if missing:
        expected = ", ".join(REQUIRED_METADATA_FILES)
        raise FileNotFoundError(
            "Missing SPARC metadata table(s): "
            f"{', '.join(missing)}. "
            f"Expected files in {DATA_DIR}: {expected}"
        )

    meta = pd.read_csv(DATA_DIR / "SPARC_Lelli2016c.mrt")
    cdr = pd.read_csv(DATA_DIR / "CDR_Lelli2016b.mrt")
    btfr = pd.read_csv(DATA_DIR / "BTFR_Lelli2019.mrt")
    # Validated for radial analyses; not merged in sparc_master_catalog.csv.
    pd.read_csv(DATA_DIR / "MassModels_Lelli2016c.mrt")

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
