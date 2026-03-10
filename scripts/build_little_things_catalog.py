#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

INPUT_TXT = "data/LITTLE_THINGS/Hunter_2012.txt"
OUTPUT_CSV = "results/LITTLE_THINGS/little_things_catalog.csv"

RENAME_MAP = {
    "Name": "galaxy",
    "Cl": "morphology",
    "Dist": "distance_mpc",
    "VMag": "abs_mag_v",
    "Rd": "disk_scale_kpc",
    "Rad": "holmberg_radius_arcmin",
    "logSFR1": "log_sfr_ha",
    "logSFR2": "log_sfr_uv",
    "MHI": "log_mhi",
    "[O/H]": "metallicity_12logOH",
    "PA": "position_angle_deg",
    "b/a": "axis_ratio",
    "i": "inclination_deg",
    "_RA": "ra_deg",
    "_DE": "dec_deg",
}

KEEP_COLS = [
    "galaxy",
    "morphology",
    "distance_mpc",
    "abs_mag_v",
    "disk_scale_kpc",
    "holmberg_radius_arcmin",
    "log_sfr_ha",
    "log_sfr_uv",
    "log_mhi",
    "metallicity_12logOH",
    "position_angle_deg",
    "axis_ratio",
    "inclination_deg",
    "ra_deg",
    "dec_deg",
]


def build_catalog(input_txt: Path, output_csv: Path) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_txt, sep=r"\s+", comment="#", engine="python")
    df = df.rename(columns=RENAME_MAP)
    keep_cols = [c for c in KEEP_COLS if c in df.columns]
    df_clean = df[keep_cols]
    df_clean.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    output_csv = build_catalog(Path(INPUT_TXT), Path(OUTPUT_CSV))
    print("Saved:", output_csv)


if __name__ == "__main__":
    main()
