#!/usr/bin/env python3

from pathlib import Path

import pandas as pd

INPUT_TXT = "data/LITTLE_THINGS/Hunter_2012.txt"
OUTPUT_CSV = "results/LITTLE_THINGS/little_things_catalog.csv"
PIPELINE_INPUT_CSV = "data/little_things_global.csv"
PIPELINE_OUTPUT_CSV = "results/LITTLE_THINGS/little_things_global_enriched.csv"

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


def _with_requested_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["MHI"] = out.get("log_mhi")
    out["logSFR"] = out.get("log_sfr_ha")
    if "logSFR" in out.columns and "log_sfr_uv" in out.columns:
        out["logSFR"] = out["logSFR"].fillna(out["log_sfr_uv"])
    out["inclination"] = out.get("inclination_deg")
    out["Rd"] = out.get("disk_scale_kpc")
    return out


def _merge_with_pipeline(df_clean: pd.DataFrame, pipeline_csv: Path, pipeline_output_csv: Path) -> Path | None:
    if not pipeline_csv.exists():
        return None
    pipeline_df = pd.read_csv(pipeline_csv)
    if "galaxy_id" not in pipeline_df.columns:
        return None
    enrich_cols = ["galaxy", "distance_mpc", "MHI", "logSFR", "inclination", "Rd"]
    merged = pipeline_df.merge(
        df_clean[enrich_cols].drop_duplicates(subset=["galaxy"]),
        left_on="galaxy_id",
        right_on="galaxy",
        how="left",
    )
    pipeline_output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(pipeline_output_csv, index=False)
    return pipeline_output_csv


def build_catalog(
    input_txt: Path,
    output_csv: Path,
    pipeline_csv: Path | None = None,
    pipeline_output_csv: Path | None = None,
) -> tuple[Path, Path | None]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_txt, sep=r"\s+", comment="#", engine="python")
    df = df.rename(columns=RENAME_MAP)
    keep_cols = [c for c in KEEP_COLS if c in df.columns]
    df_clean = _with_requested_columns(df[keep_cols])
    df_clean.to_csv(output_csv, index=False)
    merged_out = None
    if pipeline_csv is not None and pipeline_output_csv is not None:
        merged_out = _merge_with_pipeline(df_clean, pipeline_csv, pipeline_output_csv)
    return output_csv, merged_out


def main() -> None:
    output_csv, merged_out = build_catalog(
        Path(INPUT_TXT),
        Path(OUTPUT_CSV),
        pipeline_csv=Path(PIPELINE_INPUT_CSV),
        pipeline_output_csv=Path(PIPELINE_OUTPUT_CSV),
    )
    print("Saved:", output_csv)
    if merged_out is not None:
        print("Saved:", merged_out)


if __name__ == "__main__":
    main()
