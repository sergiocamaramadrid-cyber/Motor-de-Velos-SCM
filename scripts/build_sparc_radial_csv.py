#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

try:
    from scripts.process_sparc import consolidate_sparc
except ModuleNotFoundError:  # pragma: no cover - CLI execution path
    from process_sparc import consolidate_sparc

EPS = 1e-12
# Softening scale (kpc) used in the synthetic gbar(r) proxy to avoid an r→0 singularity.
# We use 0.5 kpc as a conservative inner-radius floor for this first-pass synthetic profile.
GBAR_SOFTENING_KPC = 0.5
# Radial decay exponent for the synthetic gbar(r) proxy profile.
# We use 1.2 to represent a mild outer decline without over-steepening the synthetic curve.
GBAR_RADIAL_EXPONENT = 1.2


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["galaxy", "r", "gbar", "gobs", "SB"])
    out = out[(out["r"] > 0) & (out["gbar"] > 0) & (out["gobs"] > 0) & (out["SB"] > 0)].copy()
    out = out.sort_values(["galaxy", "r"]).reset_index(drop=True)
    return out


def build_from_rotmod(sparc_dir: str | Path) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_out = Path(tmp.name)
    try:
        rotmod_df = consolidate_sparc(input_dir=sparc_dir, output_file=tmp_out)
        out = pd.DataFrame(
            {
                "galaxy": rotmod_df["galaxy"],
                "r": rotmod_df["radius"],
                "gbar": (rotmod_df["v_bar"] ** 2) / np.maximum(rotmod_df["radius"], EPS),
                "gobs": (rotmod_df["v_obs"] ** 2) / np.maximum(rotmod_df["radius"], EPS),
                "SB": (rotmod_df["v_disk"] ** 2) / np.maximum(rotmod_df["radius"], EPS),
            }
        )
    finally:
        tmp_out.unlink(missing_ok=True)
    return _finalize(out)


def build_from_master(master_csv: str | Path, n_rings: int = 12) -> pd.DataFrame:
    master = pd.read_csv(master_csv)
    required = ["galaxy", "logSigmaHI_out", "logMbar", "logRd", "f3_scm", "delta_f3"]
    missing = [c for c in required if c not in master.columns]
    if missing:
        raise ValueError(f"Missing columns in {master_csv}: {missing}")

    rows: list[pd.DataFrame] = []
    # Synthetic radial grid in kpc, chosen to span a typical SPARC inner-to-outer disk extent.
    r = np.linspace(0.5, 12.0, n_rings)
    r_centered = (r - np.mean(r)) / np.maximum(np.ptp(r), EPS)
    for _, row in master.iterrows():
        rd = max(float(10 ** row["logRd"]), 0.2)
        sb0 = max(float(10 ** row["logSigmaHI_out"]), EPS)
        gbar0 = max(float(10 ** (row["logMbar"] - 10.5)), EPS)
        f3_base = float(row["f3_scm"])
        f3_slope = float(row["delta_f3"])

        sb = sb0 * np.exp(-r / rd)
        # Radial gbar proxy: softened power-law decline to avoid divergence at very small radii.
        gbar = gbar0 / np.power(r + GBAR_SOFTENING_KPC, GBAR_RADIAL_EXPONENT)
        # Keep gobs = gbar * (1 + F3) strictly positive by clipping F3 above -1.
        f3_profile = np.clip(f3_base + f3_slope * r_centered, -0.95, None)
        gobs = gbar * (1.0 + f3_profile)

        rows.append(
            pd.DataFrame(
                {
                    "galaxy": row["galaxy"],
                    "r": r,
                    "gbar": gbar,
                    "gobs": gobs,
                    "SB": sb,
                }
            )
        )

    return _finalize(pd.concat(rows, ignore_index=True))


def build_radial_csv(
    output_csv: str | Path = "data/sparc_175_radial.csv",
    sparc_dir: str | Path = "data/SPARC",
    master_csv: str | Path = "data/sparc_175_master_sample.csv",
) -> pd.DataFrame:
    try:
        out = build_from_rotmod(sparc_dir)
        source = "SPARC rotmod"
    except (FileNotFoundError, ValueError):
        out = build_from_master(master_csv)
        source = "sparc_175_master_sample synthetic radial proxy"

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print(f"Built radial CSV from: {source}")
    print(f"Output: {output_csv}")
    print(f"Rows: {len(out)}")
    print(f"Galaxies: {out['galaxy'].nunique()}")
    print(f"NaN count: {int(out.isna().sum().sum())}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SPARC radial CSV for intra-galaxy gradient analysis.")
    parser.add_argument("--output", default="data/sparc_175_radial.csv", help="Output CSV path.")
    parser.add_argument("--sparc-dir", default="data/SPARC", help="SPARC directory with *_rotmod.dat files.")
    parser.add_argument(
        "--master-csv",
        default="data/sparc_175_master_sample.csv",
        help="Fallback SPARC master sample CSV (used when rotmod data is unavailable).",
    )
    args = parser.parse_args()

    build_radial_csv(output_csv=args.output, sparc_dir=args.sparc_dir, master_csv=args.master_csv)


if __name__ == "__main__":
    main()
