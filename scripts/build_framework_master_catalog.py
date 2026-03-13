#!/usr/bin/env python3

"""
Construye el catálogo maestro del Framework SCM.

Combina:

SPARC
LITTLE THINGS

Salida:

results/combined/framework_master_catalog.csv
"""

from pathlib import Path

import pandas as pd


def load_little_things(path):
    df = pd.read_csv(path)

    out = pd.DataFrame()

    out["galaxy"] = df["galaxy"]
    out["source_catalog"] = "LITTLE_THINGS"
    out["framework_stage"] = "early_validation"
    out["science_role"] = "dwarf_validation_sample"

    out["dist_mpc"] = df["dist_mpc"]
    out["incl_deg"] = df["incl_deg"]

    out["rmax_kpc"] = df["rmax_kpc"]
    out["r03_kpc"] = df["r03_kpc"]
    out["v_rmax_kms"] = df["v_rmax_kms"]

    out["mgas_1e7_msun"] = df["mgas_1e7_msun"]
    out["mstar_proxy_1e7_msun"] = df["mstar_sed_1e7_msun"]

    out["logmdyn"] = df["logmdyn"]

    out["alphamin"] = df["alphamin"]

    out["rotcurve_available"] = True

    return out


def build_catalog():
    little_path = Path("results/LITTLE_THINGS_Oh2015/little_things_galaxy_table.csv")

    lt = load_little_things(little_path)

    combined = lt.copy()

    outdir = Path("results/combined")
    outdir.mkdir(parents=True, exist_ok=True)

    combined.to_csv(
        outdir / "framework_master_catalog.csv",
        index=False,
    )

    print("Master catalog creado:")
    print(outdir / "framework_master_catalog.csv")


if __name__ == "__main__":
    build_catalog()
