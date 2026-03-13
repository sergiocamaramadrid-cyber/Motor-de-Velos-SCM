#!/usr/bin/env python3

"""
Calcula observables del Framework SCM para el catálogo combinado.

Entradas:
- results/combined/framework_master_catalog.csv
- results/LITTLE_THINGS_Oh2015/little_things_rotcurves.csv

Salida:
- results/combined/f3_combined_catalog.csv
"""

from pathlib import Path
import pandas as pd
import numpy as np

TAIL_THRESHOLD_R_SCALED = 0.7
F3_REFERENCE_SLOPE = 0.5


def compute_tail_slope(r, v):
    r = np.asarray(r)
    v = np.asarray(v)

    if len(r) < 3:
        return np.nan

    log_r = np.log10(r)
    log_v = np.log10(v)

    slope, _ = np.polyfit(log_r, log_v, 1)

    return slope


def compute_f3_from_rotcurve(df):
    r = df["r_scaled"].values
    v = df["v_scaled"].values

    tail_mask = r >= TAIL_THRESHOLD_R_SCALED

    r_tail = r[tail_mask]
    v_tail = v[tail_mask]

    if len(r_tail) < 3:
        return np.nan, 0

    slope = compute_tail_slope(r_tail, v_tail)

    return slope, len(r_tail)


def main():
    master_path = Path("results/combined/framework_master_catalog.csv")
    rot_path = Path("results/LITTLE_THINGS_Oh2015/little_things_rotcurves.csv")

    if not master_path.exists():
        raise FileNotFoundError(master_path)

    if not rot_path.exists():
        raise FileNotFoundError(rot_path)

    master = pd.read_csv(master_path)
    rot = pd.read_csv(rot_path)

    rows = []

    for galaxy in master["galaxy"]:
        sub = rot[rot["galaxy"] == galaxy]

        if len(sub) == 0:
            rows.append(
                {
                    "galaxy": galaxy,
                    "f3_scm": np.nan,
                    "delta_f3": np.nan,
                    "n_tail_points": 0,
                    "fit_ok": False,
                }
            )
            continue

        slope, n_tail = compute_f3_from_rotcurve(sub)

        delta_f3 = slope - F3_REFERENCE_SLOPE if not np.isnan(slope) else np.nan

        rows.append(
            {
                "galaxy": galaxy,
                "f3_scm": slope,
                "delta_f3": delta_f3,
                "n_tail_points": n_tail,
                "fit_ok": not np.isnan(slope),
            }
        )

    f3 = pd.DataFrame(rows)

    combined = master.merge(f3, on="galaxy", how="left")

    outdir = Path("results/combined")
    outdir.mkdir(parents=True, exist_ok=True)

    out = outdir / "f3_combined_catalog.csv"

    combined.to_csv(out, index=False)

    print("F3 catalog creado:")
    print(out)


if __name__ == "__main__":
    main()
