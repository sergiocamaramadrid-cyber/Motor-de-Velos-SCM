#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_sparc_radial_csv.py

Build a clean radial SPARC CSV from rotmod files inside a ZIP.
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

KPC_TO_M = 3.085677581e19
EPS = 1e-30


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_rotmod_zip(zip_path: str) -> pd.DataFrame:
    rows = []

    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.endswith("_rotmod.dat")]

        if not members:
            raise ValueError("No *_rotmod.dat files found inside the ZIP.")

        for member in members:
            galaxy = Path(member).name.replace("_rotmod.dat", "")

            with zf.open(member) as fh:
                raw = fh.read().decode("utf-8", errors="ignore")

            lines = []
            for line in raw.splitlines():
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                lines.append(s)

            if not lines:
                continue

            try:
                data = np.loadtxt(io.StringIO("\n".join(lines)))
            except Exception:
                continue

            if data.ndim == 1:
                data = data.reshape(1, -1)

            if data.shape[1] < 6:
                continue

            r = data[:, 0]
            vobs = data[:, 1]
            evobs = data[:, 2]
            vgas = data[:, 3]
            vdisk = data[:, 4]
            vbul = data[:, 5]

            vbar = np.sqrt(np.maximum(vgas**2 + vdisk**2 + vbul**2, 0.0))
            r_m = np.maximum(r * KPC_TO_M, EPS)

            gobs = (vobs * 1000.0) ** 2 / r_m
            gbar = (vbar * 1000.0) ** 2 / r_m

            sb = np.maximum(vdisk**2, 1e-6)
            f3 = (gobs - np.maximum(gbar, EPS)) / np.maximum(gbar, EPS)

            for i in range(len(r)):
                if not np.isfinite(r[i]) or r[i] <= 0:
                    continue
                if not np.isfinite(gobs[i]) or not np.isfinite(gbar[i]):
                    continue

                rows.append(
                    {
                        "galaxy": galaxy,
                        "r": float(r[i]),
                        "Vobs": float(vobs[i]),
                        "eVobs": float(evobs[i]),
                        "Vgas": float(vgas[i]),
                        "Vdisk": float(vdisk[i]),
                        "Vbul": float(vbul[i]),
                        "Vbar": float(vbar[i]),
                        "gobs": float(gobs[i]),
                        "gbar": float(gbar[i]),
                        "SB": float(sb[i]),
                        "F3": float(f3[i]),
                    }
                )

    if not rows:
        raise ValueError("No valid radial data extracted.")

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["galaxy", "r", "gobs", "gbar", "SB", "F3"])
    df = df[df["r"] > 0].copy()
    df = df.sort_values(["galaxy", "r"]).reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-zip", required=True)
    parser.add_argument("--output", default="data/sparc_175_radial.csv")
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.output) or ".")

    df = read_rotmod_zip(args.input_zip)
    df.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Rows: {len(df)}")
    print(f"Galaxies: {df['galaxy'].nunique()}")
    print(f"NaN: {int(df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
