#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

A0 = 1.2e-10


def build_persistence_input(df: pd.DataFrame, max_gbar: float = 0.3 * A0) -> pd.DataFrame:
    required = {"g_obs", "g_bar"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {', '.join(sorted(missing))}")

    out = df.copy()
    out = out[np.isfinite(out["g_obs"]) & np.isfinite(out["g_bar"])]
    out = out[out["g_bar"] > 0]
    out = out[out["g_bar"] < max_gbar]

    if "Mbar" not in out.columns and "logMbar" in out.columns:
        out["Mbar"] = 10.0 ** out["logMbar"]
    if "logMbar" not in out.columns and "Mbar" in out.columns:
        out["logMbar"] = np.log10(out["Mbar"])

    out["r"] = out["g_obs"] / out["g_bar"]

    preferred_cols = ["galaxy", "Mbar", "logMbar", "g_obs", "g_bar", "r"]
    cols = [c for c in preferred_cols if c in out.columns]
    remainder = [c for c in out.columns if c not in cols]
    return out[cols + remainder].reset_index(drop=True)


def main(input_csv: str, output_csv: str, max_gbar: float = 0.3 * A0) -> None:
    df = pd.read_csv(input_csv)
    out = build_persistence_input(df, max_gbar=max_gbar)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} rows -> {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-gbar", type=float, default=0.3 * A0)
    args = parser.parse_args()
    main(args.input, args.output, args.max_gbar)
