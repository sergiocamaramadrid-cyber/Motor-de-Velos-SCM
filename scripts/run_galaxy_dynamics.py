"""
SCM-Galaxy-Dynamics — outer rotation-curve slope analysis.

Usage
-----
    python scripts/run_galaxy_dynamics.py --data data/your_dataset.csv

Required input columns: galaxy, logM, MHI, Rdisk, slope_tail

The internal proxy used is:
    env_new = log10(MHI / Rdisk^2)
    env_std = z-score(env_new)

This is an internal HI surface-density proxy, NOT an external environment
measurement.  Do not describe env_std as an external environmental variable.
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser(
        description="SCM-Galaxy-Dynamics: outer rotation-curve slope analysis."
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Input CSV with columns: galaxy, logM, MHI, Rdisk, slope_tail",
    )
    parser.add_argument(
        "--mass-cut",
        type=float,
        default=10.0,
        help="log10(M/Msun) threshold for high-mass subsample (default: 10.0)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    required = ["galaxy", "logM", "MHI", "Rdisk", "slope_tail"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].dropna().copy()
    df = df[(df["MHI"] > 0) & (df["Rdisk"] > 0)]

    # Internal HI surface-density proxy (z-scored)
    df["env_new"] = np.log10(df["MHI"] / (df["Rdisk"] ** 2))
    df["env_std"] = (df["env_new"] - df["env_new"].mean()) / df["env_new"].std()

    high = df[df["logM"] >= args.mass_cut].copy()

    print("TOTAL:", len(df))
    print("HIGH:", len(high))
    print("mass_cut:", args.mass_cut)

    model = smf.ols(
        "slope_tail ~ env_std + logM",
        data=high,
    ).fit(cov_type="HC3")

    print(model.summary2().tables[1])


if __name__ == "__main__":
    main()
