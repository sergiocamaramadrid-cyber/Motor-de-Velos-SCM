import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--mass-cut", type=float, default=10.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.data)

    required = ["galaxy", "logM", "MHI", "Rdisk", "slope_tail"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].dropna().copy()
    df = df[(df["MHI"] > 0) & (df["Rdisk"] > 0)]

    df["env_new"] = np.log10(df["MHI"] / (df["Rdisk"] ** 2))
    df["env_std"] = (df["env_new"] - df["env_new"].mean()) / df["env_new"].std()

    high = df[df["logM"] >= args.mass_cut].copy()

    model = smf.ols("slope_tail ~ env_std + logM", data=high).fit(cov_type="HC3")
    table = model.summary2().tables[1]
    table.to_csv(os.path.join(args.outdir, "table_ols_hc3.csv"))

    plt.figure(figsize=(6, 5))
    plt.scatter(high["env_std"], high["slope_tail"], s=25, alpha=0.8)
    plt.xlabel(r"$\mathrm{env}_{\mathrm{std}} = z[\log_{10}(M_{\rm HI}/R_{\rm disk}^2)]$")
    plt.ylabel(r"$slope_{\rm tail}$")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "fig_envstd_slope_tail.png"), dpi=300)

    print("N total:", len(df))
    print("N high:", len(high))
    print(table)


if __name__ == "__main__":
    main()
