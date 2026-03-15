from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready summary tables")
    parser.add_argument("--master", default="data/sparc_175_master.csv", help="Master catalog CSV")
    parser.add_argument(
        "--regression",
        default="results/regression/f3_regression_summary.csv",
        help="Regression summary CSV",
    )
    parser.add_argument("--outdir", default="results/paper", help="Output directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    master_path = Path(args.master)
    if not master_path.exists():
        fallback = Path("data/sparc_175_master_sample.csv")
        if fallback.exists():
            master_path = fallback

    if not master_path.exists():
        print(f"[ERROR] Master catalog not found: {args.master}")
        return 1

    df = pd.read_csv(master_path)
    cols = [c for c in ["F3", "f3_scm", "delta_f3", "beta", "beta_err"] if c in df.columns]
    summary = df[cols].describe().T.reset_index().rename(columns={"index": "metric"})
    summary.to_csv(outdir / "table_f3_summary.csv", index=False)

    reg_path = Path(args.regression)
    if reg_path.exists():
        reg_df = pd.read_csv(reg_path)
    else:
        reg_df = pd.DataFrame(
            [
                {
                    "model": "linear_regression",
                    "R2": None,
                    "RMSE": None,
                    "MAE": None,
                    "AIC": None,
                    "BIC": None,
                }
            ]
        )
    reg_df.to_csv(outdir / "table_regression_results.csv", index=False)

    print(f"Paper tables written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
