from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "logSigmaHI_out",
    "logMbar",
    "logRd",
    "F3",
    "f3_scm",
    "delta_f3",
    "fit_ok",
    "quality_flag",
    "beta",
    "beta_err",
    "reliable",
    "friction_slope",
    "velo_inerte_flag",
]


def _default_source() -> Path:
    for p in [Path("data/sparc_175_master_sample.csv"), Path("results/SPARC/sparc_175_master_sample.csv")]:
        if p.exists():
            return p
    return Path("results/SPARC/sparc_175_master_sample.csv")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SPARC master catalog")
    parser.add_argument("--input", default=str(_default_source()), help="Input sample/source CSV")
    parser.add_argument("--out", default="data/sparc_175_master.csv", help="Output master catalog CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    src = Path(args.input)
    if not src.exists():
        print(f"[ERROR] Source CSV not found: {src}")
        return 1

    df = pd.read_csv(src)

    if "f3_scm" not in df.columns and "F3" in df.columns:
        df["f3_scm"] = df["F3"]
    if "delta_f3" not in df.columns and "f3_scm" in df.columns:
        df["delta_f3"] = df["f3_scm"] - 0.5
    if "friction_slope" not in df.columns and "beta" in df.columns:
        df["friction_slope"] = df["beta"]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[ERROR] Cannot build master catalog; missing columns: {missing}")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Master catalog written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
