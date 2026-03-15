from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _default_input() -> Path:
    for p in [Path("data/sparc_175_master.csv"), Path("data/sparc_175_master_sample.csv")]:
        if p.exists():
            return p
    return Path("results/SPARC/sparc_175_master_sample.csv")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate F3 analysis plots")
    parser.add_argument("--input", default=str(_default_input()), help="Input master/sample catalog CSV")
    parser.add_argument("--outdir", default="figures", help="Output figures directory")
    return parser.parse_args(argv)


def _scatter(df: pd.DataFrame, x: str, y: str, out: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[x], df[y], alpha=0.8)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input not found: {input_path}")
        return 1

    df = pd.read_csv(input_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    required = {"F3", "logSigmaHI_out", "logMbar"}
    missing = required - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns for plotting: {sorted(missing)}")
        return 1

    _scatter(df, "logSigmaHI_out", "F3", outdir / "f3_vs_sigmaHI.png", "F3 vs logSigmaHI_out")
    _scatter(df, "logMbar", "F3", outdir / "f3_vs_logMbar.png", "F3 vs logMbar")

    plt.figure(figsize=(6, 4))
    df["F3"].hist(bins=15)
    plt.xlabel("F3")
    plt.ylabel("Count")
    plt.title("F3 distribution")
    plt.tight_layout()
    plt.savefig(outdir / "f3_distribution.png")
    plt.close()

    env_col = "quality_flag" if "quality_flag" in df.columns else None
    plt.figure(figsize=(6, 4))
    if env_col:
        cats = pd.Categorical(df[env_col])
        plt.scatter(cats.codes, df["F3"], alpha=0.8)
        plt.xticks(range(len(cats.categories)), list(cats.categories), rotation=30)
        plt.xlabel("environment proxy (quality_flag)")
    else:
        plt.scatter(range(len(df)), df["F3"], alpha=0.8)
        plt.xlabel("index")
    plt.ylabel("F3")
    plt.title("F3 environment scatter")
    plt.tight_layout()
    plt.savefig(outdir / "f3_environment_scatter.png")
    plt.close()

    print(f"Figures written to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
