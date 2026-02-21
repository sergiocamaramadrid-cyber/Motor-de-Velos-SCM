"""scm_sparc_full_analysis.py — Motor de Velos SCM main analysis entry point.

Fits Baseline-SCM and Universal-SCM models to one or more rotation-curve CSV
files, writes per-galaxy results to a summary CSV and prints a brief executive
summary.

Usage
-----
Single galaxy:
    python scm_sparc_full_analysis.py data/GXY_D13.8_V144_SCM_01.csv

All CSVs in a directory:
    python scm_sparc_full_analysis.py --data-dir data/ --outdir results/

Full options:
    python scm_sparc_full_analysis.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make sure the src package is importable when run as a script
sys.path.insert(0, str(Path(__file__).parent))
from src.fitting import fit_p0_local
from src.scm_models import A0_DEFAULT


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Motor de Velos SCM — full rotation-curve analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "csv_files",
        nargs="*",
        metavar="CSV",
        help="One or more rotation-curve CSV files to analyse.",
    )
    p.add_argument(
        "--data-dir",
        metavar="DIR",
        default=None,
        help="Directory to search for *.csv files (used if no CSV_FILES given).",
    )
    p.add_argument(
        "--outdir",
        metavar="DIR",
        default="results",
        help="Output directory for summary files.",
    )
    p.add_argument(
        "--a0",
        type=float,
        default=A0_DEFAULT,
        metavar="A0",
        help="Fixed Motor-de-Velos acceleration scale [(km/s)^2/kpc] for Universal-SCM.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed (currently unused; reserved for MC tests).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-galaxy results during processing.",
    )
    return p


# ---------------------------------------------------------------------------
# Analysis runner
# ---------------------------------------------------------------------------

def _collect_csv_files(args) -> list[Path]:
    files: list[Path] = []
    for f in args.csv_files:
        p = Path(f)
        if not p.exists():
            print(f"WARNING: file not found: {p}", file=sys.stderr)
        else:
            files.append(p)

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if not data_dir.is_dir():
            print(f"ERROR: --data-dir is not a directory: {data_dir}", file=sys.stderr)
            sys.exit(1)
        found = sorted(data_dir.glob("*.csv"))
        if not found:
            print(f"WARNING: no *.csv files found in {data_dir}", file=sys.stderr)
        files.extend(found)

    if not files:
        print("ERROR: no CSV files to analyse. Provide CSV paths or --data-dir.",
              file=sys.stderr)
        sys.exit(1)

    # De-duplicate (preserve order)
    seen = set()
    unique: list[Path] = []
    for f in files:
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def run_analysis(csv_files: list[Path], a0_fixed: float, verbose: bool) -> pd.DataFrame:
    """Fit all galaxies and return a results DataFrame."""
    rows = []
    for i, csv_path in enumerate(csv_files, 1):
        try:
            res = fit_p0_local(csv_path, a0_fixed=a0_fixed)
        except Exception as exc:
            print(f"  [{i}/{len(csv_files)}] FAILED {csv_path.name}: {exc}",
                  file=sys.stderr)
            rows.append({"galaxy_id": csv_path.stem, "fit_success": False,
                         "notes": str(exc)})
            continue

        row = {
            "galaxy_id":           res["galaxy"],
            "n_points":            res["n_points"],
            "k_params":            res["k_params"],
            "chi2_baseline":       round(res["chi2_baseline"], 4),
            "chi2_universal":      round(res["chi2_universal"], 4),
            "AICc_baseline":       round(res["aicc_baseline"], 4),
            "AICc_universal":      round(res["aicc_universal"], 4),
            "delta_AICc":          round(res["delta_aicc"], 4),
            "prefers_universal":   res["delta_aicc"] < 0,
            "Upsilon_baseline":    round(res["params_baseline"][0], 5),
            "log10_a0_baseline":   round(res["params_baseline"][1], 5),
            "Upsilon_universal":   round(res["params_universal"][0], 5),
            "log10_Vext_universal": round(res["params_universal"][1], 5),
            "fit_success_baseline": res["fit_success_baseline"],
            "fit_success_universal": res["fit_success_universal"],
            "fit_success":          res["fit_success_baseline"] and res["fit_success_universal"],
            "notes":               "",
        }
        rows.append(row)

        if verbose:
            pref = "Universal" if row["prefers_universal"] else "Baseline"
            print(f"  [{i}/{len(csv_files)}] {row['galaxy_id']:<30}  "
                  f"ΔAICc={row['delta_AICc']:+.3f}  ({pref} preferred)")

    return pd.DataFrame(rows)


def save_outputs(df: pd.DataFrame, outdir: Path, args) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Full results CSV
    csv_out = outdir / "universal_term_comparison_full.csv"
    df.to_csv(csv_out, index=False)
    print(f"\nResults saved to: {csv_out}")

    # 2. Executive summary
    ok = df[df["fit_success"] == True]
    n_total  = len(ok)
    n_prefer = int(ok["prefers_universal"].sum())
    frac     = n_prefer / n_total * 100 if n_total > 0 else float("nan")
    mean_d   = ok["delta_AICc"].mean() if n_total > 0 else float("nan")
    min_d    = ok["delta_AICc"].min()  if n_total > 0 else float("nan")
    max_d    = ok["delta_AICc"].max()  if n_total > 0 else float("nan")

    summary_lines = [
        "Motor de Velos – Universal Term Comparison  (SCM Analysis)",
        "=" * 60,
        f"Total galaxies analysed     : {n_total}",
        f"Prefer Universal-SCM        : {n_prefer}/{n_total} ({frac:.1f}%)",
        f"Mean ΔAICc                  : {mean_d:.3f}",
        f"Min  ΔAICc                  : {min_d:.3f}",
        f"Max  ΔAICc                  : {max_d:.3f}",
        "",
        "ΔAICc = AICc(Universal) − AICc(Baseline); negative → Universal preferred.",
        f"a0_fixed = {args.a0:.1f} (km/s)^2/kpc  (Motor-de-Velos acceleration scale)",
    ]
    txt_out = outdir / "executive_summary.txt"
    with open(txt_out, "w") as fh:
        fh.write("\n".join(summary_lines) + "\n")
    print(f"Summary saved to: {txt_out}")

    # 3. Metadata
    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_galaxies": n_total,
        "a0_fixed": args.a0,
        "seed": args.seed,
        "outdir": str(outdir),
        "csv_files": [str(f) for f in _collect_csv_files(args)],
    }
    with open(outdir / "run_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    csv_files = _collect_csv_files(args)
    print(f"Motor de Velos SCM analysis — {len(csv_files)} galaxy/galaxies")

    t0  = time.time()
    df  = run_analysis(csv_files, a0_fixed=args.a0, verbose=args.verbose)
    elapsed = time.time() - t0

    if not df.empty:
        outdir = Path(args.outdir)
        save_outputs(df, outdir, args)

        ok = df[df.get("fit_success", True) == True]
        if not ok.empty:
            first = ok.iloc[0]
            print(f"\nFirst ΔAICc  [{first['galaxy_id']}]  =  {first['delta_AICc']:+.4f}")

    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
