"""
scripts/pilot_f3_test.py — Pilot measurement of the F3_SCM observable.

Theory
------
The SCM F3 term is a **direct physical observable** defined as::

    F_{3,SCM} = d(log V_obs) / d(log r)  |_{r >= 0.7 * R_max}

It characterises the logarithmic slope of the observed rotation curve in the
outer region of a galaxy (r ≥ 70 % of the maximum sampled radius).  Unlike
proxy-based or model-dependent estimates this quantity is:

* **Measurable directly** from rotation-curve data.
* **Reproducible** across independent datasets.
* **Independent** of any mass-model or interpolation assumption.

Interpretation
--------------
F3_SCM ≈ 0   — flat outer rotation curve (expected for virialized systems)
F3_SCM > 0   — rising outer profile
F3_SCM < 0   — declining outer profile

Usage
-----
With SPARC rotmod files in ``data/SPARC/Rotmod``::

    python scripts/pilot_f3_test.py --sparc-dir data/SPARC/Rotmod

Custom output directory::

    python scripts/pilot_f3_test.py \\
        --sparc-dir data/SPARC/Rotmod \\
        --out       results/f3_pilot \\
        --r-max-frac 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scm_models import compute_f3_scm  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

R_MAX_FRAC_DEFAULT = 0.7
OUT_DEFAULT = "results/f3_pilot"
_SEP = "=" * 64

# Column names for the SPARC rotmod DAT format (space-separated, no header)
_ROTMOD_COLS = ["Rad", "Vobs", "errV", "Vgas", "Vdisk", "Vbul", "SBdisk", "SBbul"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_rotmod(path: Path) -> pd.DataFrame | None:
    """Load a single SPARC ``*_rotmod.dat`` file.

    Returns ``None`` if the file cannot be parsed.
    """
    try:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            names=_ROTMOD_COLS,
            dtype=float,
        )
        if df.empty or "Rad" not in df.columns or "Vobs" not in df.columns:
            return None
        return df
    except Exception:  # noqa: BLE001
        return None


def _galaxy_name(path: Path) -> str:
    """Extract galaxy name from a rotmod file path.

    ``NGC1234_rotmod.dat`` → ``NGC1234``
    """
    stem = path.stem  # removes .dat
    if stem.endswith("_rotmod"):
        return stem[: -len("_rotmod")]
    return stem


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_pilot(
    sparc_dir: str | Path,
    out_dir: str | Path | None = None,
    r_max_frac: float = R_MAX_FRAC_DEFAULT,
    verbose: bool = True,
) -> pd.DataFrame:
    """Measure F3_SCM for all galaxies found in *sparc_dir*.

    Parameters
    ----------
    sparc_dir : str or Path
        Directory containing SPARC ``*_rotmod.dat`` files.
    out_dir : str or Path or None
        If provided, write ``f3_scm_results.csv`` and a summary log there.
    r_max_frac : float
        Fraction of R_max defining the outer region (default 0.7).
    verbose : bool
        Print per-galaxy progress if True.

    Returns
    -------
    pd.DataFrame
        One row per galaxy with columns:
        ``galaxy``, ``f3_scm``, ``n_outer``, ``r_min_outer``, ``r_max``.
    """
    sparc_dir = Path(sparc_dir)
    if not sparc_dir.is_dir():
        raise FileNotFoundError(f"SPARC directory not found: {sparc_dir}")

    rotmod_files = sorted(sparc_dir.glob("*_rotmod.dat"))
    if not rotmod_files:
        raise FileNotFoundError(
            f"No *_rotmod.dat files found in {sparc_dir}.\n"
            "Download SPARC data from http://astroweb.cwru.edu/SPARC/ and place\n"
            "the Rotmod_LTG/ files in the directory above."
        )

    records = []
    for fpath in rotmod_files:
        df = _load_rotmod(fpath)
        name = _galaxy_name(fpath)
        if df is None:
            if verbose:
                print(f"  [skip] {name}: could not parse {fpath.name}",
                      file=sys.stderr)
            continue

        result = compute_f3_scm(df["Rad"].values, df["Vobs"].values,
                                r_max_frac=r_max_frac)
        records.append({"galaxy": name, **result})
        if verbose:
            f3 = result["f3_scm"]
            n = result["n_outer"]
            rmax = result["r_max"]
            label = f"{f3:+.4f}" if np.isfinite(f3) else "  nan "
            print(f"  {name:<20s}  F3_SCM={label}  n_outer={n}  R_max={rmax:.2f} kpc")

    results_df = pd.DataFrame(records, columns=[
        "galaxy", "f3_scm", "n_outer", "r_min_outer", "r_max",
    ])
    results_df["n_outer"] = results_df["n_outer"].astype(int)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "f3_scm_results.csv"
        results_df.to_csv(csv_path, index=False)

        _write_summary(results_df, out_dir / "f3_scm_summary.log",
                       r_max_frac=r_max_frac)
        if verbose:
            print(f"\n  Results written to {out_dir}")

    return results_df


def _write_summary(df: pd.DataFrame, path: Path, r_max_frac: float) -> None:
    """Write a plain-text summary of the F3_SCM pilot results."""
    valid = df["f3_scm"].dropna()
    lines = [
        _SEP,
        "  Motor de Velos SCM — F3_SCM Pilot Results",
        _SEP,
        f"  Galaxies processed    : {len(df)}",
        f"  Valid F3_SCM values   : {len(valid)}",
        f"  Outer region fraction : r >= {r_max_frac} * R_max",
        "",
    ]
    if len(valid):
        lines += [
            f"  F3_SCM  mean   : {valid.mean():+.4f}",
            f"  F3_SCM  median : {valid.median():+.4f}",
            f"  F3_SCM  std    : {valid.std():.4f}",
            f"  F3_SCM  min    : {valid.min():+.4f}",
            f"  F3_SCM  max    : {valid.max():+.4f}",
            "",
            "  Interpretation:",
            "    F3_SCM ≈ 0  →  flat outer rotation curve",
            "    F3_SCM > 0  →  rising outer profile",
            "    F3_SCM < 0  →  declining outer profile",
        ]
    lines.append(_SEP)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure F3_SCM = d(log V_obs)/d(log r)|_{r>=0.7*R_max} "
            "for all galaxies in a SPARC rotmod directory."
        )
    )
    parser.add_argument(
        "--sparc-dir",
        required=True,
        metavar="DIR",
        help="Directory containing SPARC *_rotmod.dat files.",
    )
    parser.add_argument(
        "--out",
        default=OUT_DEFAULT,
        metavar="DIR",
        help=f"Output directory (default: {OUT_DEFAULT}).",
    )
    parser.add_argument(
        "--r-max-frac",
        type=float,
        default=R_MAX_FRAC_DEFAULT,
        dest="r_max_frac",
        metavar="FRAC",
        help=(
            f"Fraction of R_max defining outer region (default: {R_MAX_FRAC_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-galaxy progress output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Entry point for the F3_SCM pilot script."""
    args = _parse_args(argv)
    return run_pilot(
        sparc_dir=args.sparc_dir,
        out_dir=args.out,
        r_max_frac=args.r_max_frac,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
