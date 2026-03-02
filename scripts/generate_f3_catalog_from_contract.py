"""
scripts/generate_f3_catalog_from_contract.py — Deep-regime β (friction slope)
catalog (v2).

Reads a contract-compliant Parquet (or CSV) produced by
``ingest_sparc_contract.py``, ``ingest_big_sparc_contract.py``, or any
ingestor that satisfies the SCM data contract.  Fits the deep-MOND slope β
per galaxy and writes an F3 catalog suitable for downstream population
analysis.

Physics recap
-------------
In the deep-velos / deep-MOND regime (v_bar ≪ v_char):

    v_obs ≈ (v_bar · v_char)^(1/2)    →    β ≡ d log v_obs / d log v_bar ≈ 0.5

β is called the *friction slope* in the v2 catalog.  A ``velo_inerte_flag``
is set to ``True`` when β deviates significantly from 0.5 (structural
deviation > 3σ), indicating the galaxy is outside the deep-velos regime.

Output
------
* When ``--out`` ends with ``.parquet`` or ``.pq``, write a Parquet file
  directly at that path.
* Otherwise treat ``--out`` as a directory and write
  ``f3_beta_catalog.parquet`` inside it.

Catalog columns
---------------
  galaxy, n_total, n_deep, friction_slope, friction_slope_err,
  r_value, p_value, delta_from_mond, velo_inerte_flag, verdict

Usage
-----
::

    # From a contract directory (auto-discovers *.parquet inside)
    python scripts/generate_f3_catalog_from_contract.py \\
        --data-dir data/SPARC/processed_contract \\
        --out results/f3_catalog_sparc_from_contract.parquet

    # From an explicit contract file
    python scripts/generate_f3_catalog_from_contract.py \\
        --contract data/big_sparc/contract/big_sparc_contract.parquet \\
        --min-deep 5 \\
        --deep-threshold 0.3 \\
        --v-char 210.0 \\
        --out results/f3_v2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from contract_utils import read_table, validate_contract

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

V_CHAR_DEFAULT_KMS = 210.0       # characteristic velocity (km/s)
DEEP_THRESHOLD_DEFAULT = 0.3     # v_bar < threshold × v_char  →  "deep"
MIN_DEEP_DEFAULT = 10            # minimum deep points for a reliable β
EXPECTED_BETA = 0.5              # MOND / deep-velos prediction
OUTPUT_FILENAME = "f3_beta_catalog.parquet"

# ---------------------------------------------------------------------------
# Core per-galaxy fit
# ---------------------------------------------------------------------------


def _fit_beta(sub: pd.DataFrame, v_char: float,
              deep_threshold: float, min_deep: int) -> dict:
    """Fit β for a single galaxy subset."""
    vbar = sub["vbar_kms"].to_numpy(dtype=float)
    vobs = sub["vobs_kms"].to_numpy(dtype=float)

    # guard against non-positive values before log
    mask_pos = (vbar > 0) & (vobs > 0)
    vbar = vbar[mask_pos]
    vobs = vobs[mask_pos]

    deep_mask = vbar < deep_threshold * v_char
    n_total = len(vbar)
    n_deep = int(deep_mask.sum())

    base: dict = {
        "n_total": n_total,
        "n_deep": n_deep,
        "friction_slope": float("nan"),
        "friction_slope_err": float("nan"),
        "r_value": float("nan"),
        "p_value": float("nan"),
        "delta_from_mond": float("nan"),
        "velo_inerte_flag": False,
        "verdict": "",
    }

    if n_deep < 2:
        base["verdict"] = (
            f"⚠️  Insufficient deep points for regression (need ≥2, got {n_deep})"
        )
        return base

    log_vbar = np.log10(vbar[deep_mask])
    log_vobs = np.log10(vobs[deep_mask])

    # linregress requires non-constant x
    if np.ptp(log_vbar) == 0:
        base["verdict"] = "⚠️  All deep-regime vbar values are identical — cannot fit β"
        return base

    slope, _intercept, r_value, p_value, stderr = linregress(log_vbar, log_vobs)
    delta = slope - EXPECTED_BETA

    # velo_inerte_flag: True when β deviates > 3σ from the MOND prediction
    is_velo_inerte = abs(delta) > 3 * stderr if stderr > 0 else False

    base.update(
        friction_slope=float(slope),
        friction_slope_err=float(stderr),
        r_value=float(r_value),
        p_value=float(p_value),
        delta_from_mond=float(delta),
        velo_inerte_flag=bool(is_velo_inerte),
    )

    if n_deep < min_deep:
        base["verdict"] = (
            f"⚠️  Only {n_deep} deep points — result may not be reliable "
            f"(need ≥{min_deep})"
        )
    elif abs(delta) <= 2 * stderr:
        base["verdict"] = "✅  β consistent with MOND/deep-velos (β ≈ 0.5)"
    elif slope < EXPECTED_BETA - 3 * stderr:
        base["verdict"] = (
            f"⚠️  β = {slope:.3f} — significant structural deviation below 0.5"
        )
    elif slope > EXPECTED_BETA + 3 * stderr:
        base["verdict"] = (
            f"⚠️  β = {slope:.3f} — significant structural deviation above 0.5"
        )
    else:
        base["verdict"] = f"ℹ️  β = {slope:.3f} — mild deviation (within 3σ)"

    return base


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------


def _resolve_contract_path(contract_path: str | Path | None,
                           data_dir: str | Path | None) -> Path:
    """Return the contract Parquet path from either explicit path or directory."""
    if contract_path is not None:
        p = Path(contract_path)
        if not p.exists():
            raise FileNotFoundError(f"Contract file not found: {p}")
        return p
    if data_dir is not None:
        d = Path(data_dir)
        # Prefer known filenames, then fall back to any *.parquet in the directory
        for name in ("sparc_contract.parquet", "big_sparc_contract.parquet"):
            candidate = d / name
            if candidate.exists():
                return candidate
        parquets = sorted(d.glob("*.parquet"))
        if parquets:
            return parquets[0]
        raise FileNotFoundError(
            f"No contract Parquet found in {d}. "
            "Run ingest_sparc_contract.py or ingest_big_sparc_contract.py first."
        )
    raise ValueError("Provide either --contract FILE or --data-dir DIR.")


def _resolve_out_path(out: str | Path) -> tuple[Path, bool]:
    """Return (output_path, is_parquet).

    If *out* ends with ``.parquet`` or ``.pq`` it is treated as a file path;
    otherwise it is treated as a directory and the default filename is appended.
    """
    p = Path(out)
    if p.suffix in {".parquet", ".pq"}:
        return p, True
    return p / OUTPUT_FILENAME, True  # always Parquet in v2


def generate_catalog(
    contract_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    v_char: float = V_CHAR_DEFAULT_KMS,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    min_deep: int = MIN_DEEP_DEFAULT,
    *,
    data_dir: str | Path | None = None,
    out: str | Path | None = None,
) -> pd.DataFrame:
    """Generate the F3 friction-slope catalog from a contract-compliant table.

    Parameters
    ----------
    contract_path:
        Path to the unified contract CSV or Parquet.  Mutually exclusive with
        *data_dir*.
    out_dir:
        Directory where the catalog is written (legacy parameter, kept for
        backward compatibility — the file is now always Parquet).
    v_char:
        Characteristic velocity in km/s.
    deep_threshold:
        Fraction of v_char defining the deep regime.
    min_deep:
        Minimum number of deep points for a reliable β estimate.
    data_dir:
        Directory containing a contract Parquet; auto-discovers the file.
        Mutually exclusive with *contract_path*.
    out:
        Output file path (if ending in ``.parquet``/``.pq``) or directory.
        Takes precedence over *out_dir*.

    Returns
    -------
    pd.DataFrame
        Per-galaxy friction-slope catalog.
    """
    # Resolve input
    cp = _resolve_contract_path(contract_path, data_dir)
    df = read_table(cp)
    validate_contract(df)

    rows = []
    for galaxy, sub in df.groupby("galaxy"):
        result = _fit_beta(sub, v_char=v_char,
                           deep_threshold=deep_threshold, min_deep=min_deep)
        result["galaxy"] = galaxy
        rows.append(result)

    catalog = pd.DataFrame(rows)[
        [
            "galaxy", "n_total", "n_deep",
            "friction_slope", "friction_slope_err",
            "r_value", "p_value", "delta_from_mond",
            "velo_inerte_flag", "verdict",
        ]
    ].sort_values("galaxy").reset_index(drop=True)

    # Resolve output path
    _out = out if out is not None else (out_dir if out_dir is not None else "results/f3_v2")
    out_path, _ = _resolve_out_path(_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(out_path, index=False)
    print(f"  F3 catalog written: {out_path}  ({len(catalog)} galaxies)")
    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate per-galaxy friction-slope (β) catalog from a "
            "contract-compliant table."
        )
    )
    # Input: explicit file or directory
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--contract", default=None, metavar="FILE",
        help="Path to the unified contract CSV or Parquet.",
    )
    input_group.add_argument(
        "--data-dir", default=None, metavar="DIR",
        help=(
            "Directory containing a contract Parquet (e.g. from "
            "ingest_sparc_contract.py).  Auto-discovers the Parquet file."
        ),
    )
    parser.add_argument(
        "--out", default="results/f3_v2", metavar="PATH",
        help=(
            "Output file path (if ending in .parquet) or directory "
            "(default: results/f3_v2).  The catalog is always written as Parquet."
        ),
    )
    parser.add_argument(
        "--v-char", type=float, default=V_CHAR_DEFAULT_KMS, metavar="KMS",
        help=f"Characteristic velocity in km/s (default: {V_CHAR_DEFAULT_KMS}).",
    )
    parser.add_argument(
        "--deep-threshold", type=float, default=DEEP_THRESHOLD_DEFAULT,
        metavar="FRAC",
        help=(
            f"Fraction of v_char defining the deep regime "
            f"(default: {DEEP_THRESHOLD_DEFAULT})."
        ),
    )
    parser.add_argument(
        "--min-deep", type=int, default=MIN_DEEP_DEFAULT, metavar="N",
        help=(
            f"Minimum deep-regime points for a reliable β fit "
            f"(default: {MIN_DEEP_DEFAULT}).  Galaxies with fewer points "
            "are flagged in the verdict column."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = _parse_args(argv)
    if args.contract is None and args.data_dir is None:
        # default to --data-dir behaviour with a sensible default path
        args.data_dir = "."
    return generate_catalog(
        contract_path=args.contract,
        data_dir=args.data_dir,
        out=args.out,
        v_char=args.v_char,
        deep_threshold=args.deep_threshold,
        min_deep=args.min_deep,
    )


if __name__ == "__main__":
    main()
