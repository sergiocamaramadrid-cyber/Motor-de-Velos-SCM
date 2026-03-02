"""
scripts/generate_f3_catalog_from_contract.py — Deep-regime β catalog (v2).

Reads a contract-compliant Parquet (or CSV) produced by
``ingest_big_sparc_contract.py`` (or any ingestor that satisfies the SCM data
contract), fits the deep-MOND slope β per galaxy, and writes an F3 catalog
suitable for downstream analysis.

Physics recap
-------------
In the deep-velos / deep-MOND regime (v_bar ≪ v_char):

    v_obs ≈ (v_bar · v_char)^(1/2)    →    β ≡ d log v_obs / d log v_bar ≈ 0.5

β is estimated via OLS in log–log space using only the "deep" points of each
galaxy (v_bar < ``--deep-threshold`` × ``--v-char``).

Output
------
``<out-dir>/f3_beta_catalog.csv`` with columns:

  galaxy, n_total, n_deep, beta, beta_err, r_value, p_value,
  delta_from_mond, verdict

Usage
-----
::

    python scripts/generate_f3_catalog_from_contract.py \\
        --contract data/big_sparc/contract/big_sparc_contract.parquet

    python scripts/generate_f3_catalog_from_contract.py \\
        --contract data/my_survey.parquet \\
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
OUTPUT_FILENAME = "f3_beta_catalog.csv"

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
        "beta": float("nan"),
        "beta_err": float("nan"),
        "r_value": float("nan"),
        "p_value": float("nan"),
        "delta_from_mond": float("nan"),
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

    base.update(
        beta=float(slope),
        beta_err=float(stderr),
        r_value=float(r_value),
        p_value=float(p_value),
        delta_from_mond=float(delta),
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


def generate_catalog(
    contract_path: str | Path,
    out_dir: str | Path,
    v_char: float = V_CHAR_DEFAULT_KMS,
    deep_threshold: float = DEEP_THRESHOLD_DEFAULT,
    min_deep: int = MIN_DEEP_DEFAULT,
) -> pd.DataFrame:
    """Generate the F3 β catalog from a contract-compliant table.

    Parameters
    ----------
    contract_path:
        Path to the unified contract CSV or Parquet.
    out_dir:
        Directory where ``f3_beta_catalog.csv`` is written.
    v_char:
        Characteristic velocity in km/s.
    deep_threshold:
        Fraction of v_char defining the deep regime.
    min_deep:
        Minimum number of deep points for a reliable β estimate.

    Returns
    -------
    pd.DataFrame
        Per-galaxy β catalog.
    """
    df = read_table(contract_path)
    validate_contract(df)

    rows = []
    for galaxy, sub in df.groupby("galaxy"):
        result = _fit_beta(sub, v_char=v_char,
                           deep_threshold=deep_threshold, min_deep=min_deep)
        result["galaxy"] = galaxy
        rows.append(result)

    catalog = pd.DataFrame(rows)[
        [
            "galaxy", "n_total", "n_deep", "beta", "beta_err",
            "r_value", "p_value", "delta_from_mond", "verdict",
        ]
    ].sort_values("galaxy").reset_index(drop=True)

    out_path = Path(out_dir) / OUTPUT_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)
    print(f"  F3 catalog written: {out_path}  ({len(catalog)} galaxies)")
    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-galaxy β catalog from a contract-compliant table."
    )
    parser.add_argument(
        "--contract", required=True, metavar="FILE",
        help="Path to the unified contract CSV or Parquet.",
    )
    parser.add_argument(
        "--out", default="results/f3_v2", metavar="DIR",
        help="Output directory (default: results/f3_v2).",
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
    return generate_catalog(
        contract_path=args.contract,
        out_dir=args.out,
        v_char=args.v_char,
        deep_threshold=args.deep_threshold,
        min_deep=args.min_deep,
    )


if __name__ == "__main__":
    main()
