"""compute_residuals_binned.py — Bin SCM/RAR model residuals by g_bar.

Reads an acceleration-pair dataset (g_bar, g_obs columns in m/s^2),
computes log-residuals  Δ = log10(g_obs) − log10(g_pred)  where g_pred
is the RAR interpolation formula with the Motor-de-Velos a0 = 1.2e-10 m/s²,
then bins those residuals in equal-width log10(g_bar) bins and writes the
summary statistics to ``results/residuals_binned_v02.csv``.

Usage
-----
    python scripts/compute_residuals_binned.py [--data PATH] [--out PATH]
         [--bins N] [--min-count N]

Defaults
--------
    --data      data/sparc_rar_sample.csv
    --out       results/residuals_binned_v02.csv
    --bins      15
    --min-count 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Motor-de-Velos / MOND acceleration scale  [m s⁻²]
A0_SI: float = 1.2e-10


# ---------------------------------------------------------------------------
# RAR model (SI units — accelerations in m s⁻²)
# ---------------------------------------------------------------------------

def nu_rar(x: np.ndarray) -> np.ndarray:
    """RAR interpolation function  ν(x) = 1 / (1 − exp(−√x)).

    Parameters
    ----------
    x : array_like
        Dimensionless ratio g_bar / a0.

    Returns
    -------
    nu : ndarray
        Interpolation factor  ≥ 1.
    """
    x = np.asarray(x, dtype=float)
    sqrtx = np.sqrt(np.clip(x, 1e-30, None))
    return 1.0 / (1.0 - np.exp(-sqrtx))


def g_pred_rar(g_bar: np.ndarray, a0: float = A0_SI) -> np.ndarray:
    """Predicted observed acceleration from the RAR formula.

    g_pred = g_bar · ν(g_bar / a0)

    Parameters
    ----------
    g_bar : array_like
        Baryonic centripetal acceleration [m s⁻²].
    a0 : float, optional
        Motor-de-Velos acceleration scale [m s⁻²].

    Returns
    -------
    g_pred : ndarray
        Predicted g_obs values [m s⁻²].
    """
    g_bar = np.asarray(g_bar, dtype=float)
    return g_bar * nu_rar(g_bar / a0)


# ---------------------------------------------------------------------------
# Residual computation and binning
# ---------------------------------------------------------------------------

def compute_log_residuals(
    g_bar: np.ndarray,
    g_obs: np.ndarray,
    a0: float = A0_SI,
) -> np.ndarray:
    """Log-residuals  Δ = log10(g_obs) − log10(g_pred).

    Parameters
    ----------
    g_bar : array_like
        Baryonic acceleration [m s⁻²].
    g_obs : array_like
        Observed centripetal acceleration [m s⁻²].
    a0 : float, optional
        Motor-de-Velos acceleration scale.

    Returns
    -------
    delta : ndarray
        Log-residuals in dex.
    """
    g_bar = np.asarray(g_bar, dtype=float)
    g_obs = np.asarray(g_obs, dtype=float)
    g_pred = g_pred_rar(g_bar, a0=a0)
    return np.log10(g_obs) - np.log10(g_pred)


def bin_residuals(
    g_bar: np.ndarray,
    delta: np.ndarray,
    n_bins: int = 15,
    min_count: int = 3,
) -> pd.DataFrame:
    """Bin log-residuals into equal-width log10(g_bar) bins.

    Parameters
    ----------
    g_bar : array_like
        Baryonic acceleration values [m s⁻²].
    delta : array_like
        Log-residuals (Δ = log10(g_obs/g_pred)) [dex].
    n_bins : int, optional
        Number of equal-width bins in log10(g_bar) space (default 15).
    min_count : int, optional
        Minimum number of data points required to retain a bin (default 3).

    Returns
    -------
    df : pd.DataFrame
        Table with columns:
        ``g_bar_center``, ``median_residual``, ``mad_residual``, ``count``.

    Notes
    -----
    *MAD* (median absolute deviation) is computed as the median of |Δ − median(Δ)|
    within each bin — a robust scatter estimator.
    """
    g_bar = np.asarray(g_bar, dtype=float)
    delta = np.asarray(delta, dtype=float)

    log_g = np.log10(g_bar)
    edges = np.linspace(log_g.min(), log_g.max(), n_bins + 1)

    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Include right edge only for the last bin
        if i < n_bins - 1:
            mask = (log_g >= lo) & (log_g < hi)
        else:
            mask = (log_g >= lo) & (log_g <= hi)

        cnt = int(mask.sum())
        if cnt < min_count:
            continue

        d_bin = delta[mask]
        med = float(np.median(d_bin))
        mad = float(np.median(np.abs(d_bin - med)))
        center = float(10 ** ((lo + hi) / 2.0))
        rows.append(
            {
                "g_bar_center": center,
                "median_residual": med,
                "mad_residual": mad,
                "count": cnt,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> pd.DataFrame:
    """Load an acceleration-pair CSV (g_bar, g_obs columns required).

    Parameters
    ----------
    path : Path
        Path to the CSV file.  Comment lines starting with ``#`` are skipped.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with at least ``g_bar`` and ``g_obs`` columns.

    Raises
    ------
    ValueError
        If required columns are absent or data contain non-positive values.
    """
    df = pd.read_csv(path, comment="#")
    df.columns = df.columns.str.strip()

    for col in ("g_bar", "g_obs"):
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found in {path.name}. "
                f"Available columns: {list(df.columns)}"
            )

    df = df.dropna(subset=["g_bar", "g_obs"])
    df = df[(df["g_bar"] > 0) & (df["g_obs"] > 0)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError(f"No valid data rows found in {path.name}.")

    return df


def write_csv(df: pd.DataFrame, out_path: Path, metadata: dict) -> None:
    """Write the binned-residuals DataFrame to a CSV with metadata header.

    The output starts with three comment lines carrying scalar metadata
    (``bins_effective``, ``g_bar_center_min``, ``g_bar_center_max``)
    followed by the tabular data.

    Parameters
    ----------
    df : pd.DataFrame
        Binned residuals table.
    out_path : Path
        Destination file path.
    metadata : dict
        Scalar metadata to embed as ``# key: value`` comments.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for key, val in metadata.items():
            fh.write(f"# {key}: {val}\n")
        df.to_csv(fh, index=False, float_format="%.6e")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute log-residuals of the RAR/SCM model and write a "
            "binned summary CSV (residuals_binned_v02.csv)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        default="data/sparc_rar_sample.csv",
        metavar="PATH",
        help="Input CSV with g_bar and g_obs columns [m s⁻²].",
    )
    p.add_argument(
        "--out",
        default="results/residuals_binned_v02.csv",
        metavar="PATH",
        help="Output path for the binned-residuals CSV.",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=15,
        metavar="N",
        help="Number of equal-width log10(g_bar) bins.",
    )
    p.add_argument(
        "--min-count",
        type=int,
        default=3,
        metavar="N",
        dest="min_count",
        help="Minimum data points per bin (bins below threshold are dropped).",
    )
    p.add_argument(
        "--a0",
        type=float,
        default=A0_SI,
        metavar="A0",
        help="Motor-de-Velos / MOND acceleration scale [m s⁻²].",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point for the residuals-binning pipeline."""
    args = _build_parser().parse_args(argv)

    data_path = Path(args.data)
    out_path = Path(args.out)

    print(f"Loading data from {data_path} …", flush=True)
    df = load_dataset(data_path)
    print(f"  {len(df)} valid data points loaded.", flush=True)

    delta = compute_log_residuals(df["g_bar"].values, df["g_obs"].values, a0=args.a0)
    print(
        f"  Residuals  Δ = log10(g_obs/g_pred): "
        f"median={np.median(delta):.4f} dex, "
        f"MAD={np.median(np.abs(delta - np.median(delta))):.4f} dex",
        flush=True,
    )

    binned = bin_residuals(
        df["g_bar"].values, delta, n_bins=args.bins, min_count=args.min_count
    )

    if binned.empty:
        print("ERROR: no bins survived the min-count filter.", file=sys.stderr)
        sys.exit(1)

    bins_effective = len(binned)
    g_bar_min = float(binned["g_bar_center"].min())
    g_bar_max = float(binned["g_bar_center"].max())

    metadata = {
        "bins_effective": bins_effective,
        "g_bar_center_min": f"{g_bar_min:.6e}",
        "g_bar_center_max": f"{g_bar_max:.6e}",
    }

    write_csv(binned, out_path, metadata)
    print(
        f"Written {bins_effective} bins to {out_path}\n"
        f"  g_bar range: [{g_bar_min:.3e}, {g_bar_max:.3e}] m/s²",
        flush=True,
    )


if __name__ == "__main__":
    main()
