"""scm_oos_fit.py — SCM v0.1 out-of-sample (OOS) fitting in acceleration space.

Fits the Motor de Velos SCM acceleration model to a CSV dataset using an
80/20 train/test split, then evaluates the out-of-sample negative
log-likelihood and AICc on the held-out test set.

The model
---------
    g_pred(g_bar; a0, beta) = g_bar + a0 · (√(1 + (g_bar/a0)^beta) − 1)

    k = 2 free parameters: (a0, beta)

Limits:
    beta = 1, g_bar ≪ a0  →  g_pred ≈ √(g_bar · a0)   (deep-MOND)
    beta = 1, g_bar ≫ a0  →  g_pred ≈ g_bar             (Newtonian)

Usage
-----
    python scripts/scm_oos_fit.py
    python scripts/scm_oos_fit.py --data data/sparc_rar_sample.csv
    python scripts/scm_oos_fit.py --data sparc.csv --seed 42 --test-size 0.3

Column auto-detection
---------------------
The loader looks for the first matching column name from each group:
    g_bar  : "g_bar", "gbar", "g_bary"
    g_obs  : "g_obs", "gobs", "g_total"
    m_bar  : "m_bar", "M_bar", "Mb"
    g_err  : "g_err", "dg", "e_gobs"  (optional; σ = 1 when absent)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# Allow running from repository root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scm_models import aicc_from_nll, nll_gauss_accel

# ---------------------------------------------------------------------------
# Default configuration (frozen; match problem statement)
# ---------------------------------------------------------------------------
SEED: int = 20260211
TEST_SIZE: float = 0.2
DATA_FILE: str = "data/sparc_rar_sample.csv"

# Initial parameter guess: a0 [m s^-2], beta [dimensionless]
P0 = [1.2e-10, 1.0]

# Number of free parameters
K: int = 2


# ---------------------------------------------------------------------------
# Column auto-detection
# ---------------------------------------------------------------------------
_COL_CANDIDATES = {
    "g_bar": ["g_bar", "gbar", "g_bary"],
    "g_obs": ["g_obs", "gobs", "g_total"],
    "m_bar": ["m_bar", "M_bar", "Mb"],
    "g_err": ["g_err", "dg", "e_gobs"],
}


def _find_column(df: pd.DataFrame, role: str, required: bool = True) -> str | None:
    """Return the first matching column name for *role*, or None / raises."""
    for cand in _COL_CANDIDATES[role]:
        if cand in df.columns:
            return cand
    if required:
        raise ValueError(
            f"Could not find a column for '{role}'. "
            f"Expected one of: {_COL_CANDIDATES[role]}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


# ---------------------------------------------------------------------------
# Core fitting function (importable for testing)
# ---------------------------------------------------------------------------

def run_oos_fit(
    data_file: str = DATA_FILE,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
    p0=None,
) -> dict:
    """Load data, split, fit SCM, evaluate OOS NLL and AICc.

    Parameters
    ----------
    data_file : str
        Path to the input CSV file.
    seed : int, optional
        Random seed for the train/test split.
    test_size : float, optional
        Fraction of data held out for OOS evaluation (default 0.2).
    p0 : list, optional
        Initial parameter guess ``[a0, beta]``.  Defaults to ``[1.2e-10, 1.0]``.

    Returns
    -------
    result : dict
        Keys: ``a0``, ``beta``, ``nll_oos``, ``aicc_oos``, ``n_train``,
        ``n_test``, ``fit_success``.

    Raises
    ------
    FileNotFoundError
        If *data_file* does not exist.
    ValueError
        If required columns are missing.
    RuntimeError
        If optimisation fails to converge.
    """
    if p0 is None:
        p0 = P0

    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    c_bar  = _find_column(df, "g_bar", required=True)
    c_obs  = _find_column(df, "g_obs", required=True)
    _find_column(df, "m_bar", required=True)   # validated but not used in fit
    c_err  = _find_column(df, "g_err", required=False)

    required_cols = [c_bar, c_obs]
    df = df.dropna(subset=required_cols).copy()

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)

    # --- Fit on training set ---
    train_err = train_df[c_err].values if c_err else None
    res = minimize(
        nll_gauss_accel,
        p0,
        args=(train_df[c_bar].values, train_df[c_obs].values, train_err),
        method="Nelder-Mead",
        options={"maxiter": 50_000, "xatol": 1e-12, "fatol": 1e-12},
    )

    # --- Evaluate on test set (OOS) ---
    test_err   = test_df[c_err].values if c_err else None
    nll_oos    = nll_gauss_accel(
        res.x,
        test_df[c_bar].values,
        test_df[c_obs].values,
        test_err,
    )
    n_test    = len(test_df)
    aicc_oos  = aicc_from_nll(nll_oos, k=K, n=n_test)

    return {
        "a0":          float(res.x[0]),
        "beta":        float(res.x[1]),
        "nll_oos":     float(nll_oos),
        "aicc_oos":    float(aicc_oos),
        "n_train":     len(train_df),
        "n_test":      n_test,
        "fit_success": bool(res.success),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SCM v0.1 — out-of-sample fit in acceleration space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",      default=DATA_FILE, metavar="CSV",
                   help="Input CSV file with g_bar / g_obs columns.")
    p.add_argument("--seed",      type=int,   default=SEED,
                   help="Random seed for train/test split.")
    p.add_argument("--test-size", type=float, default=TEST_SIZE,
                   dest="test_size",
                   help="Fraction of rows held out for OOS evaluation.")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    try:
        res = run_oos_fit(
            data_file=args.data,
            seed=args.seed,
            test_size=args.test_size,
        )

        print(f"\n✅ SCM v0.1 - RESULTADOS OOS")
        print(f"   Parámetros: a0={res['a0']:.4e}, beta={res['beta']:.4f}")
        print(f"   LL_OOS:   {-res['nll_oos']:.4f}")
        print(f"   AICc_OOS: {res['aicc_oos']:.4f}")
        print(f"   n_train={res['n_train']}, n_test={res['n_test']}, "
              f"fit_success={res['fit_success']}")
        print("\n💡 Ahora compara este AICc con el de RAR o NFW para obtener el Delta.")

    except Exception as exc:
        print(f"❌ Error: {exc}. Revisa el nombre del CSV y las columnas.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
