"""model_comparison_oos.py — SCM vs RAR vs NFW out-of-sample comparison.

Fits three competing acceleration-space models on a shared 80/20 train/test
split and evaluates each model's Gaussian log-likelihood and AICc on the
held-out test set.  All three models use the **identical** :func:`gaussian_ll`
function so the log-likelihoods are directly comparable.

Models
------
+-------+--------------------------------------------+---------+
| Name  | Predicted g_obs                            | k (free |
|       |                                            | params) |
+-------+--------------------------------------------+---------+
| SCM   | g_bar + a0·(√(1+(g_bar/a0)^β)−1)         | 2       |
| RAR   | g_bar · ν(g_bar/g†)                        | 1       |
| NFW   | g_bar + G·M_NFW(r_eff;ρ_s,r_s)/r_eff²     | 2       |
+-------+--------------------------------------------+---------+

Normalisation rule
------------------
Every ``ll_oos()`` call reduces to::

    gaussian_ll(y, yhat, sigma) =
        -0.5 * sum((y - yhat)**2 / sigma**2 + log(2*pi*sigma**2))

with the same *sigma* vector (column ``g_err``) for all models.

Usage
-----
    python scripts/model_comparison_oos.py
    python scripts/model_comparison_oos.py --data data/sparc_rar_sample.csv
    python scripts/model_comparison_oos.py --out results/model_comparison_oos.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scm_models import (
    aicc_from_nll,
    gaussian_ll,
    nll_gauss_accel,
    nll_nfw_accel,
    nll_rar_accel,
)

# ---------------------------------------------------------------------------
# Frozen configuration (matches scm_oos_fit.py to guarantee same test split)
# ---------------------------------------------------------------------------
SEED:      int   = 20260211
TEST_SIZE: float = 0.2
DATA_FILE: str   = "data/sparc_rar_sample.csv"
OUT_FILE:  str   = "results/model_comparison_oos.csv"

# Free-parameter counts (MNRAS checklist)
K_SCM: int = 2   # (a0, beta)
K_RAR: int = 1   # (g_dagger)
K_NFW: int = 2   # (log10_rho_s, log10_r_s)

# ---------------------------------------------------------------------------
# Column aliases
# ---------------------------------------------------------------------------
_COL_CANDIDATES = {
    "g_bar": ["g_bar", "gbar", "g_bary"],
    "g_obs": ["g_obs", "gobs", "g_total"],
    "m_bar": ["m_bar", "M_bar", "Mb"],
    "g_err": ["g_err", "dg", "e_gobs"],
}


def _find_col(df: pd.DataFrame, role: str, required: bool = True) -> str | None:
    for cand in _COL_CANDIDATES[role]:
        if cand in df.columns:
            return cand
    if required:
        raise ValueError(
            f"Could not find column for '{role}'. "
            f"Expected: {_COL_CANDIDATES[role]}. "
            f"Got: {list(df.columns)}"
        )
    return None


# ---------------------------------------------------------------------------
# Core comparison function (importable for tests)
# ---------------------------------------------------------------------------

def run_comparison(
    data_file: str = DATA_FILE,
    seed: int = SEED,
    test_size: float = TEST_SIZE,
) -> pd.DataFrame:
    """Fit SCM, RAR and NFW; return a DataFrame with OOS metrics.

    All three models share:

    * The same train/test split (determined by *seed* and *test_size*).
    * The same observed target variable (column ``g_obs``).
    * The same uncertainty vector σ (column ``g_err``; σ = 1 when absent).
    * The same :func:`~src.scm_models.gaussian_ll` formula.

    Parameters
    ----------
    data_file : str
        Path to the input CSV.
    seed : int
        Random seed for the train/test split.
    test_size : float
        Fraction held out for OOS evaluation.

    Returns
    -------
    df_res : pd.DataFrame
        Columns: ``name``, ``k``, ``ll_oos``, ``aicc_oos``,
        ``delta_aicc_vs_scm``, ``n_train``, ``n_test``, ``fit_success``,
        ``params``.
    """
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    c_bar = _find_col(df, "g_bar")
    c_obs = _find_col(df, "g_obs")
    c_mbar = _find_col(df, "m_bar")
    c_err = _find_col(df, "g_err", required=False)

    df = df.dropna(subset=[c_bar, c_obs]).copy()
    train_df, test_df = train_test_split(df, test_size=test_size,
                                         random_state=seed)

    # Arrays
    g_bar_tr  = train_df[c_bar].values
    g_obs_tr  = train_df[c_obs].values
    m_bar_tr  = train_df[c_mbar].values
    g_err_tr  = train_df[c_err].values if c_err else None

    g_bar_te  = test_df[c_bar].values
    g_obs_te  = test_df[c_obs].values
    m_bar_te  = test_df[c_mbar].values
    g_err_te  = test_df[c_err].values if c_err else None

    # Build sigma vectors for LL evaluation (same for all models)
    if g_err_te is not None:
        sigma_te = np.where(
            np.isfinite(g_err_te) & (g_err_te > 0), g_err_te, 1.0
        )
    else:
        sigma_te = np.ones(len(test_df))

    n_te = len(test_df)

    opt_kw = dict(method="Nelder-Mead",
                  options={"maxiter": 50_000, "xatol": 1e-12, "fatol": 1e-12})

    rows = []

    # ------------------------------------------------------------------
    # 1. SCM (k = 2)  —  (a0, beta)
    # ------------------------------------------------------------------
    res_scm = minimize(
        nll_gauss_accel, [1.2e-10, 1.0],
        args=(g_bar_tr, g_obs_tr, g_err_tr),
        **opt_kw,
    )
    from src.scm_models import scm_model_accel
    g_pred_scm = g_bar_te + scm_model_accel(g_bar_te, *res_scm.x)
    ll_scm     = gaussian_ll(g_obs_te, g_pred_scm, sigma_te)
    aicc_scm   = aicc_from_nll(-ll_scm, K_SCM, n_te)
    rows.append({
        "name":        "SCM",
        "k":           K_SCM,
        "ll_oos":      ll_scm,
        "aicc_oos":    aicc_scm,
        "n_train":     len(train_df),
        "n_test":      n_te,
        "fit_success": bool(res_scm.success),
        "params":      str(dict(zip(["a0", "beta"], res_scm.x.tolist()))),
    })

    # ------------------------------------------------------------------
    # 2. RAR (k = 1)  —  (g_dagger,)
    # ------------------------------------------------------------------
    res_rar = minimize(
        nll_rar_accel, [1.2e-10],
        args=(g_bar_tr, g_obs_tr, g_err_tr),
        **opt_kw,
    )
    from src.scm_models import rar_model_accel
    g_pred_rar = rar_model_accel(g_bar_te, res_rar.x[0])
    ll_rar     = gaussian_ll(g_obs_te, g_pred_rar, sigma_te)
    aicc_rar   = aicc_from_nll(-ll_rar, K_RAR, n_te)
    rows.append({
        "name":        "RAR",
        "k":           K_RAR,
        "ll_oos":      ll_rar,
        "aicc_oos":    aicc_rar,
        "n_train":     len(train_df),
        "n_test":      n_te,
        "fit_success": bool(res_rar.success),
        "params":      str(dict(zip(["g_dagger"], res_rar.x.tolist()))),
    })

    # ------------------------------------------------------------------
    # 3. NFW (k = 2)  —  (log10_rho_s, log10_r_s)
    # rho_s ≈ 6.8e-22 kg m^-3 → log10 ≈ -21.2
    # r_s   ≈ 10 kpc  = 3.086e20 m → log10 ≈ 20.5
    # ------------------------------------------------------------------
    res_nfw = minimize(
        nll_nfw_accel, [-21.2, 20.5],
        args=(g_bar_tr, g_obs_tr, m_bar_tr, g_err_tr),
        **opt_kw,
    )
    from src.scm_models import nfw_model_accel
    g_pred_nfw = nfw_model_accel(g_bar_te, m_bar_te, *res_nfw.x)
    ll_nfw     = gaussian_ll(g_obs_te, g_pred_nfw, sigma_te)
    aicc_nfw   = aicc_from_nll(-ll_nfw, K_NFW, n_te)
    rows.append({
        "name":        "NFW",
        "k":           K_NFW,
        "ll_oos":      ll_nfw,
        "aicc_oos":    aicc_nfw,
        "n_train":     len(train_df),
        "n_test":      n_te,
        "fit_success": bool(res_nfw.success),
        "params":      str(dict(zip(["log10_rho_s", "log10_r_s"],
                                    res_nfw.x.tolist()))),
    })

    df_res = pd.DataFrame(rows)
    df_res["delta_aicc_vs_scm"] = df_res["aicc_oos"] - aicc_scm
    return df_res


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OOS model comparison: SCM vs RAR vs NFW",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",  default=DATA_FILE, metavar="CSV",
                   help="Input CSV (g_bar, g_obs, m_bar columns).")
    p.add_argument("--seed",  type=int,   default=SEED)
    p.add_argument("--out",   default=OUT_FILE, metavar="CSV",
                   help="Output CSV for comparison table.")
    p.add_argument("--test-size", type=float, default=TEST_SIZE,
                   dest="test_size")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        df_res = run_comparison(
            data_file=args.data,
            seed=args.seed,
            test_size=args.test_size,
        )

        # --- SCM results (reference) ---
        scm = df_res[df_res["name"] == "SCM"].iloc[0]
        print(f"\n✅ SCM - RESULTADOS OOS")
        print(f"   LL_OOS:   {scm['ll_oos']:.4f}")
        print(f"   AICc_OOS: {scm['aicc_oos']:.4f}")
        print(f"   DeltaAICc_vs_SCM: 0.0000")

        # --- Notario output (exact format from problem statement) ---
        for name in ["RAR", "NFW"]:
            row = df_res[df_res["name"] == name].iloc[0]
            print(f"\n✅ {name} - RESULTADOS OOS")
            print(f"   LL_OOS:   {row['ll_oos']:.4f}")
            print(f"   AICc_OOS: {row['aicc_oos']:.4f}")
            print(f"   DeltaAICc_vs_SCM: {row['delta_aicc_vs_scm']:.4f}")

        # --- Ranking table ---
        print("\n" + "─" * 65)
        print(f"{'Rank':<5} {'Model':<8} {'k':<4} {'LL_OOS':>12} "
              f"{'AICc_OOS':>12} {'ΔAICc_vs_SCM':>14}")
        print("─" * 65)
        ranked = df_res.sort_values("aicc_oos").reset_index(drop=True)
        for i, row in ranked.iterrows():
            verdict = ("✓ mejor" if row["delta_aicc_vs_scm"] < -2
                       else ("≈ empate" if abs(row["delta_aicc_vs_scm"]) <= 2
                             else "✗ peor"))
            print(f"{i+1:<5} {row['name']:<8} {row['k']:<4} "
                  f"{row['ll_oos']:>12.4f} {row['aicc_oos']:>12.4f} "
                  f"{row['delta_aicc_vs_scm']:>14.4f}  {verdict}")
        print("─" * 65)

        # --- Save ---
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(args.out, index=False)
        print(f"\n📋 Tabla guardada en '{args.out}'")

    except Exception as exc:
        print(f"❌ Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
