"""
scripts/audit_scm.py — Structural audit for the Motor de Velos SCM framework.

Three audit modules
-------------------
1. **GroupKFold cross-validation** — fits model coefficients separately in
   every training fold, then evaluates on the held-out fold.  This is a true
   generalisation test, not merely evaluating fixed coefficients on different
   subsets.

2. **Model comparison with correct AICc** — three nested models are compared:

   * BTFR      (k=2): log g_obs = β·log g_bar + C
   * SCM       (k=4): + a·log g_bar² + b·r_norm
   * SCM-full  (k=6): + d·max(0, log g_0 − log g_bar)   [hinge term]

   k is the number of free parameters; AICc is only meaningful when k is
   counted correctly.

3. **Permutation test** — log g_bar is permuted *within* each galaxy,
   destroying the radial correlation while preserving galaxy-level structure.
   The RMSE of the full SCM on permuted data is compared with the real RMSE.

Input
-----
The per-radial-point CSV produced by ``src.scm_analysis.run_pipeline()``:
``results/universal_term_comparison_full.csv``
Required columns: ``galaxy``, ``log_g_bar``, ``log_g_obs``, ``r_kpc``.

Usage
-----
::

    python scripts/audit_scm.py \\
        --csv  results/universal_term_comparison_full.csv \\
        --out  results/diagnostics/audit \\
        --n-splits 5 \\
        --n-perm 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical number of free parameters for each model.
MODEL_K: dict[str, int] = {
    "btfr": 2,
    "scm_nohinge": 4,
    "scm_full": 6,
}

#: Reference radius for normalisation (kpc).
_R_REF = 10.0

_SEP = "=" * 64


# ---------------------------------------------------------------------------
# Model predictors
# ---------------------------------------------------------------------------

def scm_btfr_pred(log_g_bar: np.ndarray, r_norm: np.ndarray,
                  params: np.ndarray) -> np.ndarray:
    """BTFR model: log g_obs = β·log g_bar + C.

    Parameters
    ----------
    log_g_bar : ndarray
        log₁₀ of baryonic centripetal acceleration.
    r_norm : ndarray
        Normalised radius (not used; included for uniform signature).
    params : array_like of length 2
        [β, C].

    Returns
    -------
    ndarray
        Predicted log₁₀ g_obs.
    """
    beta, C = params[0], params[1]
    return beta * log_g_bar + C


def scm_nohinge_pred(log_g_bar: np.ndarray, r_norm: np.ndarray,
                     params: np.ndarray) -> np.ndarray:
    """SCM without hinge: log g_obs = β·log g_bar + a·log g_bar² + b·r_norm + C.

    Parameters
    ----------
    log_g_bar, r_norm : ndarray
        Predictor arrays.
    params : array_like of length 4
        [β, C, a, b].

    Returns
    -------
    ndarray
        Predicted log₁₀ g_obs.
    """
    beta, C, a, b = params[0], params[1], params[2], params[3]
    return beta * log_g_bar + a * log_g_bar ** 2 + b * r_norm + C


def scm_full_pred(log_g_bar: np.ndarray, r_norm: np.ndarray,
                  params: np.ndarray) -> np.ndarray:
    """SCM full model (with hinge term).

    log g_obs = β·log g_bar + a·log g_bar² + b·r_norm
                + d·max(0, log g_0 − log g_bar) + C

    Parameters
    ----------
    log_g_bar, r_norm : ndarray
        Predictor arrays.
    params : array_like of length 6
        [β, C, a, b, d, log g_0].

    Returns
    -------
    ndarray
        Predicted log₁₀ g_obs.
    """
    beta, C, a, b, d, logg0 = (params[0], params[1], params[2],
                                params[3], params[4], params[5])
    hinge = np.maximum(0.0, logg0 - log_g_bar)
    return beta * log_g_bar + a * log_g_bar ** 2 + b * r_norm + d * hinge + C


#: Registry: name → (predictor, k, initial params).
_MODELS = {
    "btfr": (scm_btfr_pred, MODEL_K["btfr"],
             np.array([0.5, 0.0])),
    "scm_nohinge": (scm_nohinge_pred, MODEL_K["scm_nohinge"],
                    np.array([0.5, 0.0, 0.0, 0.0])),
    "scm_full": (scm_full_pred, MODEL_K["scm_full"],
                 np.array([0.5, 0.0, 0.0, 0.0, 0.1, -10.0])),
}


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_model(log_g_bar: np.ndarray, r_norm: np.ndarray,
              log_g_obs: np.ndarray, pred_fn, p0: np.ndarray,
              ) -> tuple[np.ndarray, float]:
    """Fit model parameters by minimising MSE.

    Parameters
    ----------
    log_g_bar, r_norm, log_g_obs : ndarray
        Feature and target arrays (same length).
    pred_fn : callable
        One of the ``scm_*_pred`` functions.
    p0 : ndarray
        Initial parameter vector.

    Returns
    -------
    params : ndarray
        Best-fit parameter vector.
    rmse : float
        Root-mean-square error at the best-fit solution.
    """
    def objective(params: np.ndarray) -> float:
        y_pred = pred_fn(log_g_bar, r_norm, params)
        return float(np.mean((log_g_obs - y_pred) ** 2))

    result = minimize(
        objective, p0, method="Nelder-Mead",
        options={"maxiter": 20_000, "xatol": 1e-7, "fatol": 1e-10},
    )
    best_params = result.x
    rmse = float(np.sqrt(np.mean((log_g_obs - pred_fn(log_g_bar, r_norm, best_params)) ** 2)))
    return best_params, rmse


# ---------------------------------------------------------------------------
# Information criterion
# ---------------------------------------------------------------------------

def safe_aicc(y: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """AICc under a Gaussian error model.

    AICc = −2·LL + 2k + 2k(k+1)/(n−k−1)

    where  LL = −n/2 · (ln(2π σ̂²) + 1)  and  σ̂² = mean squared residual.

    Parameters
    ----------
    y : ndarray   Target values.
    y_pred : ndarray
        Predicted values.
    k : int
        Number of free parameters (must be ≥ 1).

    Returns
    -------
    float
        AICc value (lower is better).
    """
    n = len(y)
    sigma2 = max(float(np.mean((y - y_pred) ** 2)), 1e-30)
    ll = -0.5 * n * (np.log(2.0 * np.pi * sigma2) + 1.0)
    aic = -2.0 * ll + 2.0 * k
    correction = 2.0 * k * (k + 1) / max(n - k - 1, 1)
    return float(aic + correction)


# ---------------------------------------------------------------------------
# GroupKFold splitting (no sklearn dependency)
# ---------------------------------------------------------------------------

def _group_kfold_splits(groups: np.ndarray, n_splits: int = 5,
                        seed: int = 42):
    """Yield ``(train_idx, test_idx)`` pairs grouped by *groups*.

    All rows belonging to the same group are assigned to the same fold,
    so no galaxy leaks between train and test sets.

    Parameters
    ----------
    groups : array_like
        Group label for each row (e.g. galaxy name).
    n_splits : int
        Number of folds.
    seed : int
        Random seed for group shuffling.

    Yields
    ------
    train_idx, test_idx : ndarray of int
        Row indices for training and test sets in this fold.
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)

    # Assign each group to a fold round-robin
    fold_of: dict = {g: int(i % n_splits) for i, g in enumerate(shuffled)}

    for fold_idx in range(n_splits):
        test_groups = {g for g, f in fold_of.items() if f == fold_idx}
        test_mask = np.array([g in test_groups for g in groups], dtype=bool)
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(train_idx) > 0 and len(test_idx) > 0:
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# Audit module 1: GroupKFold with per-fold fitting
# ---------------------------------------------------------------------------

def groupkfold_audit(df: pd.DataFrame, n_splits: int = 5,
                     seed: int = 42) -> dict:
    """GroupKFold cross-validation with genuine per-fold coefficient fitting.

    For every fold:
      1. Fit β, C, a, b, d, log g_0 using *train_idx* rows.
      2. Predict on *test_idx* rows.
      3. Record RMSE_train, RMSE_test and the fitted coefficient vector.

    Parameters
    ----------
    df : pd.DataFrame
        Per-radial-point data with columns
        ``galaxy``, ``log_g_bar``, ``log_g_obs``, ``r_kpc``.
    n_splits : int
        Number of GroupKFold folds.
    seed : int
        Random seed for group shuffling.

    Returns
    -------
    dict with keys:
        ``rmse_train``  — per-fold training RMSE (list)
        ``rmse_test``   — per-fold test RMSE (list)
        ``params``      — per-fold parameter array (list of ndarray [β,C,a,b,d,logg0])
        ``n_folds``     — actual number of folds produced
        ``param_names`` — list of parameter name strings
    """
    log_g_bar = df["log_g_bar"].to_numpy()
    log_g_obs = df["log_g_obs"].to_numpy()
    r_norm = df["r_kpc"].to_numpy() / _R_REF
    groups = df["galaxy"].to_numpy()

    pred_fn, _, p0 = _MODELS["scm_full"]

    rmse_train_list: list[float] = []
    rmse_test_list: list[float] = []
    params_list: list[np.ndarray] = []

    for train_idx, test_idx in _group_kfold_splits(groups, n_splits=n_splits, seed=seed):
        # --- fit on training data ---
        params, rmse_train = fit_model(
            log_g_bar[train_idx], r_norm[train_idx], log_g_obs[train_idx],
            pred_fn, p0.copy(),
        )

        # --- evaluate on test data ---
        y_pred_test = pred_fn(log_g_bar[test_idx], r_norm[test_idx], params)
        rmse_test = float(np.sqrt(np.mean((log_g_obs[test_idx] - y_pred_test) ** 2)))

        rmse_train_list.append(rmse_train)
        rmse_test_list.append(rmse_test)
        params_list.append(params.copy())

    return {
        "rmse_train": rmse_train_list,
        "rmse_test": rmse_test_list,
        "params": params_list,
        "n_folds": len(rmse_train_list),
        "param_names": ["beta", "C", "a", "b", "d", "logg0"],
    }


# ---------------------------------------------------------------------------
# Audit module 2: Model comparison with correct AICc
# ---------------------------------------------------------------------------

def model_comparison(df: pd.DataFrame) -> dict:
    """Fit all three models on the full dataset and compare via AICc.

    Parameter counts used:
      * BTFR      → k = 2
      * SCM       → k = 4
      * SCM-full  → k = 6

    Each model is warm-started from the fitted solution of the previous
    (simpler) model so that nested models always have RMSE ≤ the simpler
    model and Nelder-Mead does not start from an unphysical point.

    Parameters
    ----------
    df : pd.DataFrame
        Per-radial-point data (same columns as :func:`groupkfold_audit`).

    Returns
    -------
    dict mapping model name to sub-dict with keys:
        ``params``, ``rmse``, ``aicc``, ``k``.
    """
    log_g_bar = df["log_g_bar"].to_numpy()
    log_g_obs = df["log_g_obs"].to_numpy()
    r_norm = df["r_kpc"].to_numpy() / _R_REF

    results: dict = {}

    # --- BTFR (k=2) ---
    pred_fn_btfr, k_btfr, p0_btfr = _MODELS["btfr"]
    p_btfr, rmse_btfr = fit_model(log_g_bar, r_norm, log_g_obs, pred_fn_btfr, p0_btfr.copy())
    results["btfr"] = {
        "params": p_btfr,
        "rmse": rmse_btfr,
        "aicc": safe_aicc(log_g_obs, pred_fn_btfr(log_g_bar, r_norm, p_btfr), k_btfr),
        "k": k_btfr,
    }

    # --- SCM no-hinge (k=4) — warm-start from BTFR solution ---
    pred_fn_nh, k_nh, _ = _MODELS["scm_nohinge"]
    p0_nh = np.array([p_btfr[0], p_btfr[1], 0.0, 0.0])
    p_nh, rmse_nh = fit_model(log_g_bar, r_norm, log_g_obs, pred_fn_nh, p0_nh)
    results["scm_nohinge"] = {
        "params": p_nh,
        "rmse": rmse_nh,
        "aicc": safe_aicc(log_g_obs, pred_fn_nh(log_g_bar, r_norm, p_nh), k_nh),
        "k": k_nh,
    }

    # --- SCM full (k=6) — warm-start from no-hinge solution ---
    pred_fn_full, k_full, _ = _MODELS["scm_full"]
    p0_full = np.array([p_nh[0], p_nh[1], p_nh[2], p_nh[3], 0.0, -10.0])
    p_full, rmse_full = fit_model(log_g_bar, r_norm, log_g_obs, pred_fn_full, p0_full)
    results["scm_full"] = {
        "params": p_full,
        "rmse": rmse_full,
        "aicc": safe_aicc(log_g_obs, pred_fn_full(log_g_bar, r_norm, p_full), k_full),
        "k": k_full,
    }

    # Compute ΔAICc relative to best model
    best_aicc = min(v["aicc"] for v in results.values())
    for v in results.values():
        v["delta_aicc"] = v["aicc"] - best_aicc

    return results


# ---------------------------------------------------------------------------
# Audit module 3: Permutation test
# ---------------------------------------------------------------------------

def permutation_test(df: pd.DataFrame, n_perm: int = 200,
                     seed: int = 42) -> dict:
    """Permutation test: permute log g_bar *within* each galaxy.

    Permuting within galaxy destroys the radial correlation between
    log g_bar and log g_obs without breaking the galaxy-level sample
    structure, providing a clean null distribution for the model RMSE.

    Parameters
    ----------
    df : pd.DataFrame
        Per-radial-point data.
    n_perm : int
        Number of permutations.
    seed : int
        Base random seed.

    Returns
    -------
    dict with keys:
        ``rmse_real``   — RMSE of the full SCM fitted to real data.
        ``rmse_perm``   — array of RMSE values from permuted datasets.
        ``p_value``     — fraction of permuted RMSE ≤ real RMSE
                          (small → model captures genuine signal).
    """
    log_g_bar = df["log_g_bar"].to_numpy()
    log_g_obs = df["log_g_obs"].to_numpy()
    r_norm = df["r_kpc"].to_numpy() / _R_REF
    galaxies = df["galaxy"].to_numpy()

    pred_fn, _, p0 = _MODELS["scm_full"]

    # Fit on real data
    _, rmse_real = fit_model(log_g_bar, r_norm, log_g_obs, pred_fn, p0.copy())

    # Permuted fits
    unique_galaxies = np.unique(galaxies)
    rng = np.random.default_rng(seed)
    rmse_perm: list[float] = []

    for _ in range(n_perm):
        perm_log_g_bar = log_g_bar.copy()
        for gal in unique_galaxies:
            mask = galaxies == gal
            idx = np.where(mask)[0]
            perm_log_g_bar[idx] = rng.permutation(perm_log_g_bar[idx])

        _, rmse_p = fit_model(perm_log_g_bar, r_norm, log_g_obs, pred_fn, p0.copy())
        rmse_perm.append(rmse_p)

    rmse_perm_arr = np.array(rmse_perm)
    # p-value: fraction of permuted RMSE ≤ real RMSE (small p → real signal)
    p_value = float(np.mean(rmse_perm_arr <= rmse_real))

    return {
        "rmse_real": rmse_real,
        "rmse_perm": rmse_perm_arr,
        "p_value": p_value,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _format_kfold_report(kf: dict) -> list[str]:
    """Format the GroupKFold audit report."""
    lines = [
        _SEP,
        "  GroupKFold Audit — SCM Full Model",
        _SEP,
        f"  Folds: {kf['n_folds']}",
    ]
    if kf["n_folds"] == 0:
        lines.append("  ⚠️  No folds produced (insufficient groups?).")
        return lines

    rmse_train = np.array(kf["rmse_train"])
    rmse_test = np.array(kf["rmse_test"])
    lines += [
        "",
        f"  RMSE Train : {rmse_train.mean():.4f} ± {rmse_train.std():.4f}",
        f"  RMSE Test  : {rmse_test.mean():.4f} ± {rmse_test.std():.4f}",
        "",
        "  Parameter stability across folds:",
    ]
    params_arr = np.array(kf["params"])  # shape (n_folds, n_params)
    for i, name in enumerate(kf["param_names"]):
        col = params_arr[:, i]
        lines.append(f"    {name:<8}: {col.mean():.4f} ± {col.std():.4f}")

    gap = rmse_test.mean() - rmse_train.mean()
    verdict = ("✅  Train/test gap small → model generalises well."
               if abs(gap) < 0.5 * rmse_train.std() + 0.01
               else "⚠️  Notable train/test gap → possible overfitting.")
    lines += ["", f"  Gap (test−train): {gap:+.4f}", f"  Verdict: {verdict}", _SEP]
    return lines


def _format_comparison_report(comp: dict) -> list[str]:
    """Format the model comparison report."""
    lines = [
        _SEP,
        "  Model Comparison (AICc)",
        _SEP,
        f"  {'Model':<14} {'k':>3} {'RMSE':>8} {'AICc':>12} {'ΔAICc':>8}",
        "  " + "-" * 50,
    ]
    for name in sorted(comp, key=lambda n: comp[n]["delta_aicc"]):
        v = comp[name]
        lines.append(
            f"  {name:<14} {v['k']:>3} {v['rmse']:>8.4f} {v['aicc']:>12.2f} "
            f"{v['delta_aicc']:>8.2f}"
        )
    winner = min(comp, key=lambda n: comp[n]["delta_aicc"])
    lines += ["  " + "-" * 50, f"  Winner: {winner}", _SEP]
    return lines


def _format_perm_report(perm: dict) -> list[str]:
    """Format the permutation test report."""
    rmse_perm = perm["rmse_perm"]
    lines = [
        _SEP,
        "  Permutation Test",
        _SEP,
        f"  Permutations   : {len(rmse_perm)}",
        f"  RMSE (real)    : {perm['rmse_real']:.4f}",
        f"  RMSE (perm)    : {rmse_perm.mean():.4f} ± {rmse_perm.std():.4f}",
        f"  p-value        : {perm['p_value']:.4f}",
    ]
    if perm["p_value"] < 0.05:
        verdict = "✅  p < 0.05 → model captures genuine radial signal."
    else:
        verdict = "⚠️  p ≥ 0.05 → cannot reject null hypothesis of no signal."
    lines += [f"  Verdict: {verdict}", _SEP]
    return lines


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

def run_audit(csv_path: str | Path, out_dir: str | Path,
              n_splits: int = 5, n_perm: int = 200,
              seed: int = 42) -> dict:
    """Run all three audit modules and write results to *out_dir*.

    Parameters
    ----------
    csv_path : str or Path
        Path to ``universal_term_comparison_full.csv``.
    out_dir : str or Path
        Output directory for CSV results and the audit log.
    n_splits : int
        Number of GroupKFold folds.
    n_perm : int
        Number of permutations for the permutation test.
    seed : int
        Master random seed.

    Returns
    -------
    dict with keys ``kfold``, ``comparison``, ``permutation``.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required = {"galaxy", "log_g_bar", "log_g_obs", "r_kpc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}.\n"
            "Regenerate with run_pipeline() first."
        )

    # --- Audit 1: GroupKFold ---
    kf = groupkfold_audit(df, n_splits=n_splits, seed=seed)

    # --- Audit 2: Model comparison ---
    comp = model_comparison(df)

    # --- Audit 3: Permutation test ---
    perm = permutation_test(df, n_perm=n_perm, seed=seed)

    # --- Format reports ---
    lines: list[str] = (
        _format_kfold_report(kf)
        + [""]
        + _format_comparison_report(comp)
        + [""]
        + _format_perm_report(perm)
    )
    for line in lines:
        print(line)

    # --- Write artefacts ---
    # Fold-level CSV
    if kf["n_folds"] > 0:
        fold_records = []
        for i in range(kf["n_folds"]):
            row: dict = {
                "fold": i,
                "rmse_train": kf["rmse_train"][i],
                "rmse_test": kf["rmse_test"][i],
            }
            for j, pname in enumerate(kf["param_names"]):
                row[pname] = kf["params"][i][j]
            fold_records.append(row)
        pd.DataFrame(fold_records).to_csv(out_dir / "kfold_results.csv", index=False)

    # Model comparison CSV
    comp_records = []
    for name, v in comp.items():
        rec: dict = {"model": name, "k": v["k"], "rmse": v["rmse"],
                     "aicc": v["aicc"], "delta_aicc": v["delta_aicc"]}
        for j, pname in enumerate(["beta", "C", "a", "b", "d", "logg0"][:len(v["params"])]):
            rec[pname] = v["params"][j]
        comp_records.append(rec)
    pd.DataFrame(comp_records).to_csv(out_dir / "model_comparison.csv", index=False)

    # Permutation CSV
    pd.DataFrame({
        "rmse_perm": perm["rmse_perm"],
    }).assign(rmse_real=perm["rmse_real"], p_value=perm["p_value"]).to_csv(
        out_dir / "permutation_test.csv", index=False
    )

    # Audit log
    (out_dir / "audit.log").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n  Results written to {out_dir}")

    return {"kfold": kf, "comparison": comp, "permutation": perm}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_CSV_DEFAULT = "results/universal_term_comparison_full.csv"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SCM structural audit: GroupKFold + AICc + permutation test."
    )
    parser.add_argument(
        "--csv", default=_CSV_DEFAULT,
        help=f"Per-radial-point CSV (default: {_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--out", default="results/diagnostics/audit", metavar="DIR",
        help="Output directory (default: results/diagnostics/audit).",
    )
    parser.add_argument(
        "--n-splits", type=int, default=5, dest="n_splits",
        help="Number of GroupKFold folds (default: 5).",
    )
    parser.add_argument(
        "--n-perm", type=int, default=200, dest="n_perm",
        help="Number of permutations for the permutation test (default: 200).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master random seed (default: 42).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    """Run the full SCM audit and return the result dict."""
    args = _parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run 'python -m src.scm_analysis --data-dir data/SPARC --out results/' first."
        )
    return run_audit(
        csv_path=csv_path,
        out_dir=args.out,
        n_splits=args.n_splits,
        n_perm=args.n_perm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
