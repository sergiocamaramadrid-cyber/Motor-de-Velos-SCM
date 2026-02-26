"""
scripts/audit_scm.py — Reproducible audit pipeline for the Motor de Velos SCM.

What it does (in a single execution)
--------------------------------------
0) Read a flat CSV with columns (galaxy_id, r, v_obs, e_v_obs, …features…),
   build galaxy_key, and optionally validate via sparc_validate.py.

1) GroupKFold cross-validation (n_splits=5 by default, configurable).
   Groups = galaxy_id so no galaxy appears in both train and test.
   Trains a ridge-regularised linear model in log-space and predicts
   log(v_obs) for unseen galaxies.
   Outputs:
     results/audit/groupkfold_metrics-vX.csv
     results/audit/groupkfold_per_galaxy-vX.csv
     results/audit/coeffs_by_fold-vX.csv

2) Permutation test on log_gbar (anti-self-deception).
   Shuffles log_gbar within each galaxy and re-runs the full GroupKFold.
   Compares ΔRMSE distribution (real vs permuted).
   Outputs:
     results/audit/permutation_summary-vX.json
     results/audit/permutation_runs-vX.csv

3) Model duel (AICc).
   Evaluates three nested models:
     btfr      — BTFR (mass alone: log_v ~ log_gbar only)
     scm_base  — SCM without hinge: log_v ~ log_gbar + all features
     scm_full  — SCM complete:      log_v ~ log_gbar + hinge(log_gbar) + all features
   Uses Gaussian log-likelihood + AICc for model selection.
   Outputs:
     results/audit/model_comparison_aicc-vX.csv

4) "Freeze" master coefficients (fit on full dataset).
   Outputs:
     results/audit/master_coeffs-vX.json

CLI
---
python scripts/audit_scm.py \\
  --input data/SPARC/sparc_raw.csv \\
  --outdir results/audit \\
  --ref $(git rev-parse --short HEAD) \\
  --seed 123 \\
  --kfold 5 \\
  --permutations 200 \\
  --strict

Column conventions (flat CSV)
------------------------------
  galaxy_id   — galaxy identifier (string)
  r           — galactocentric radius (kpc)
  v_obs       — observed rotation velocity (km/s)
  e_v_obs     — uncertainty on v_obs (km/s)
  v_bar       — total baryonic velocity (km/s)
  g_bar       — baryonic centripetal acceleration (m/s²)   [optional; computed if absent]
  log_gbar    — log10(g_bar)                               [optional; computed if absent]
  log_vobs    — log10(v_obs)                               [optional; computed if absent]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Physics constants (shared with src/)
# ---------------------------------------------------------------------------
KPC_TO_M = 3.085677581e16  # metres per kpc (IAU 2012)
_CONV = 1e6 / KPC_TO_M     # (km/s)²/kpc → m/s²


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log10_safe(x: np.ndarray, floor: float = 1e-30) -> np.ndarray:
    return np.log10(np.maximum(x, floor))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _gaussian_ll(y_true: np.ndarray, y_pred: np.ndarray,
                 sigma: float | None = None) -> float:
    """Gaussian log-likelihood; sigma estimated from residuals if None."""
    resid = y_true - y_pred
    if sigma is None:
        sigma = float(np.std(resid, ddof=1)) or 1.0
    n = len(y_true)
    ll = -0.5 * n * np.log(2 * np.pi * sigma ** 2) - \
         0.5 * np.sum(resid ** 2) / sigma ** 2
    return float(ll)


def _aicc(ll: float, k: int, n: int) -> float:
    """AICc = -2·LL + 2k + 2k(k+1)/(n-k-1)."""
    aic = -2.0 * ll + 2.0 * k
    correction = 2.0 * k * (k + 1) / max(n - k - 1, 1)
    return aic + correction


def _hinge(x: np.ndarray, pivot: float = -10.5) -> np.ndarray:
    """Hinge / ReLU feature: max(x - pivot, 0)."""
    return np.maximum(x - pivot, 0.0)


# ---------------------------------------------------------------------------
# 0) Data loading and validation
# ---------------------------------------------------------------------------

def load_and_prepare(csv_path: Path, strict: bool = False) -> pd.DataFrame:
    """Load flat CSV and ensure required derived columns are present.

    Derived columns computed on-the-fly if absent:
      log_gbar  = log10(g_bar)  — or computed from v_bar and r
      log_vobs  = log10(v_obs)

    Parameters
    ----------
    csv_path : Path
        Flat CSV file.
    strict : bool
        If True, raise ValueError when required source columns are missing.

    Returns
    -------
    pd.DataFrame
        Augmented dataframe, rows with non-positive v_obs or g_bar dropped.
    """
    df = pd.read_csv(csv_path)

    required = {"galaxy_id", "r", "v_obs", "e_v_obs"}
    missing = required - set(df.columns)
    if missing:
        msg = f"Input CSV missing required columns: {missing}"
        if strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}", file=sys.stderr)

    # Build g_bar if absent
    if "g_bar" not in df.columns:
        if "v_bar" in df.columns:
            df["g_bar"] = df["v_bar"] ** 2 / df["r"].clip(lower=1e-10) * _CONV
        elif strict:
            raise ValueError(
                "Neither 'g_bar' nor 'v_bar' found; cannot compute g_bar."
            )

    # Build log_gbar and log_vobs
    if "g_bar" in df.columns:
        df["log_gbar"] = _log10_safe(df["g_bar"].values)
    if "v_obs" in df.columns:
        df["log_vobs"] = _log10_safe(df["v_obs"].values)

    # Drop unphysical rows
    if "v_obs" in df.columns:
        df = df[df["v_obs"] > 0].copy()
    if "g_bar" in df.columns:
        df = df[df["g_bar"] > 0].copy()

    return df.reset_index(drop=True)


def _select_features(df: pd.DataFrame, include_hinge: bool = True,
                     base_only: bool = False) -> tuple[np.ndarray, list[str]]:
    """Return feature matrix X and feature names.

    Model hierarchy:
      btfr      base_only=True,  include_hinge=False → X = [log_gbar]
      scm_base  base_only=False, include_hinge=False → X = [log_gbar, …extra]
      scm_full  base_only=False, include_hinge=True  → X = [log_gbar, hinge, …extra]
    """
    candidate_extra = [c for c in df.columns
                       if c not in {"galaxy_id", "r", "v_obs", "e_v_obs",
                                    "v_bar", "g_bar", "log_gbar", "log_vobs"}
                       and pd.api.types.is_numeric_dtype(df[c])]

    if base_only:
        feature_cols = ["log_gbar"]
    else:
        feature_cols = ["log_gbar"] + candidate_extra

    X = df[feature_cols].values.astype(float)

    if include_hinge and not base_only and "log_gbar" in feature_cols:
        h = _hinge(df["log_gbar"].values).reshape(-1, 1)
        X = np.hstack([X, h])
        feature_cols = feature_cols + ["hinge_log_gbar"]

    return X, feature_cols


# ---------------------------------------------------------------------------
# 1) GroupKFold cross-validation
# ---------------------------------------------------------------------------

def run_groupkfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """GroupKFold cross-validation predicting log(v_obs).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared flat dataframe.
    n_splits : int
        Number of folds.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    metrics_df : DataFrame
        Per-fold RMSE/bias.
    per_galaxy_df : DataFrame
        Per-galaxy RMSE (OOS predictions).
    coeffs_df : DataFrame
        Coefficients per fold.
    """
    y = df["log_vobs"].values
    X, feat_names = _select_features(df, include_hinge=True, base_only=False)
    groups = df["galaxy_id"].values

    gkf = GroupKFold(n_splits=n_splits)
    scaler = StandardScaler()
    model = Ridge(alpha=1.0, random_state=seed)

    fold_records = []
    oos_preds = np.full(len(df), np.nan)
    coeff_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        oos_preds[test_idx] = y_pred

        rmse = _rmse(y_te, y_pred)
        bias = float(np.mean(y_pred - y_te))
        n_train_gal = len(np.unique(groups[train_idx]))
        n_test_gal = len(np.unique(groups[test_idx]))

        fold_records.append({
            "fold": fold_idx,
            "rmse_log_vobs": rmse,
            "bias_log_vobs": bias,
            "n_train_points": len(train_idx),
            "n_test_points": len(test_idx),
            "n_train_galaxies": n_train_gal,
            "n_test_galaxies": n_test_gal,
        })

        coeff_row = {"fold": fold_idx, "intercept": float(model.intercept_)}
        for fname, coef in zip(feat_names, model.coef_):
            coeff_row[fname] = float(coef)
        coeff_records.append(coeff_row)

    metrics_df = pd.DataFrame(fold_records)

    # Per-galaxy OOS RMSE
    df2 = df.copy()
    df2["_oos_pred_log_vobs"] = oos_preds
    per_gal_rows = []
    for gid, grp in df2.groupby("galaxy_id"):
        valid = grp["_oos_pred_log_vobs"].notna()
        if valid.sum() == 0:
            continue
        g_rmse = _rmse(grp.loc[valid, "log_vobs"].values,
                       grp.loc[valid, "_oos_pred_log_vobs"].values)
        per_gal_rows.append({"galaxy_id": gid,
                              "n_points": int(valid.sum()),
                              "rmse_log_vobs": g_rmse})
    per_galaxy_df = pd.DataFrame(per_gal_rows)
    coeffs_df = pd.DataFrame(coeff_records)

    return metrics_df, per_galaxy_df, coeffs_df


# ---------------------------------------------------------------------------
# 2) Permutation test
# ---------------------------------------------------------------------------

def run_permutation_test(
    df: pd.DataFrame,
    n_splits: int = 5,
    n_permutations: int = 200,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Permutation test: shuffle log_gbar within each galaxy and re-run CV.

    The test checks whether the real improvement over a random model is
    statistically meaningful.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared flat dataframe.
    n_splits : int
        GroupKFold splits (same as main CV).
    n_permutations : int
        Number of permutation repetitions.
    seed : int
        Base random seed.

    Returns
    -------
    summary : dict
        Aggregate statistics.
    runs_df : DataFrame
        Per-permutation mean RMSE.
    """
    # Real RMSE (already computed, but recompute here for internal consistency)
    metrics_real, _, _ = run_groupkfold(df, n_splits=n_splits, seed=seed)
    real_rmse = float(metrics_real["rmse_log_vobs"].mean())

    rng = np.random.default_rng(seed)
    perm_rmses = []

    for i in range(n_permutations):
        df_perm = df.copy()
        # Shuffle log_gbar within each galaxy independently
        for gid in df_perm["galaxy_id"].unique():
            mask = df_perm["galaxy_id"] == gid
            idx = df_perm.index[mask]
            df_perm.loc[idx, "log_gbar"] = rng.permutation(
                df_perm.loc[idx, "log_gbar"].values
            )
        # Also update g_bar for consistency (not used directly in model)
        df_perm["g_bar"] = 10.0 ** df_perm["log_gbar"]

        metrics_perm, _, _ = run_groupkfold(df_perm, n_splits=n_splits,
                                            seed=seed + i + 1)
        perm_rmses.append(float(metrics_perm["rmse_log_vobs"].mean()))

    perm_arr = np.array(perm_rmses)
    delta_real = float(np.mean(perm_arr)) - real_rmse  # positive = real beats perm

    # p-value: fraction of permutations that achieved equal or better RMSE
    p_value = float(np.mean(perm_arr <= real_rmse))

    summary = {
        "real_rmse": real_rmse,
        "perm_rmse_mean": float(np.mean(perm_arr)),
        "perm_rmse_std": float(np.std(perm_arr)),
        "perm_rmse_min": float(np.min(perm_arr)),
        "perm_rmse_max": float(np.max(perm_arr)),
        "delta_rmse_real_vs_perm": delta_real,
        "p_value_le": p_value,
        "n_permutations": n_permutations,
        "significant": bool(p_value < 0.05),
    }

    runs_df = pd.DataFrame({
        "permutation": np.arange(n_permutations),
        "rmse_log_vobs": perm_rmses,
    })

    return summary, runs_df


# ---------------------------------------------------------------------------
# 3) Model duel (AICc)
# ---------------------------------------------------------------------------

def run_model_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Fit three nested models on the full dataset and rank by AICc.

    Models
    ------
    btfr      — log_v ~ log_gbar only (baryonic Tully-Fisher analogue)
    scm_base  — log_v ~ log_gbar + extra features (SCM without hinge)
    scm_full  — log_v ~ log_gbar + hinge(log_gbar) + extra features (SCM complete)

    Parameters
    ----------
    df : pd.DataFrame
        Prepared flat dataframe (full dataset).

    Returns
    -------
    pd.DataFrame
        Per-model AICc comparison table.
    """
    y = df["log_vobs"].values
    n = len(y)
    scaler = StandardScaler()

    records = []
    for model_name, base_only, include_hinge in [
        ("btfr",     True,  False),
        ("scm_base", False, False),
        ("scm_full", False, True),
    ]:
        X, feat_names = _select_features(df, include_hinge=include_hinge,
                                         base_only=base_only)
        k_feats = X.shape[1]
        # +1 intercept, +1 sigma → k = k_feats + 2
        k = k_feats + 2

        X_s = scaler.fit_transform(X)
        model = Ridge(alpha=1.0)
        model.fit(X_s, y)
        y_pred = model.predict(X_s)

        ll = _gaussian_ll(y, y_pred)
        aic_c = _aicc(ll, k, n)

        records.append({
            "model": model_name,
            "n_features": k_feats,
            "k_total": k,
            "n_points": n,
            "LL": ll,
            "AICc": aic_c,
            "RMSE_log_vobs": _rmse(y, y_pred),
        })

    comp_df = pd.DataFrame(records)
    best_aicc = comp_df["AICc"].min()
    comp_df["delta_AICc"] = comp_df["AICc"] - best_aicc
    comp_df = comp_df.sort_values("AICc").reset_index(drop=True)
    return comp_df


# ---------------------------------------------------------------------------
# 4) Master coefficient freeze
# ---------------------------------------------------------------------------

def freeze_master_coeffs(df: pd.DataFrame, seed: int = 42) -> dict:
    """Fit the full SCM model on the complete dataset and return coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared flat dataframe (full dataset).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Coefficient dictionary ready for JSON serialisation.
    """
    y = df["log_vobs"].values
    X, feat_names = _select_features(df, include_hinge=True, base_only=False)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = Ridge(alpha=1.0, random_state=seed)
    model.fit(X_s, y)
    y_pred = model.predict(X_s)

    coeffs = {
        "model": "scm_full",
        "n_points": int(len(y)),
        "n_galaxies": int(df["galaxy_id"].nunique()),
        "intercept": float(model.intercept_),
        "features": feat_names,
        "coefficients": [float(c) for c in model.coef_],
        "scaler_mean": [float(m) for m in scaler.mean_],
        "scaler_scale": [float(s) for s in scaler.scale_],
        "RMSE_log_vobs": _rmse(y, y_pred),
    }
    return coeffs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Motor de Velos SCM — reproducible audit pipeline."
    )
    parser.add_argument(
        "--input", required=True, metavar="CSV",
        help="Flat input CSV with galaxy_id, r, v_obs, e_v_obs, …features…",
    )
    parser.add_argument(
        "--outdir", default="results/audit", metavar="DIR",
        help="Output directory (default: results/audit).",
    )
    parser.add_argument(
        "--ref", default="HEAD", metavar="GITREF",
        help="Short git ref to embed in output file names (default: HEAD).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="N",
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--kfold", type=int, default=5, metavar="K",
        help="Number of GroupKFold splits (default: 5).",
    )
    parser.add_argument(
        "--permutations", type=int, default=200, metavar="N",
        help="Number of permutation repetitions (default: 200).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Raise on missing columns instead of warning.",
    )
    parser.add_argument(
        "--skip-permutation", action="store_true",
        help="Skip the permutation test (faster, for quick checks).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = args.ref.replace("/", "_")  # safe for filenames
    suffix = f"-v{ref}"

    print(f"[audit_scm] Loading {args.input} …")
    df = load_and_prepare(Path(args.input), strict=args.strict)
    n_rows = len(df)
    n_gal = df["galaxy_id"].nunique()
    print(f"[audit_scm] {n_rows} rows, {n_gal} galaxies after filtering.")

    if n_gal < args.kfold:
        msg = (f"Not enough galaxies ({n_gal}) for {args.kfold}-fold CV. "
               "Reduce --kfold or provide more data.")
        if args.strict:
            raise ValueError(msg)
        print(f"[WARN] {msg}", file=sys.stderr)
        args.kfold = max(2, n_gal)

    # ------------------------------------------------------------------
    # 1) GroupKFold
    # ------------------------------------------------------------------
    print(f"[audit_scm] Step 1 — GroupKFold(n_splits={args.kfold}) …")
    metrics_df, per_gal_df, coeffs_df = run_groupkfold(
        df, n_splits=args.kfold, seed=args.seed
    )

    metrics_path = out_dir / f"groupkfold_metrics{suffix}.csv"
    per_gal_path = out_dir / f"groupkfold_per_galaxy{suffix}.csv"
    coeffs_path = out_dir / f"coeffs_by_fold{suffix}.csv"

    metrics_df.to_csv(metrics_path, index=False)
    per_gal_df.to_csv(per_gal_path, index=False)
    coeffs_df.to_csv(coeffs_path, index=False)

    mean_rmse = float(metrics_df["rmse_log_vobs"].mean())
    std_rmse = float(metrics_df["rmse_log_vobs"].std())
    print(f"  RMSE (mean ± std): {mean_rmse:.4f} ± {std_rmse:.4f} dex")

    # ------------------------------------------------------------------
    # 2) Permutation test
    # ------------------------------------------------------------------
    if not args.skip_permutation:
        print(f"[audit_scm] Step 2 — Permutation test "
              f"(n={args.permutations}, seed={args.seed}) …")
        perm_summary, perm_runs_df = run_permutation_test(
            df,
            n_splits=args.kfold,
            n_permutations=args.permutations,
            seed=args.seed,
        )

        perm_summary_path = out_dir / f"permutation_summary{suffix}.json"
        perm_runs_path = out_dir / f"permutation_runs{suffix}.csv"

        with open(perm_summary_path, "w", encoding="utf-8") as fh:
            json.dump(perm_summary, fh, indent=2)
        perm_runs_df.to_csv(perm_runs_path, index=False)

        sig = "YES" if perm_summary["significant"] else "NO"
        print(f"  Real RMSE: {perm_summary['real_rmse']:.4f}  "
              f"Perm mean: {perm_summary['perm_rmse_mean']:.4f}  "
              f"ΔRMSE: {perm_summary['delta_rmse_real_vs_perm']:+.4f}  "
              f"p≤0.05: {sig}")
    else:
        print("[audit_scm] Step 2 — Permutation test skipped (--skip-permutation).")

    # ------------------------------------------------------------------
    # 3) Model duel (AICc)
    # ------------------------------------------------------------------
    print("[audit_scm] Step 3 — Model duel (AICc) …")
    comp_df = run_model_comparison(df)
    comp_path = out_dir / f"model_comparison_aicc{suffix}.csv"
    comp_df.to_csv(comp_path, index=False)

    winner = comp_df.iloc[0]["model"]
    print(f"  Winner (lowest AICc): {winner}")
    for _, row in comp_df.iterrows():
        print(f"    {row['model']:<12} AICc={row['AICc']:.2f}  "
              f"ΔAICc={row['delta_AICc']:.2f}  "
              f"RMSE={row['RMSE_log_vobs']:.4f}")

    # ------------------------------------------------------------------
    # 4) Master coefficient freeze
    # ------------------------------------------------------------------
    print("[audit_scm] Step 4 — Freezing master coefficients …")
    master = freeze_master_coeffs(df, seed=args.seed)
    master["git_ref"] = args.ref
    master_path = out_dir / f"master_coeffs{suffix}.json"
    with open(master_path, "w", encoding="utf-8") as fh:
        json.dump(master, fh, indent=2)
    print(f"  Frozen RMSE: {master['RMSE_log_vobs']:.4f}")

    print(f"\n[audit_scm] All outputs written to {out_dir}")


if __name__ == "__main__":
    main()
