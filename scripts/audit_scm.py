#!/usr/bin/env python3
"""
scripts/audit_scm.py
Auditoría quirúrgica del Framework SCM.
Realiza:
- GroupKFold por galaxia (5 folds)
- Permutation test "hard" (baraja log_gbar entre galaxias)
- Comparación de modelos (BTFR, SCM sin hinge, SCM completo) por RMSE OOS
- Estabilidad de coeficientes entre folds
- Exporta coeficientes maestros y summary JSON

Ejecución:
    python scripts/audit_scm.py --input data/sparc_raw.csv --outdir results/audit --seed 123
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import wilcoxon
import sys
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Predicciones
# ------------------------------------------------------------
def btfr_pred(logM, beta, C):
    return beta * logM + C

def scm_no_hinge_pred(logM, log_gbar, log_j, beta, C, a, b):
    return beta * logM + C + a * log_gbar + b * log_j

def scm_full_pred(logM, log_gbar, log_j, beta, C, a, b, d, logg0):
    hinge = d * np.maximum(0, logg0 - log_gbar)
    return beta * logM + C + a * log_gbar + b * log_j + hinge

# ------------------------------------------------------------
# Fitting (per-fold, TRAIN only) — OOS real by galaxy
# ------------------------------------------------------------
def _lstsq_fit(X, y):
    """
    Solve min ||Xw - y||_2. Returns w, rss.
    X must already include intercept column if desired.
    """
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ w
    rss = float(np.sum(resid**2))
    return w, rss

def fit_btfr(y_train, logM_train):
    # y = beta*logM + C
    X = np.column_stack([logM_train, np.ones_like(logM_train)])
    w, rss = _lstsq_fit(X, y_train)
    beta, C = float(w[0]), float(w[1])
    return {"beta": beta, "C": C}, rss

def fit_scm_no_hinge(y_train, logM, log_gbar, log_j):
    # y = beta*logM + C + a*log_gbar + b*log_j
    X = np.column_stack([logM, np.ones_like(logM), log_gbar, log_j])
    w, rss = _lstsq_fit(X, y_train)
    beta, C, a, b = map(float, w)
    return {"beta": beta, "C": C, "a": a, "b": b}, rss

def fit_scm_full(y_train, logM, log_gbar, log_j, logg0):
    # y = beta*logM + C + a*log_gbar + b*log_j + d*max(0, logg0-log_gbar)
    h = np.maximum(0.0, logg0 - log_gbar)
    X = np.column_stack([logM, np.ones_like(logM), log_gbar, log_j, h])
    w, rss = _lstsq_fit(X, y_train)
    beta, C, a, b, d = map(float, w)
    # Enforce physically sensible hinge (d >= 0) with minimal intervention:
    # if d < 0, drop hinge term and refit without it (d=0).
    if d < 0:
        X2 = np.column_stack([logM, np.ones_like(logM), log_gbar, log_j])
        w2, rss2 = _lstsq_fit(X2, y_train)
        beta, C, a, b = map(float, w2)
        d = 0.0
        rss = rss2
    return {"beta": beta, "C": C, "a": a, "b": b, "d": d, "logg0": float(logg0)}, rss

# ------------------------------------------------------------
# Métricas
# ------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def safe_aicc(y_true, y_pred, k):
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    if rss <= 0 or n == 0:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    if n > k + 1:
        aicc = aic + 2 * k * (k + 1) / (n - k - 1)
    else:
        aicc = np.inf
    return float(aicc)

# ------------------------------------------------------------
# Carga y validación
# ------------------------------------------------------------
def load_and_validate(args):
    df = pd.read_csv(args.input)
    required = [args.galaxy_col, args.vobs_col, 'logM', 'log_gbar', 'log_j']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

    if args.strict:
        for col in required[1:]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Columna {col} no es numérica.")

    # Eliminar filas con NaN en las columnas clave
    df = df.dropna(subset=required[1:]).copy()

    # Garantizar que es "galaxy-level" (1 fila por galaxy_id) si se espera:
    # Si hay múltiples filas por galaxy_id, NO fallamos, pero avisamos.
    dup_gals = df[args.galaxy_col].duplicated().sum()
    if dup_gals > 0:
        print(f"WARNING: detected {dup_gals} duplicated galaxy_id rows. This audit assumes 1 row per galaxy for strict interpretation.")

    return df

# ------------------------------------------------------------
# GroupKFold por galaxia
# ------------------------------------------------------------
def groupkfold_audit(df, args, logg0_fixed):
    """
    Ajusta parámetros SOLO en TRAIN y evalúa en TEST (OOS real por galaxia).
    Devuelve: resultados por fold, por galaxia (RMSE gal-level), y coeficientes por fold.
    """
    groups = df[args.galaxy_col]
    X = df[['logM', 'log_gbar', 'log_j']].values
    y = df[args.vobs_col].values  # asumimos que ya es log(v)

    gkf = GroupKFold(n_splits=args.kfold)

    fold_results = []
    gal_results = []
    coeffs_by_fold = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        # Separar datos
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups.iloc[test_idx].values

        # FIT en TRAIN (OOS real)
        btfr_params, _ = fit_btfr(y_train, X_train[:, 0])
        nohinge_params, _ = fit_scm_no_hinge(y_train, X_train[:, 0], X_train[:, 1], X_train[:, 2])
        full_params, _ = fit_scm_full(y_train, X_train[:, 0], X_train[:, 1], X_train[:, 2], logg0_fixed)

        # Predicciones en TEST
        y_pred_btfr = btfr_pred(X_test[:, 0], btfr_params['beta'], btfr_params['C'])
        y_pred_no_hinge = scm_no_hinge_pred(
            X_test[:, 0], X_test[:, 1], X_test[:, 2],
            nohinge_params['beta'], nohinge_params['C'],
            nohinge_params['a'], nohinge_params['b']
        )
        y_pred_full = scm_full_pred(
            X_test[:, 0], X_test[:, 1], X_test[:, 2],
            full_params['beta'], full_params['C'],
            full_params['a'], full_params['b'], full_params['d'], full_params['logg0']
        )

        # Métricas por fold
        rmse_btfr = rmse(y_test, y_pred_btfr)
        rmse_no_hinge = rmse(y_test, y_pred_no_hinge)
        rmse_full = rmse(y_test, y_pred_full)

        # Agregación "por galaxia" (si hay 1 fila por galaxia, coincide; si hay >1, esto es lo correcto)
        df_test = pd.DataFrame({
            "galaxy_id": groups_test,
            "y": y_test,
            "btfr": y_pred_btfr,
            "no_hinge": y_pred_no_hinge,
            "full": y_pred_full
        })
        gal_agg = []
        for gal, sub in df_test.groupby("galaxy_id"):
            r_btfr = sub["y"].values - sub["btfr"].values
            r_no = sub["y"].values - sub["no_hinge"].values
            r_full = sub["y"].values - sub["full"].values
            gal_agg.append({
                "galaxy_id": gal,
                "fold": fold,
                "n_rows": int(len(sub)),
                "rmse_btfr_gal": float(np.sqrt(np.mean(r_btfr**2))),
                "rmse_no_hinge_gal": float(np.sqrt(np.mean(r_no**2))),
                "rmse_full_gal": float(np.sqrt(np.mean(r_full**2))),
                "delta_rmse_full_minus_btfr": float(np.sqrt(np.mean(r_full**2)) - np.sqrt(np.mean(r_btfr**2))),
                "delta_rmse_full_minus_no_hinge": float(np.sqrt(np.mean(r_full**2)) - np.sqrt(np.mean(r_no**2))),
            })
        gal_results.extend(gal_agg)

        fold_results.append({
            'fold': fold,
            'rmse_btfr': rmse_btfr,
            'rmse_no_hinge': rmse_no_hinge,
            'rmse_full': rmse_full,
            'n_test_rows': int(len(test_idx)),
            'n_test_galaxies': int(df_test["galaxy_id"].nunique())
        })

        # Coeficientes por fold (para "universalidad")
        coeffs_by_fold.append({
            "fold": fold,
            "btfr_beta": btfr_params["beta"],
            "btfr_C": btfr_params["C"],
            "nohinge_beta": nohinge_params["beta"],
            "nohinge_C": nohinge_params["C"],
            "nohinge_a": nohinge_params["a"],
            "nohinge_b": nohinge_params["b"],
            "full_beta": full_params["beta"],
            "full_C": full_params["C"],
            "full_a": full_params["a"],
            "full_b": full_params["b"],
            "full_d": full_params["d"],
            "full_logg0_fixed": full_params["logg0"],
        })

    return fold_results, gal_results, coeffs_by_fold

# ------------------------------------------------------------
# Permutation test
# ------------------------------------------------------------
def permutation_test(df, args, logg0_fixed):
    """
    Permuta log_gbar ENTRE galaxias (hard test) y repite GroupKFold con fit en TRAIN.
    Devuelve:
      - lista de RMSE medios por permutación (full model)
      - rmse_real (full model) del audit real
      - p-value empírico (menor o igual)
    """
    groups = df[args.galaxy_col]
    X_base = df[['logM', 'log_gbar', 'log_j']].values
    y = df[args.vobs_col].values

    gkf = GroupKFold(n_splits=args.kfold)
    rng = np.random.default_rng(args.seed)

    # RMSE real (re-ajustando en TRAIN)
    rmse_real_folds = []
    for train_idx, test_idx in gkf.split(X_base, y, groups):
        X_train, X_test = X_base[train_idx], X_base[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        full_params, _ = fit_scm_full(y_train, X_train[:, 0], X_train[:, 1], X_train[:, 2], logg0_fixed)
        y_pred = scm_full_pred(
            X_test[:, 0], X_test[:, 1], X_test[:, 2],
            full_params["beta"], full_params["C"],
            full_params["a"], full_params["b"],
            full_params["d"], full_params["logg0"]
        )
        rmse_real_folds.append(rmse(y_test, y_pred))
    rmse_real = float(np.mean(rmse_real_folds))

    perm_rmse = []

    for _ in range(args.permutations):
        # Baraja log_gbar entre galaxias (mantiene marginal pero rompe asociación)
        X_perm = X_base.copy()
        X_perm[:, 1] = rng.permutation(X_perm[:, 1])

        rmse_folds = []
        for train_idx, test_idx in gkf.split(X_perm, y, groups):
            X_train, X_test = X_perm[train_idx], X_perm[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            full_params, _ = fit_scm_full(y_train, X_train[:, 0], X_train[:, 1], X_train[:, 2], logg0_fixed)
            y_pred = scm_full_pred(
                X_test[:, 0], X_test[:, 1], X_test[:, 2],
                full_params["beta"], full_params["C"],
                full_params["a"], full_params["b"],
                full_params["d"], full_params["logg0"]
            )
            rmse_folds.append(rmse(y_test, y_pred))
        perm_rmse.append(np.mean(rmse_folds))

    perm_rmse = [float(x) for x in perm_rmse]
    # Empirical p-value: fraction of perms that are as good or better than real
    p_emp = float((1 + sum(r <= rmse_real for r in perm_rmse)) / (1 + len(perm_rmse)))
    return perm_rmse, rmse_real, p_emp

# ------------------------------------------------------------
# Comparación AICc por galaxia
# ------------------------------------------------------------
def model_comparison(df, args, coeffs):
    """
    Comparación de modelos (BTFR, SCM sin hinge, SCM completo) por AICc usando
    datos radiales (una fila por punto radial por galaxia).
    Devuelve DataFrame con AICc por galaxia para cada modelo, o None si falla.
    """
    if not args.radial:
        return None

    try:
        df_rad = pd.read_csv(args.radial)
    except Exception as e:
        print(f"WARNING: No se pudo cargar datos radiales: {e}")
        return None

    req_rad = [args.galaxy_col, args.vobs_col, 'logM', 'log_gbar', 'log_j']
    missing = [c for c in req_rad if c not in df_rad.columns]
    if missing:
        print(f"WARNING: Columnas faltantes en datos radiales: {missing}")
        return None

    rows = []
    for gal, gdf in df_rad.groupby(args.galaxy_col):
        logM = gdf['logM'].values
        log_gbar = gdf['log_gbar'].values
        log_j = gdf['log_j'].values
        y_obs = gdf[args.vobs_col].values
        n = len(y_obs)
        if n < 3:
            continue

        # BTFR (k=2: beta, C)
        y_btfr = btfr_pred(logM, coeffs['beta'], coeffs['C'])
        aicc_btfr = safe_aicc(y_obs, y_btfr, k=2)

        # SCM sin hinge (k=4: beta, C, a, b)
        y_no_hinge = scm_no_hinge_pred(logM, log_gbar, log_j,
                                        coeffs['beta'], coeffs['C'],
                                        coeffs['a'], coeffs['b'])
        aicc_no_hinge = safe_aicc(y_obs, y_no_hinge, k=4)

        # SCM completo (k=5: beta, C, a, b, d)
        y_full = scm_full_pred(logM, log_gbar, log_j,
                               coeffs['beta'], coeffs['C'],
                               coeffs['a'], coeffs['b'], coeffs['d'], coeffs['logg0'])
        aicc_full = safe_aicc(y_obs, y_full, k=5)

        best = min(
            [('btfr', aicc_btfr), ('no_hinge', aicc_no_hinge), ('full', aicc_full)],
            key=lambda x: x[1]
        )[0]

        rows.append({
            'galaxy_id': gal,
            'n': n,
            'aicc_btfr': aicc_btfr,
            'aicc_no_hinge': aicc_no_hinge,
            'aicc_full': aicc_full,
            'best_model': best,
            'delta_aicc_full_vs_btfr': aicc_full - aicc_btfr,
        })

    if not rows:
        return None
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Coeficientes maestros
# ------------------------------------------------------------
def fit_master_coeffs(df, args):
    if args.coeffs:
        with open(args.coeffs) as f:
            coeffs = json.load(f)
    else:
        # Coeficientes por defecto (de tu última versión)
        coeffs = {
            'beta': 0.324,
            'C': 0.0,
            'a': -0.048,
            'b': 0.040,
            'd': 0.062,
            'logg0': -10.45
        }
    return coeffs

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Auditoría quirúrgica del Framework SCM')
    parser.add_argument('--input', required=True, help='CSV con datos globales por galaxia (logM, log_gbar, log_j, v_obs, galaxy_id)')
    parser.add_argument('--radial', help='CSV opcional con datos radiales para cálculo de AICc por galaxia')
    parser.add_argument('--outdir', default='results/audit', help='Directorio de salida')
    parser.add_argument('--galaxy-col', default='galaxy_id', dest='galaxy_col', help='Nombre de la columna con identificador de galaxia')
    parser.add_argument('--vobs-col', default='v_obs', dest='vobs_col', help='Nombre de la columna con velocidad observada (en log)')
    parser.add_argument('--coeffs', help='Archivo JSON con coeficientes fijos (si no se da, usa valores por defecto)')
    parser.add_argument('--kfold', type=int, default=5, help='Número de folds en GroupKFold')
    parser.add_argument('--permutations', type=int, default=200, help='Número de permutaciones en test')
    parser.add_argument('--seed', type=int, default=42, help='Semilla aleatoria')
    parser.add_argument('--strict', action='store_true', help='Activar validaciones estrictas')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Cargar y validar datos
    df = load_and_validate(args)

    # Obtener logg0 fijo (de archivo o defaults)
    coeffs_in = fit_master_coeffs(df, args)
    logg0_fixed = float(coeffs_in.get("logg0", -10.45))

    # 1. GroupKFold
    print("Ejecutando GroupKFold...")
    fold_res, gal_res, coeffs_fold = groupkfold_audit(df, args, logg0_fixed)

    # Guardar resultados
    pd.DataFrame(fold_res).to_csv(outdir / 'groupkfold_metrics.csv', index=False)
    pd.DataFrame(gal_res).to_csv(outdir / 'groupkfold_per_galaxy.csv', index=False)
    pd.DataFrame(coeffs_fold).to_csv(outdir / 'coeffs_by_fold.csv', index=False)

    # Wilcoxon on ΔRMSE (galaxy-level). Alternative="less": full better => delta < 0.
    gal_df = pd.DataFrame(gal_res)
    wilcoxon_p = None
    wilcoxon_n = int(len(gal_df))
    if wilcoxon_n >= 5:
        try:
            stat, p = wilcoxon(gal_df["delta_rmse_full_minus_btfr"].values, alternative="less")
            wilcoxon_p = float(p)
        except Exception:
            wilcoxon_p = None

    # 2. Permutation test
    print("Ejecutando permutation test...")
    perm_rmse, rmse_real_perm, p_emp = permutation_test(df, args, logg0_fixed)
    perm_mean = float(np.mean(perm_rmse))
    perm_std = float(np.std(perm_rmse))
    perm_summary = {
        'rmse_real_full': float(rmse_real_perm),
        'perm_mean_rmse': perm_mean,
        'perm_std_rmse': perm_std,
        'permutations': int(args.permutations),
        'p_empirical_perm_leq_real': float(p_emp)
    }
    with open(outdir / 'permutation_summary.json', 'w') as f:
        json.dump(perm_summary, f, indent=2)
    pd.DataFrame({'perm_rmse': perm_rmse}).to_csv(outdir / 'permutation_runs.csv', index=False)

    # 3. Comparación de modelos AICc (si hay datos radiales)
    if args.radial:
        print("Calculando AICc por galaxia...")
        aicc_df = model_comparison(df, args, coeffs_in)
        if aicc_df is not None:
            aicc_df.to_csv(outdir / 'model_comparison_aicc.csv', index=False)

    # 4. Coeficientes maestros (fit global sobre TODO el dataset, para congelar)
    # NOTE: This is descriptive — the scientific claim still rests on OOS.
    X_all = df[['logM', 'log_gbar', 'log_j']].values
    y_all = df[args.vobs_col].values
    btfr_master, _ = fit_btfr(y_all, X_all[:, 0])
    nohinge_master, _ = fit_scm_no_hinge(y_all, X_all[:, 0], X_all[:, 1], X_all[:, 2])
    full_master, _ = fit_scm_full(y_all, X_all[:, 0], X_all[:, 1], X_all[:, 2], logg0_fixed)
    master = {
        "logg0_fixed": float(logg0_fixed),
        "btfr": btfr_master,
        "scm_no_hinge": nohinge_master,
        "scm_full": full_master
    }
    with open(outdir / 'master_coeffs.json', 'w') as f:
        json.dump(master, f, indent=2)

    # 5. Summary JSON (single source for ROA / paper numbers)
    fold_df = pd.DataFrame(fold_res)
    summary = {
        "kfold": int(args.kfold),
        "n_galaxies": int(df[args.galaxy_col].nunique()),
        "rmse_btfr_mean": float(fold_df["rmse_btfr"].mean()),
        "rmse_btfr_std": float(fold_df["rmse_btfr"].std()),
        "rmse_no_hinge_mean": float(fold_df["rmse_no_hinge"].mean()),
        "rmse_no_hinge_std": float(fold_df["rmse_no_hinge"].std()),
        "rmse_full_mean": float(fold_df["rmse_full"].mean()),
        "rmse_full_std": float(fold_df["rmse_full"].std()),
        "median_delta_rmse_full_minus_btfr_gal": float(gal_df["delta_rmse_full_minus_btfr"].median()),
        "wilcoxon_p_delta_rmse_full_minus_btfr_less": wilcoxon_p,
        "perm_p_empirical": float(p_emp),
        "perm_rmse_real_full": float(rmse_real_perm),
        "logg0_fixed": float(logg0_fixed),
    }
    with open(outdir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 5. Resumen rápido
    print("\n=== RESUMEN AUDITORÍA ===")
    print(f"RMSE BTFR: {fold_df['rmse_btfr'].mean():.4f} ± {fold_df['rmse_btfr'].std():.4f}")
    print(f"RMSE SCM sin hinge: {fold_df['rmse_no_hinge'].mean():.4f} ± {fold_df['rmse_no_hinge'].std():.4f}")
    print(f"RMSE SCM completo: {fold_df['rmse_full'].mean():.4f} ± {fold_df['rmse_full'].std():.4f}")
    print(f"Mejora SCM full vs BTFR: {(fold_df['rmse_btfr'].mean() - fold_df['rmse_full'].mean()) / fold_df['rmse_btfr'].mean() * 100:.2f}%")
    if wilcoxon_p is not None:
        print(f"Wilcoxon (ΔRMSE_gal full-btfr < 0): p = {wilcoxon_p:.3e} (n={wilcoxon_n})")
    print(f"Permutation (hard): RMSE_real_full = {rmse_real_perm:.4f}; perm mean = {perm_mean:.4f} ± {perm_std:.4f}; p_emp = {p_emp:.3e}")
    print(f"Resultados guardados en {outdir}")

if __name__ == '__main__':
    main()
