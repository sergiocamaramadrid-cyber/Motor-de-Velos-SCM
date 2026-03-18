#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analisis_persistencia_delta_f3.py

Exploratory persistence test for ΔF3.

This script evaluates whether ΔF3 shows linear or quadratic recurrence
structure under inter-galaxy or intra-galaxy ordering schemes.

Current repository interpretation:
- technically validated
- scientifically useful as a falsation module
- no robust positive signal established in the inter-galaxy formulation
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-12
MIN_POINTS_RECURRENCIA = 6


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_bool_filter(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        return df[df[col].astype(bool)]
    return df


def pick_first_existing(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"No se encontró columna válida para {label}. "
        f"Candidatas probadas: {candidates}"
    )


def require_columns(df: pd.DataFrame, required: list[str], context: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas para {context}: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )


def compute_aicc(rss: float, n: int, k: int) -> float:
    rss = max(float(rss), EPS)
    if n <= k + 1:
        return np.inf
    aic = n * np.log(rss / n) + 2 * k
    return aic + (2 * k * (k + 1)) / (n - k - 1)


def finite_clean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=cols)
    return out


def is_valid_delta_pair(delta_i: float, delta_j: float) -> bool:
    return (
        np.isfinite(delta_i)
        and np.isfinite(delta_j)
        and abs(float(delta_i)) > EPS
        and abs(float(delta_j)) > EPS
    )


def model_null(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    c = p[0]
    return np.full_like(x, c, dtype=float)


def model_linear(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, c = p
    return a * x + c


def model_quadratic(x: np.ndarray, p: np.ndarray) -> np.ndarray:
    a, b, c = p
    return a * x + b * x**2 + c


MODEL_FUNCS = {
    "nulo": model_null,
    "linear": model_linear,
    "quadratic": model_quadratic,
}


MODEL_K = {
    "nulo": 1,
    "linear": 2,
    "quadratic": 3,
}


MODEL_P0 = {
    "nulo": np.array([0.0], dtype=float),
    "linear": np.array([0.8, 0.0], dtype=float),
    "quadratic": np.array([0.8, 0.1, 0.0], dtype=float),
}


def ajustar_modelo(dF3: np.ndarray) -> dict:
    """Ajusta recurrencia de una serie ΔF3 con modelos nulo/lineal/cuadrático.

    Parameters
    ----------
    dF3
        Serie unidimensional con valores de ΔF3. Se filtran automáticamente
        valores no finitos.

    Returns
    -------
    dict
        Coeficientes y métricas AICc/RSS para los tres modelos, incluyendo:
        a/b/c del cuadrático y deltas de AICc frente a nulo y lineal.

    Raises
    ------
    ValueError
        Si tras limpiar no hay al menos 6 puntos válidos para construir pares
        x=dF3[:-1], y=dF3[1:].
    """
    dF3 = np.asarray(dF3, dtype=float)
    dF3 = dF3[np.isfinite(dF3)]

    # 6 puntos => 5 pares (x,y), mínimo razonable para comparar nulo/lineal/cuadrático.
    if len(dF3) < MIN_POINTS_RECURRENCIA:
        raise ValueError("Muy pocos puntos para ajustar recurrencia.")

    x = dF3[:-1]
    y = dF3[1:]
    n = len(y)

    x_mean = float(np.mean(x))
    x0 = x - x_mean

    c0 = float(np.mean(y))
    yhat_null = np.full_like(y, c0)
    rss_null = float(np.sum((y - yhat_null) ** 2))
    aicc_null = compute_aicc(rss_null, n, 1)

    A_lin = np.column_stack([x0, np.ones_like(x0)])
    # Se descartan diagnósticos de lstsq porque aquí solo se usan coeficientes y RSS.
    coef_lin, *_ = np.linalg.lstsq(A_lin, y, rcond=None)
    a_lin, c_lin = coef_lin
    yhat_lin = A_lin @ coef_lin
    rss_lin = float(np.sum((y - yhat_lin) ** 2))
    aicc_lin = compute_aicc(rss_lin, n, 2)

    A_quad = np.column_stack([x0, x0**2, np.ones_like(x0)])
    # Se descartan diagnósticos de lstsq porque aquí solo se usan coeficientes y RSS.
    coef_quad, *_ = np.linalg.lstsq(A_quad, y, rcond=None)
    a_quad, b_quad, c_quad = coef_quad
    yhat_quad = A_quad @ coef_quad
    rss_quad = float(np.sum((y - yhat_quad) ** 2))
    aicc_quad = compute_aicc(rss_quad, n, 3)

    return {
        "x_mean": x_mean,
        "a": float(a_quad),
        "b": float(b_quad),
        "c": float(c_quad),
        "a_lin": float(a_lin),
        "c_lin": float(c_lin),
        "c_null": c0,
        "rss_null": rss_null,
        "rss_lin": rss_lin,
        "rss_quad": rss_quad,
        "aicc_null": float(aicc_null),
        "aicc_lin": float(aicc_lin),
        "aicc_quad": float(aicc_quad),
        "delta_aicc_quad_vs_lin": float(aicc_quad - aicc_lin),
        "delta_aicc_quad_vs_null": float(aicc_quad - aicc_null),
    }


def fit_model(name: str, x: np.ndarray, y: np.ndarray) -> dict:
    """Ajusta un modelo individual usando mínimos cuadrados sobre x centrado.

    El centrado en la media de `x` mejora estabilidad numérica para términos
    polinómicos. El resultado devuelve parámetros, RSS, AICc y predicción.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = float(np.mean(x))
    x0 = x - x_mean
    n = len(x)

    if name == "nulo":
        c = float(np.mean(y))
        yhat = np.full_like(y, c)
        rss = float(np.sum((y - yhat) ** 2))
        params = {"c": c, "x_mean": x_mean}
    elif name == "linear":
        design = np.column_stack([x0, np.ones_like(x0)])
        # Se descartan diagnósticos de lstsq porque aquí solo se usan coeficientes y RSS.
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        a, c = coef
        yhat = design @ coef
        rss = float(np.sum((y - yhat) ** 2))
        params = {"a": float(a), "c": float(c), "x_mean": x_mean}
    else:
        design = np.column_stack([x0, x0**2, np.ones_like(x0)])
        # Se descartan diagnósticos de lstsq porque aquí solo se usan coeficientes y RSS.
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        a, b, c = coef
        yhat = design @ coef
        rss = float(np.sum((y - yhat) ** 2))
        params = {"a": float(a), "b": float(b), "c": float(c), "x_mean": x_mean}

    aicc = compute_aicc(rss, n, MODEL_K[name])
    return {
        "model": name,
        "params": params,
        "k": MODEL_K[name],
        "rss": rss,
        "aicc": aicc,
        "yhat": yhat,
    }


def build_intra_galaxy_pairs(
    df: pd.DataFrame,
    galaxy_col: str,
    f3_col: str,
    order_col: str,
    min_points_per_galaxy: int = 4,
) -> pd.DataFrame:
    pairs: list[dict[str, float | str]] = []

    for galaxy, gdf in df.groupby(galaxy_col, sort=False):
        gdf = gdf.sort_values(order_col).reset_index(drop=True)
        if len(gdf) < min_points_per_galaxy:
            continue

        f3 = gdf[f3_col].to_numpy(dtype=float)
        scale = gdf[order_col].to_numpy(dtype=float)

        delta = np.diff(f3)
        scale_mid = 0.5 * (scale[:-1] + scale[1:])
        if len(delta) < 2:
            # Se requieren al menos 2 deltas para construir (delta_i, delta_{i+1}).
            continue

        for idx in range(len(delta) - 1):
            if not is_valid_delta_pair(float(delta[idx]), float(delta[idx + 1])):
                continue
            pairs.append(
                {
                    "mode": "intra-galaxy",
                    "galaxy": str(galaxy),
                    "order_i": float(scale_mid[idx]),
                    "order_j": float(scale_mid[idx + 1]),
                    "delta_f3_i": float(delta[idx]),
                    "delta_f3_j": float(delta[idx + 1]),
                }
            )

    return pd.DataFrame(pairs)


def build_inter_galaxy_pairs(
    df: pd.DataFrame,
    galaxy_col: str,
    f3_col: str,
    order_col: str,
    delta_col: str | None = None,
) -> pd.DataFrame:
    cols = [galaxy_col, order_col]
    if delta_col is None:
        cols.append(f3_col)
    else:
        cols.append(delta_col)
    work = df[cols].copy().sort_values(order_col).reset_index(drop=True)
    if len(work) < 4:
        raise ValueError("No hay suficientes filas para construir pares inter-galaxy.")

    order = work[order_col].to_numpy(dtype=float)
    galaxy = work[galaxy_col].astype(str).to_numpy()
    if delta_col is None:
        f3 = work[f3_col].to_numpy(dtype=float)
        delta = np.diff(f3)
        order_i = 0.5 * (order[:-2] + order[1:-1])
        order_j = 0.5 * (order[1:-1] + order[2:])
        galaxy_i = galaxy[:-2]
        galaxy_j = galaxy[1:-1]
    else:
        delta = work[delta_col].to_numpy(dtype=float)
        order_i = order[:-1]
        order_j = order[1:]
        galaxy_i = galaxy[:-1]
        galaxy_j = galaxy[1:]
    if len(delta) < 2:
        raise ValueError("No hay suficientes ΔF3 para construir pares inter-galaxy.")

    rows: list[dict[str, float | str]] = []
    for idx in range(len(delta) - 1):
        if not is_valid_delta_pair(float(delta[idx]), float(delta[idx + 1])):
            continue
        rows.append(
            {
                "mode": "inter-galaxy",
                "galaxy_i": galaxy_i[idx],
                "galaxy_j": galaxy_j[idx],
                "order_i": float(order_i[idx]),
                "order_j": float(order_j[idx]),
                "delta_f3_i": float(delta[idx]),
                "delta_f3_j": float(delta[idx + 1]),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_quadratic(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    n = len(x)
    failures = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb = x[idx]
        yb = y[idx]
        try:
            fit = fit_model("quadratic", xb, yb)
        except (ValueError, RuntimeError, FloatingPointError):
            failures += 1
            continue
        rows.append(
            {
                "a": fit["params"]["a"],
                "b": fit["params"]["b"],
                "c": fit["params"]["c"],
                "rss": fit["rss"],
                "aicc": fit["aicc"],
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["failed_fits"] = failures
    return out


def make_figure(
    pairs_df: pd.DataFrame,
    best_fit: dict,
    out_png: str,
    title: str,
) -> None:
    x = pairs_df["delta_f3_i"].to_numpy(dtype=float)
    y = pairs_df["delta_f3_j"].to_numpy(dtype=float)

    plt.figure(figsize=(7.6, 6.2))
    plt.scatter(x, y, alpha=0.75, edgecolors="k", linewidth=0.4, label="Pares observados")

    x_line = np.linspace(np.min(x), np.max(x), 300)
    if best_fit["model"] == "nulo":
        c = best_fit["params"]["c"]
        y_line = np.full_like(x_line, c)
        label = f"Nulo: y = {c:.4g}"
    elif best_fit["model"] == "linear":
        a = best_fit["params"]["a"]
        c = best_fit["params"]["c"]
        x0_line = x_line - best_fit["params"]["x_mean"]
        y_line = a * x0_line + c
        label = f"Lineal: y = {a:.4g} (x-μ) + {c:.4g}"
    elif best_fit["model"] == "quadratic":
        b = best_fit["params"]["b"]
        a = best_fit["params"]["a"]
        c = best_fit["params"]["c"]
        x0_line = x_line - best_fit["params"]["x_mean"]
        y_line = b * x0_line**2 + a * x0_line + c
        label = f"Cuadrático: y = {b:.4g} (x-μ)² + {a:.4g} (x-μ) + {c:.4g}"
    else:
        raise ValueError(f"Modelo no soportado para figura: {best_fit['model']}")

    plt.plot(x_line, y_line, linewidth=2.2, label=label)
    plt.axhline(0.0, linestyle="--", linewidth=0.8)
    plt.axvline(0.0, linestyle="--", linewidth=0.8)
    plt.xlabel(r"$\Delta F3_i$")
    plt.ylabel(r"$\Delta F3_{i+1}$")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Análisis de persistencia en ΔF3.")
    parser.add_argument("--input", required=True, help="CSV de entrada.")
    parser.add_argument("--mode", choices=["intra-galaxy", "inter-galaxy"], default="intra-galaxy")
    parser.add_argument("--outdir", default="results/delta_f3_persistence")
    parser.add_argument("--galaxy-col", default=None)
    parser.add_argument("--f3-col", default=None)
    parser.add_argument("--order-col", default=None)
    parser.add_argument("--delta-col", default=None)
    parser.add_argument("--filter-fit-ok", action="store_true", help="Filtra fit_ok == True si existe.")
    parser.add_argument(
        "--filter-reliable",
        action="store_true",
        help="Filtra reliable == True si existe.",
    )
    parser.add_argument(
        "--min-points-per-galaxy",
        type=int,
        default=4,
        help="Mínimo de puntos por galaxia para modo intra-galaxy.",
    )
    parser.add_argument("--n-boot", type=int, default=1000, help="Número de remuestreos bootstrap.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para bootstrap.")
    args = parser.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.input)
    if args.filter_fit_ok:
        df = safe_bool_filter(df, "fit_ok")
    if args.filter_reliable:
        df = safe_bool_filter(df, "reliable")

    galaxy_col = args.galaxy_col or pick_first_existing(df, ["galaxy", "galaxy_id", "name"], "galaxy")
    if args.f3_col:
        f3_col = args.f3_col
    else:
        try:
            f3_col = pick_first_existing(df, ["F3", "f3_scm", "f3"], "F3")
        except ValueError:
            if args.delta_col:
                f3_col = args.delta_col
            else:
                raise

    if args.mode == "intra-galaxy":
        order_col = args.order_col or pick_first_existing(
            df, ["r_kpc", "radius_kpc", "R_kpc", "r"], "order_col (intra-galaxy)"
        )
        delta_col = args.delta_col
    else:
        order_col = args.order_col or pick_first_existing(
            df, ["logMbar", "Mbar", "logM"], "order_col (inter-galaxy)"
        )
        delta_col = args.delta_col or pick_first_existing(
            df, ["delta_f3", "delta_F3"], "delta_col (inter-galaxy)"
        )
        require_columns(df, [delta_col, order_col], "inter-galaxy")

    clean_cols = [galaxy_col, order_col, f3_col]
    if delta_col:
        clean_cols.append(delta_col)
    df = finite_clean(df, clean_cols)

    if args.mode == "intra-galaxy":
        pairs_df = build_intra_galaxy_pairs(
            df=df,
            galaxy_col=galaxy_col,
            f3_col=f3_col,
            order_col=order_col,
            min_points_per_galaxy=args.min_points_per_galaxy,
        )
    else:
        pairs_df = build_inter_galaxy_pairs(
            df=df,
            galaxy_col=galaxy_col,
            f3_col=f3_col,
            order_col=order_col,
            delta_col=delta_col,
        )

    min_pairs_required = 5 if args.mode == "intra-galaxy" else 2
    if pairs_df.empty or len(pairs_df) < min_pairs_required:
        raise ValueError(
            f"Se generaron muy pocos pares de persistencia (n={len(pairs_df)}; mínimo={min_pairs_required}). "
            "Revisa columnas, filtros o el modo seleccionado."
        )

    x = pairs_df["delta_f3_i"].to_numpy(dtype=float)
    y = pairs_df["delta_f3_j"].to_numpy(dtype=float)
    fits = [
        fit_model("nulo", x, y),
        fit_model("linear", x, y),
        fit_model("quadratic", x, y),
    ]

    model_rows = []
    for fit in fits:
        row = {
            "model": fit["model"],
            "rss": fit["rss"],
            "aicc": fit["aicc"],
        }
        row.update(fit["params"])
        model_rows.append(row)

    models_df = pd.DataFrame(model_rows).sort_values("aicc").reset_index(drop=True)
    best_model_name = str(models_df.iloc[0]["model"])
    best_fit = next(fit for fit in fits if fit["model"] == best_model_name)

    pairs_csv = os.path.join(args.outdir, "delta_f3_pairs.csv")
    models_csv = os.path.join(args.outdir, "delta_f3_model_comparison.csv")
    summary_txt = os.path.join(args.outdir, "delta_f3_summary.txt")
    fig_png = os.path.join(args.outdir, "delta_f3_persistence_fit.png")
    boot_csv = os.path.join(args.outdir, "delta_f3_bootstrap_quadratic.csv")

    pairs_df.to_csv(pairs_csv, index=False)
    models_df.to_csv(models_csv, index=False)
    boot_df = bootstrap_quadratic(x, y, n_boot=args.n_boot, seed=args.seed)
    boot_df.to_csv(boot_csv, index=False)
    make_figure(pairs_df=pairs_df, best_fit=best_fit, out_png=fig_png, title=f"Persistencia en ΔF3 ({args.mode})")

    fit_null = next(f for f in fits if f["model"] == "nulo")
    fit_linear = next(f for f in fits if f["model"] == "linear")
    fit_quadratic = next(f for f in fits if f["model"] == "quadratic")
    delta_aicc_quadratic_vs_null = fit_quadratic["aicc"] - fit_null["aicc"]
    delta_aicc_linear_vs_null = fit_linear["aicc"] - fit_null["aicc"]

    with open(summary_txt, "w", encoding="utf-8") as fout:
        fout.write("ANÁLISIS DE PERSISTENCIA EN ΔF3\n")
        fout.write("================================\n")
        fout.write(f"Archivo de entrada: {args.input}\n")
        fout.write(f"Modo: {args.mode}\n")
        fout.write(f"Columna galaxia: {galaxy_col}\n")
        fout.write(f"Columna F3: {f3_col}\n")
        fout.write(f"Columna orden: {order_col}\n")
        fout.write("Interpretación del repositorio:\n")
        fout.write("- technically validated\n")
        fout.write("- scientifically useful as a falsation module\n")
        fout.write("- no robust positive signal established in the inter-galaxy formulation\n")
        fout.write(
            "Interpretación técnica: secuencia global inter-galaxy ordenada por masa/escala; "
            "no es dinámica radial interna.\n"
        )
        fout.write(f"Número de pares: {len(pairs_df)}\n\n")
        fout.write("Comparación de modelos (ordenado por AICc):\n")
        fout.write(models_df.to_string(index=False))
        fout.write("\n\n")
        fout.write(f"Mejor modelo: {best_model_name}\n")
        fout.write(f"ΔAICc (quadratic - nulo): {delta_aicc_quadratic_vs_null:.6f}\n")
        fout.write(f"ΔAICc (linear - nulo): {delta_aicc_linear_vs_null:.6f}\n")
        if not boot_df.empty:
            fout.write("\nBootstrap 95% (2.5 | 50 | 97.5):\n")
            for par in ["a", "b", "c"]:
                vals = boot_df[par].to_numpy(dtype=float)
                lo, med, hi = (
                    float(np.nanpercentile(vals, 2.5)),
                    float(np.nanpercentile(vals, 50)),
                    float(np.nanpercentile(vals, 97.5)),
                )
                fout.write(f"{par}: {lo:.6g} | {med:.6g} | {hi:.6g}\n")

    print("\n=== Persistencia en ΔF3 ===")
    print(f"Archivo de entrada : {args.input}")
    print(f"Modo               : {args.mode}")
    print(f"Columna galaxia    : {galaxy_col}")
    print(f"Columna F3         : {f3_col}")
    print(f"Columna orden      : {order_col}")
    print(f"Número de pares    : {len(pairs_df)}")
    print("\n--- COMPARACIÓN DE MODELOS (AICc) ---")
    print(models_df.to_string(index=False))
    print(f"\nMejor modelo       : {best_model_name}")
    print(f"ΔAICc cuadrático-nulo: {delta_aicc_quadratic_vs_null:.6f}")
    print(f"ΔAICc lineal-nulo    : {delta_aicc_linear_vs_null:.6f}")
    if not boot_df.empty:
        print("\n--- BOOTSTRAP 95% ---")
        for par in ["a", "b", "c"]:
            vals = boot_df[par].to_numpy(dtype=float)
            lo, med, hi = (
                float(np.nanpercentile(vals, 2.5)),
                float(np.nanpercentile(vals, 50)),
                float(np.nanpercentile(vals, 97.5)),
            )
            print(f"{par}: {lo:.6g} | {med:.6g} | {hi:.6g}")
    print("\nSalidas:")
    print(f"- Pares            : {pairs_csv}")
    print(f"- Modelos          : {models_csv}")
    print(f"- Bootstrap        : {boot_csv}")
    print(f"- Resumen          : {summary_txt}")
    print(f"- Figura           : {fig_png}")
    failed_boot = int(boot_df.attrs.get("failed_fits", 0))
    if failed_boot:
        print(f"- Bootstrap fallidos: {failed_boot}")


if __name__ == "__main__":
    main()
