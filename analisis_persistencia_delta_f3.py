#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis de persistencia para ΔF3 en SPARC / SCM.

Modos:
1) intra-galaxy:
   construye pares (ΔF3_i, ΔF3_{i+1}) dentro de cada galaxia,
   ordenando por una variable de escala (por defecto r_kpc).

2) inter-galaxy:
   construye pares entre objetos ordenados por una variable global
   (por defecto logMbar). Útil como test de estructura global del catálogo,
   no como dinámica interna estricta.

Modelos comparados:
- nulo:       y = c
- lineal:     y = a x + c
- cuadrático: y = b x^2 + a x + c

Salidas:
- CSV con pares
- CSV con comparación de modelos
- PNG con figura
- TXT resumen
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EPS = 1e-12


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


def compute_aicc(n: int, rss: float, k: int) -> float:
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


def fit_null(x: np.ndarray, y: np.ndarray) -> dict:
    c = float(np.mean(y))
    yhat = np.full_like(y, c, dtype=float)
    rss = float(np.sum((y - yhat) ** 2))
    return {
        "model": "nulo",
        "params": {"c": c},
        "k": 1,
        "rss": rss,
        "yhat": yhat,
    }


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict:
    a, c = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    yhat = a * x + c
    rss = float(np.sum((y - yhat) ** 2))
    return {
        "model": "linear",
        "params": {"a": float(a), "c": float(c)},
        "k": 2,
        "rss": rss,
        "yhat": yhat,
    }


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> dict:
    b, a, c = np.linalg.lstsq(np.vstack([x**2, x, np.ones_like(x)]).T, y, rcond=None)[0]
    yhat = b * x**2 + a * x + c
    rss = float(np.sum((y - yhat) ** 2))
    return {
        "model": "quadratic",
        "params": {"b": float(b), "a": float(a), "c": float(c)},
        "k": 3,
        "rss": rss,
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
            continue

        for idx in range(len(delta) - 1):
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
) -> pd.DataFrame:
    work = df[[galaxy_col, order_col, f3_col]].copy().sort_values(order_col).reset_index(drop=True)
    if len(work) < 4:
        raise ValueError("No hay suficientes filas para construir pares inter-galaxy.")

    f3 = work[f3_col].to_numpy(dtype=float)
    order = work[order_col].to_numpy(dtype=float)
    galaxy = work[galaxy_col].astype(str).to_numpy()

    delta = np.diff(f3)
    order_mid = 0.5 * (order[:-1] + order[1:])
    if len(delta) < 2:
        raise ValueError("No hay suficientes ΔF3 para construir pares inter-galaxy.")

    rows: list[dict[str, float | str]] = []
    for idx in range(len(delta) - 1):
        rows.append(
            {
                "mode": "inter-galaxy",
                "galaxy_i": galaxy[idx],
                "galaxy_j": galaxy[idx + 1],
                "order_i": float(order_mid[idx]),
                "order_j": float(order_mid[idx + 1]),
                "delta_f3_i": float(delta[idx]),
                "delta_f3_j": float(delta[idx + 1]),
            }
        )
    return pd.DataFrame(rows)


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
        y_line = a * x_line + c
        label = f"Lineal: y = {a:.4g} x + {c:.4g}"
    elif best_fit["model"] == "quadratic":
        b = best_fit["params"]["b"]
        a = best_fit["params"]["a"]
        c = best_fit["params"]["c"]
        y_line = b * x_line**2 + a * x_line + c
        label = f"Cuadrático: y = {b:.4g} x² + {a:.4g} x + {c:.4g}"
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
    args = parser.parse_args()

    ensure_dir(args.outdir)
    df = pd.read_csv(args.input)
    if args.filter_fit_ok:
        df = safe_bool_filter(df, "fit_ok")
    if args.filter_reliable:
        df = safe_bool_filter(df, "reliable")

    galaxy_col = args.galaxy_col or pick_first_existing(df, ["galaxy", "galaxy_id", "name"], "galaxy")
    f3_col = args.f3_col or pick_first_existing(df, ["F3", "f3_scm", "f3"], "F3")

    if args.mode == "intra-galaxy":
        order_col = args.order_col or pick_first_existing(
            df, ["r_kpc", "radius_kpc", "R_kpc", "r"], "order_col (intra-galaxy)"
        )
    else:
        order_col = args.order_col or pick_first_existing(
            df, ["logMbar", "r_kpc", "radius_kpc", "bin_mass_log"], "order_col (inter-galaxy)"
        )

    df = finite_clean(df, [galaxy_col, f3_col, order_col])

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
        )

    if pairs_df.empty or len(pairs_df) < 5:
        raise ValueError(
            "Se generaron muy pocos pares de persistencia. "
            "Revisa columnas, filtros o el modo seleccionado."
        )

    x = pairs_df["delta_f3_i"].to_numpy(dtype=float)
    y = pairs_df["delta_f3_j"].to_numpy(dtype=float)
    fits = [fit_null(x, y), fit_linear(x, y), fit_quadratic(x, y)]

    model_rows = []
    for fit in fits:
        row = {
            "model": fit["model"],
            "rss": fit["rss"],
            "aicc": compute_aicc(len(x), fit["rss"], fit["k"]),
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

    pairs_df.to_csv(pairs_csv, index=False)
    models_df.to_csv(models_csv, index=False)
    make_figure(pairs_df=pairs_df, best_fit=best_fit, out_png=fig_png, title=f"Persistencia en ΔF3 ({args.mode})")

    with open(summary_txt, "w", encoding="utf-8") as fout:
        fout.write("ANÁLISIS DE PERSISTENCIA EN ΔF3\n")
        fout.write("================================\n")
        fout.write(f"Archivo de entrada: {args.input}\n")
        fout.write(f"Modo: {args.mode}\n")
        fout.write(f"Columna galaxia: {galaxy_col}\n")
        fout.write(f"Columna F3: {f3_col}\n")
        fout.write(f"Columna orden: {order_col}\n")
        fout.write(f"Número de pares: {len(pairs_df)}\n\n")
        fout.write("Comparación de modelos (ordenado por AICc):\n")
        fout.write(models_df.to_string(index=False))
        fout.write("\n\n")
        fout.write(f"Mejor modelo: {best_model_name}\n")

    print("\n=== Persistencia en ΔF3 ===")
    print(f"Archivo de entrada : {args.input}")
    print(f"Modo               : {args.mode}")
    print(f"Columna galaxia    : {galaxy_col}")
    print(f"Columna F3         : {f3_col}")
    print(f"Columna orden      : {order_col}")
    print(f"Número de pares    : {len(pairs_df)}")
    print("\nComparación de modelos:")
    print(models_df.to_string(index=False))
    print(f"\nMejor modelo       : {best_model_name}")
    print("\nSalidas:")
    print(f"- Pares            : {pairs_csv}")
    print(f"- Modelos          : {models_csv}")
    print(f"- Resumen          : {summary_txt}")
    print(f"- Figura           : {fig_png}")


if __name__ == "__main__":
    main()
