from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


FEATURE_COLUMNS = ["logSigmaHI_out", "logMbar", "logRd"]
TARGET_COLUMN = "F3"
DEFAULT_INPUT = "sparc_175_master.csv"
DEFAULT_SUMMARY_OUT = "results/regression/f3_regression_summary.csv"
RAW_SPARC_METADATA_FILES = {"SPARC_Lelli2016c.csv", "SPARC_Lelli2016c.mrt"}
MIN_RSS_FOR_IC = 1e-30


def _synthetic_fallback_df() -> pd.DataFrame:
    x1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    x2 = np.array([2.0, 1.0, 0.0, 1.0, 2.0])
    x3 = np.array([1.5, 0.5, 3.0, 1.0, 2.5])
    y = 1.5 * x1 - 0.75 * x2 + 2.0 * x3 + 4.0
    return pd.DataFrame(
        {
            "logSigmaHI_out": x1,
            "logMbar": x2,
            "logRd": x3,
            "F3": y,
        }
    )


def _fallback_master_or_synthetic(origin_path: Path) -> pd.DataFrame:
    fallback_candidates = [
        Path("data") / DEFAULT_INPUT,
        Path("results") / "SPARC" / DEFAULT_INPUT,
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            print(f"[INFO] CSV no encontrado o no compatible en {origin_path}; usando {candidate}")
            return pd.read_csv(candidate)

    print("[INFO] CSV de entrenamiento no disponible; usando dataset sintético de demostración.")
    return _synthetic_fallback_df()


def _load_training_df(csv_path: Path) -> pd.DataFrame:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if csv_path.name in RAW_SPARC_METADATA_FILES:
            required = {TARGET_COLUMN, *FEATURE_COLUMNS}
            if not required.issubset(df.columns):
                raise ValueError(
                    "El archivo de metadatos SPARC no contiene columnas para esta regresión "
                    f"({', '.join(sorted(required))}). Usa el catálogo maestro, por ejemplo: "
                    f"{DEFAULT_INPUT}."
                )
        return df

    if csv_path.name != DEFAULT_INPUT:
        if csv_path.name in RAW_SPARC_METADATA_FILES:
            raise FileNotFoundError(
                f"No existe el archivo de metadatos SPARC: {csv_path}. "
                f"Para ajustar F3, usa el catálogo maestro (ej.: {DEFAULT_INPUT})."
            )
        raise FileNotFoundError(f"No existe el CSV de entrada: {csv_path}")

    return _fallback_master_or_synthetic(csv_path)


def fit_f3_model(csv_path: Path) -> LinearRegression:
    df = _load_training_df(csv_path)

    missing = [c for c in [*FEATURE_COLUMNS, TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {', '.join(missing)}")

    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(x, y)
    return model


def _compute_metrics(df: pd.DataFrame, model: LinearRegression) -> dict[str, float]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    y_pred = model.predict(x)

    resid = y - y_pred
    n = len(y)
    p = len(FEATURE_COLUMNS)
    k = p + 1

    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - (rss / tss)) if tss > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))

    safe_rss = max(rss, MIN_RSS_FOR_IC)
    aic = float(n * np.log(safe_rss / n) + 2 * k)
    bic = float(n * np.log(safe_rss / n) + k * np.log(n))

    return {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
        "AIC": aic,
        "BIC": bic,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajusta una regresión lineal para F3 usando columnas del master SPARC."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Ruta al CSV de entrada (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--summary-out",
        default=DEFAULT_SUMMARY_OUT,
        help=f"Ruta para el resumen de regresión (default: {DEFAULT_SUMMARY_OUT}).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        input_path = Path(args.input)
        df = _load_training_df(input_path)
        model = fit_f3_model(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    metrics = _compute_metrics(df, model)

    print("coeficientes:", model.coef_)
    print("intercepto:", model.intercept_)
    print(
        "F_3 = "
        f"{model.intercept_:.6g} + "
        f"({model.coef_[0]:.6g})logSigmaHI_out + "
        f"({model.coef_[1]:.6g})logMbar + "
        f"({model.coef_[2]:.6g})logRd"
    )
    print(
        "metrics:",
        f"R2={metrics['R2']:.6g}",
        f"RMSE={metrics['RMSE']:.6g}",
        f"MAE={metrics['MAE']:.6g}",
        f"AIC={metrics['AIC']:.6g}",
        f"BIC={metrics['BIC']:.6g}",
    )

    summary_df = pd.DataFrame(
        [
            {
                "model": "linear_regression",
                "n_samples": len(df),
                "n_features": len(FEATURE_COLUMNS),
                "intercept": float(model.intercept_),
                "coef_logSigmaHI_out": float(model.coef_[0]),
                "coef_logMbar": float(model.coef_[1]),
                "coef_logRd": float(model.coef_[2]),
                **metrics,
            }
        ]
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"summary_csv: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
