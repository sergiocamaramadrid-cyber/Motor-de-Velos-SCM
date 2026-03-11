from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


FEATURE_COLUMNS = ["logSigmaHI_out", "logMbar", "logRd"]
TARGET_COLUMN = "F3"
DEFAULT_INPUT = "sparc_175_master.csv"
RAW_SPARC_METADATA_FILES = {"SPARC_Lelli2016c.csv", "SPARC_Lelli2016c.mrt"}


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
                return _fallback_master_or_synthetic(csv_path)
        return df

    if csv_path.name != DEFAULT_INPUT:
        if csv_path.name in RAW_SPARC_METADATA_FILES:
            return _fallback_master_or_synthetic(csv_path)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajusta una regresión lineal para F3 usando columnas del master SPARC."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Ruta al CSV de entrada (default: {DEFAULT_INPUT}).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model = fit_f3_model(Path(args.input))
    print("coeficientes:", model.coef_)
    print("intercepto:", model.intercept_)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
