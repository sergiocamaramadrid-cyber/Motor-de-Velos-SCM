from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression


FEATURE_COLUMNS = ["logSigmaHI_out", "logMbar", "logRd"]
TARGET_COLUMN = "F3"


def fit_f3_model(csv_path: Path) -> LinearRegression:
    df = pd.read_csv(csv_path)

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
        default="sparc_175_master.csv",
        help="Ruta al CSV de entrada (default: sparc_175_master.csv).",
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
