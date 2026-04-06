"""Diagnostic script: inspect results/env_real/sparc_f3_chae_merged.csv."""

import pandas as pd

CSV_PATH = "results/env_real/sparc_f3_chae_merged.csv"
OUT_PATH = "debug_output.txt"


def main():
    df = pd.read_csv(CSV_PATH)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"SHAPE: {df.shape}\n\n")
        f.write(f"COLUMNS: {df.columns.tolist()}\n\n")
        f.write(f"NANS:\n{df.isna().sum()}\n\n")
        f.write(f"HEAD:\n{df.head(5).to_string(index=False)}\n")

    print(f"Archivo generado: {OUT_PATH}")


if __name__ == "__main__":
    main()
