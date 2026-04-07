from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}

    exact_x = cols.get("delta_mass_std")
    exact_y = cols.get("slope_tail")
    if exact_x and exact_y:
        return exact_x, exact_y

    x_candidates = [
        c for c in df.columns
        if "delta" in c.lower() and "mass" in c.lower()
    ]
    y_candidates = [
        c for c in df.columns
        if "slope" in c.lower() and "tail" in c.lower()
    ]

    if x_candidates and y_candidates:
        return x_candidates[0], y_candidates[0]

    return None, None


def try_load(path: Path):
    if path.suffix.lower() != ".csv":
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if len(df) < 2:
        return None

    return df


def generate_figure(df: pd.DataFrame, outpath: Path):
    xcol, ycol = find_columns(df)
    if not xcol or not ycol:
        raise ValueError("required columns not found")

    plt.figure(figsize=(6, 5))
    plt.scatter(df[xcol], df[ycol])
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title("F3 vs environment")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    data_path = Path("data/scm_final_dataset_79.csv")
    outpath = Path("results/figure_env_correlation.pdf")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    df = try_load(data_path)
    if df is None:
        raise FileNotFoundError(f"could not load valid csv: {data_path}")

    generate_figure(df, outpath)


if __name__ == "__main__":
    main()
