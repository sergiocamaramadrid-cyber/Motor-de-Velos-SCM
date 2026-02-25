"""
tests/test_pipeline.py — "Notarial" integration test for run_pipeline().

Validates three invariants that must hold on every run:
  (i)  Exact column contract of the returned DataFrame.
  (ii) Row count ≥ 150 (SPARC-scale dataset).
  (iii) per_galaxy_summary.csv is written with the same column contract.

Uses a 175-galaxy synthetic dataset so the test runs without a real
SPARC download and CI does not need external data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scm_analysis import run_pipeline

# Canonical column order expected from run_pipeline()
EXPECTED_COLS = [
    "galaxy",
    "upsilon_disk",
    "chi2_reduced",
    "n_points",
    "Vflat_kms",
    "M_bar_BTFR_Msun",
]

N_GALAXIES = 175  # matches full SPARC sample size


@pytest.fixture(scope="module")
def sparc175_dir(tmp_path_factory):
    """Synthetic 175-galaxy SPARC-like dataset.

    Each galaxy gets a 20-point rotation curve with a flat profile plus
    small Gaussian noise so the fitter converges quickly.
    """
    root = tmp_path_factory.mktemp("sparc175")
    rng = np.random.default_rng(0)

    galaxy_names = [f"G{i:04d}" for i in range(N_GALAXIES)]
    v_flats = np.linspace(80.0, 320.0, N_GALAXIES)

    galaxy_table = pd.DataFrame(
        {
            "Galaxy": galaxy_names,
            "D": np.linspace(3.0, 80.0, N_GALAXIES),
            "Inc": np.linspace(25.0, 85.0, N_GALAXIES),
            "L36": 1e9 * np.arange(1, N_GALAXIES + 1, dtype=float),
            "Vflat": v_flats,
            "e_Vflat": np.full(N_GALAXIES, 5.0),
        }
    )
    galaxy_table.to_csv(root / "SPARC_Lelli2016c.csv", index=False)

    n_pts = 20
    for name, vf in zip(galaxy_names, v_flats):
        r = np.linspace(0.5, 15.0, n_pts)
        rc = pd.DataFrame(
            {
                "r": r,
                "v_obs": np.full(n_pts, vf) + rng.normal(0, 3, n_pts),
                "v_obs_err": np.full(n_pts, 5.0),
                "v_gas": 0.3 * vf * np.ones(n_pts),
                "v_disk": 0.75 * vf * np.ones(n_pts),
                "v_bul": np.zeros(n_pts),
                "SBdisk": np.zeros(n_pts),
                "SBbul": np.zeros(n_pts),
            }
        )
        rc.to_csv(root / f"{name}_rotmod.dat", sep=" ", index=False, header=False)

    return root


def test_run_pipeline_outputs(sparc175_dir, tmp_path):
    """Notarial test: columns, N, all core artefacts, and data sanity."""
    out = tmp_path / "results"
    df = run_pipeline(sparc175_dir, str(out), verbose=False)

    # (i) exact column contract
    assert df.columns.tolist() == EXPECTED_COLS, (
        f"Column mismatch: got {df.columns.tolist()}"
    )

    # (ii) row count
    assert len(df) >= 150, f"Expected ≥150 rows, got {len(df)}"

    # (iii) all five core artefacts must exist
    per_gal = out / "per_galaxy_summary.csv"
    uni = out / "universal_term_comparison_full.csv"
    summary = out / "executive_summary.txt"
    top10 = out / "top10_universal.tex"
    sparc_global = out / "sparc_global.csv"

    assert per_gal.exists(), "per_galaxy_summary.csv not written"
    assert uni.exists(), "universal_term_comparison_full.csv not written"
    assert summary.exists(), "executive_summary.txt not written"
    assert top10.exists(), "top10_universal.tex not written"
    assert sparc_global.exists(), "sparc_global.csv not written"

    # (iv) per_galaxy_summary contract and data sanity
    df2 = pd.read_csv(per_gal)
    assert df2.columns.tolist() == EXPECTED_COLS, (
        f"per_galaxy_summary.csv column mismatch: got {df2.columns.tolist()}"
    )
    assert df2["galaxy"].notna().all(), "galaxy column contains NaN values"

    # (v) sparc_global contract: single row with expected columns
    sg = pd.read_csv(sparc_global)
    assert len(sg) == 1, f"sparc_global.csv should have 1 row, got {len(sg)}"
    expected_global_cols = [
        "n_galaxies", "n_radial_points",
        "chi2_reduced_median", "chi2_reduced_mean",
        "upsilon_disk_median", "upsilon_disk_mean",
        "Vflat_kms_median",
        "log_g_bar_median", "log_g_obs_median",
    ]
    for col in expected_global_cols:
        assert col in sg.columns, f"sparc_global.csv missing column: {col}"
    assert int(sg["n_galaxies"].iloc[0]) >= 150, "n_galaxies should be ≥ 150"


def test_run_pipeline_sorted(sparc175_dir, tmp_path):
    """Returned DataFrame must be sorted by galaxy name."""
    out = tmp_path / "results_sorted"
    df = run_pipeline(sparc175_dir, str(out), verbose=False)
    assert list(df["galaxy"]) == sorted(df["galaxy"].tolist())


def test_run_pipeline_types(sparc175_dir, tmp_path):
    """n_points must be int; numeric columns must be float."""
    out = tmp_path / "results_types"
    df = run_pipeline(sparc175_dir, str(out), verbose=False)
    assert pd.api.types.is_integer_dtype(df["n_points"]), (
        f"n_points dtype should be integer, got {df['n_points'].dtype}"
    )
    for col in ("upsilon_disk", "chi2_reduced", "Vflat_kms", "M_bar_BTFR_Msun"):
        assert pd.api.types.is_float_dtype(df[col]), f"{col} is not float"
