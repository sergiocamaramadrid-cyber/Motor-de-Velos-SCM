"""
test_generate_f3_catalog.py

End-to-end integration test for scripts/generate_f3_catalog_from_contract.py.
Uses a synthetic contract (galaxies.parquet + rc_points.parquet) to avoid
external data dependencies.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_contract(out_dir: Path) -> None:
    """Write minimal synthetic contract files to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two synthetic galaxies
    df_gal = pd.DataFrame({"galaxy_id": ["GAL_A", "GAL_B"]})

    # GAL_A: outer points in the deep-MOND regime (g_bar << a0 ≈ 1.2e-10 m/s²)
    # For r=30 kpc, vbar=5 km/s: g_bar = 5²/30 * 1e6/3.086e19 ≈ 2.7e-14 << a0
    rA = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0])
    vbar_A = np.array([50.0, 65.0, 75.0, 85.0, 80.0, 40.0, 10.0, 5.0])
    vrot_A = np.array([55.0, 70.0, 80.0, 90.0, 88.0, 60.0, 30.0, 20.0])
    vgas_A = np.array([15.0, 18.0, 20.0, 22.0, 18.0, 10.0, 4.0, 2.0])
    vstar_A = np.sqrt(np.maximum(vbar_A**2 - vgas_A**2, 0.0))

    # GAL_B: fewer points, all in non-deep regime (vbar large)
    rB = np.array([0.5, 1.0, 3.0, 6.0])
    vrot_B = np.array([80.0, 100.0, 120.0, 125.0])
    vgas_B = np.array([20.0, 25.0, 30.0, 28.0])
    vstar_B = np.array([70.0, 85.0, 100.0, 105.0])

    rows_A = pd.DataFrame({
        "galaxy_id": "GAL_A",
        "r_kpc": rA,
        "vrot_kms": vrot_A,
        "vgas_kms": vgas_A,
        "vstar_kms": vstar_A,
    })
    rows_B = pd.DataFrame({
        "galaxy_id": "GAL_B",
        "r_kpc": rB,
        "vrot_kms": vrot_B,
        "vgas_kms": vgas_B,
        "vstar_kms": vstar_B,
    })
    df_rc = pd.concat([rows_A, rows_B], ignore_index=True)

    df_gal.to_parquet(out_dir / "galaxies.parquet", index=False)
    df_rc.to_parquet(out_dir / "rc_points.parquet", index=False)


def test_generate_f3_catalog(tmp_path: Path):
    contract_dir = tmp_path / "contract"
    out_path = tmp_path / "f3.parquet"
    _make_contract(contract_dir)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_f3_catalog_from_contract.py",
            "--data-dir", str(contract_dir),
            "--out", str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Script failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    assert out_path.exists(), "Output parquet was not created"

    df = pd.read_parquet(out_path)

    # Required columns present
    required = {"galaxy_id", "friction_slope", "friction_slope_err",
                "n_deep", "velo_inerte_flag"}
    assert required.issubset(df.columns), (
        f"Missing columns: {required - set(df.columns)}"
    )

    # Two rows, one per galaxy
    assert len(df) == 2
    assert set(df["galaxy_id"]) == {"GAL_A", "GAL_B"}

    # GAL_A should have deep-regime points
    gal_a = df[df["galaxy_id"] == "GAL_A"].iloc[0]
    assert gal_a["n_deep"] >= 2, "GAL_A should have ≥2 deep-regime points"
    assert np.isfinite(gal_a["friction_slope"]), "friction_slope should be finite for GAL_A"
    assert np.isfinite(gal_a["friction_slope_err"]), "friction_slope_err should be finite for GAL_A"

    # Column types
    assert np.issubdtype(df["n_deep"].dtype, np.integer)
    assert df["velo_inerte_flag"].dtype == bool


def test_generate_f3_catalog_vbar_column(tmp_path: Path):
    """Contract with vbar_kms column (instead of vgas+vstar) should also work."""
    contract_dir = tmp_path / "contract_vbar"
    out_path = tmp_path / "f3_vbar.parquet"
    contract_dir.mkdir(parents=True, exist_ok=True)

    df_gal = pd.DataFrame({"galaxy_id": ["GAL_VBAR"]})
    rC = np.array([1.0, 5.0, 15.0, 30.0])
    vbar_C = np.array([60.0, 70.0, 20.0, 5.0])
    vrot_C = np.array([65.0, 78.0, 38.0, 22.0])
    df_rc = pd.DataFrame({
        "galaxy_id": "GAL_VBAR",
        "r_kpc": rC,
        "vrot_kms": vrot_C,
        "vbar_kms": vbar_C,
    })
    df_gal.to_parquet(contract_dir / "galaxies.parquet", index=False)
    df_rc.to_parquet(contract_dir / "rc_points.parquet", index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_f3_catalog_from_contract.py",
            "--data-dir", str(contract_dir),
            "--out", str(out_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Script failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    df = pd.read_parquet(out_path)
    assert "friction_slope" in df.columns
    assert len(df) == 1
