"""
tests/test_mw_radial_scan.py — Tests for scripts/mw_radial_scan.py.

Creates synthetic MW Cepheid-like data to verify:
- compute_radial_spearman returns correct dict keys
- scan_radii returns correctly-structured DataFrame
- main() end-to-end output
- rho and pval ranges
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.mw_radial_scan import (
    R_SCAN_MIN_DEFAULT,
    R_SCAN_MAX_DEFAULT,
    R_SCAN_STEP_DEFAULT,
    R_MIN_N_DEFAULT,
    compute_radial_spearman,
    scan_radii,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mw_df(n: int = 500, seed: int = 55) -> pd.DataFrame:
    """Synthetic MW Cepheid dataframe."""
    rng = np.random.default_rng(seed)
    R_kpc   = rng.uniform(2, 25, n)
    Vc_kms  = 230 - 5 * (R_kpc - 8) / 8 + rng.normal(0, 5, n)
    lon_deg = rng.uniform(0, 360, n)
    lat_deg = rng.normal(0, 15, n)
    return pd.DataFrame({
        "R_kpc":   R_kpc,
        "Vc_kms":  Vc_kms,
        "e_Vc":    rng.uniform(5, 15, n),
        "lon_deg": lon_deg,
        "lat_deg": lat_deg,
        "source":  "synthetic_gaia",
    })


def _make_csv(tmp_path: Path, df: pd.DataFrame, name: str = "mw_cepheids.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# compute_radial_spearman
# ---------------------------------------------------------------------------

class TestComputeRadialSpearman:
    def test_returns_dict(self):
        df = _make_mw_df()
        result = compute_radial_spearman(df, r_cut=5.0)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = _make_mw_df()
        result = compute_radial_spearman(df, r_cut=5.0)
        for key in ["r_cut_kpc", "n", "rho", "pval"]:
            assert key in result

    def test_r_cut_stored_correctly(self):
        df = _make_mw_df()
        result = compute_radial_spearman(df, r_cut=8.0)
        assert result["r_cut_kpc"] == pytest.approx(8.0)

    def test_n_matches_filter(self):
        df = _make_mw_df(n=300)
        r_cut = 10.0
        expected_n = int((df["R_kpc"] >= r_cut).sum())
        result = compute_radial_spearman(df, r_cut=r_cut)
        assert result["n"] == expected_n

    def test_rho_in_valid_range(self):
        df = _make_mw_df()
        result = compute_radial_spearman(df, r_cut=5.0)
        assert -1 <= result["rho"] <= 1

    def test_pval_in_unit_interval(self):
        df = _make_mw_df()
        result = compute_radial_spearman(df, r_cut=5.0)
        assert 0 <= result["pval"] <= 1

    def test_custom_columns(self):
        df = _make_mw_df()
        df = df.rename(columns={"lon_deg": "l_deg", "Vc_kms": "V_kms"})
        result = compute_radial_spearman(df, r_cut=5.0, lon_col="l_deg", vc_col="V_kms")
        assert "rho" in result

    def test_very_high_r_cut_returns_nan_for_small_n(self):
        df = _make_mw_df(n=10, seed=1)
        df["R_kpc"] = 3.0  # all stars at R=3, none above r_cut=25
        result = compute_radial_spearman(df, r_cut=30.0)
        assert result["n"] == 0
        import math
        assert math.isnan(result["rho"])


# ---------------------------------------------------------------------------
# scan_radii
# ---------------------------------------------------------------------------

class TestScanRadii:
    def test_returns_dataframe(self):
        df = _make_mw_df()
        result = scan_radii(df, r_min=5.0, r_max=15.0, r_step=1.0, min_n=10)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self):
        df = _make_mw_df()
        result = scan_radii(df, r_min=5.0, r_max=15.0, r_step=1.0, min_n=10)
        for col in ["r_cut_kpc", "n", "rho", "pval"]:
            assert col in result.columns

    def test_r_cut_monotone_nondecreasing(self):
        df = _make_mw_df()
        result = scan_radii(df, r_min=5.0, r_max=20.0, r_step=1.0, min_n=10)
        vals = result["r_cut_kpc"].values
        assert all(vals[i] <= vals[i+1] for i in range(len(vals)-1))

    def test_all_rho_in_valid_range(self):
        df = _make_mw_df()
        result = scan_radii(df, r_min=5.0, r_max=15.0, r_step=1.0, min_n=10)
        assert (result["rho"] >= -1).all() and (result["rho"] <= 1).all()

    def test_all_pval_in_unit_interval(self):
        df = _make_mw_df()
        result = scan_radii(df, r_min=5.0, r_max=15.0, r_step=1.0, min_n=10)
        assert (result["pval"] >= 0).all() and (result["pval"] <= 1).all()

    def test_min_n_filter_respected(self):
        df = _make_mw_df()
        min_n = 50
        result = scan_radii(df, r_min=5.0, r_max=25.0, r_step=1.0, min_n=min_n)
        assert (result["n"] >= min_n).all()

    def test_nonempty_for_valid_range(self):
        df = _make_mw_df(n=500)
        result = scan_radii(df, r_min=5.0, r_max=15.0, r_step=2.0, min_n=5)
        assert len(result) > 0

    def test_r_cut_values_within_bounds(self):
        df = _make_mw_df()
        r_min, r_max = 6.0, 14.0
        result = scan_radii(df, r_min=r_min, r_max=r_max, r_step=2.0, min_n=5)
        assert (result["r_cut_kpc"] >= r_min).all()
        assert (result["r_cut_kpc"] <= r_max + 1e-9).all()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMWRadialScanMain:
    def test_creates_output_csv(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path)])
        assert (tmp_path / "mw_radial_scan.csv").exists()

    def test_returns_dict(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        assert isinstance(result, dict)

    def test_returns_required_keys(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        for key in ["scan_df", "r_scan_min", "r_scan_max", "out_path"]:
            assert key in result

    def test_scan_df_is_dataframe(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        assert isinstance(result["scan_df"], pd.DataFrame)

    def test_output_csv_has_correct_columns(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path)])
        out = pd.read_csv(tmp_path / "mw_radial_scan.csv")
        for col in ["r_cut_kpc", "n", "rho", "pval"]:
            assert col in out.columns

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "no_file.csv"), "--out", str(tmp_path)])

    def test_r_scan_min_returned(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--r-min", "6.0"])
        assert result["r_scan_min"] == pytest.approx(6.0)

    def test_r_scan_max_returned(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--r-max", "18.0"])
        assert result["r_scan_max"] == pytest.approx(18.0)
