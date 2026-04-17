"""
tests/test_build_galaxy_signal_table.py — Tests for build_galaxy_signal_table.py
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_galaxy_signal_table import (
    BETA_REF,
    HE_CORRECTION,
    MIN_OUTER_POINTS,
    OUTER_FRAC,
    OUTPUT_COLUMNS,
    UPSILON_DEFAULT,
    _KPC_TO_M,
    build_signal_table,
    compute_rotation_stats,
    load_sparc_properties,
    main,
)

_REPO_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Synthetic SPARC fixture builders
# ---------------------------------------------------------------------------


def _make_sparc_dir(
    tmp_path: Path,
    n_gal: int = 4,
    v_flat: float = 2.0,
    n_pts: int = 30,
    include_mhi: bool = True,
    include_rdisk: bool = True,
    seed: int = 0,
) -> Path:
    """Create a minimal synthetic SPARC directory for testing.

    The rotation curves are approximately flat so that deep-regime points
    exist and slope_tail is computable.
    """
    rng = np.random.default_rng(seed)
    names = [f"G{i:02d}" for i in range(n_gal)]

    galaxy_row: dict = {
        "Galaxy": names,
        "D":      np.linspace(5.0, 25.0, n_gal),
        "Inc":    np.linspace(40.0, 70.0, n_gal),
        "L36":    np.linspace(1.0, 5.0, n_gal),       # 1e9 Lsun
        "Vflat":  np.full(n_gal, v_flat),
        "e_Vflat": np.full(n_gal, 0.1),
    }
    if include_mhi:
        galaxy_row["MHI"] = np.linspace(0.2, 1.0, n_gal)  # 1e9 Msun
    if include_rdisk:
        galaxy_row["Rdisk"] = np.linspace(1.5, 4.0, n_gal)  # kpc

    pd.DataFrame(galaxy_row).to_csv(
        tmp_path / "SPARC_Lelli2016c.csv", index=False
    )

    # Rotation curves: small velocities → deep regime exists at large r
    r = np.linspace(0.2, 15.0, n_pts)
    for name in names:
        v_obs = np.full(n_pts, v_flat) + rng.normal(0, 0.02, n_pts)
        rc = pd.DataFrame({
            "r":         r,
            "v_obs":     np.clip(v_obs, 0.01, None),
            "v_obs_err": np.full(n_pts, 0.05),
            "v_gas":     0.3 * v_flat * np.ones(n_pts),
            "v_disk":    0.7 * v_flat * np.ones(n_pts),
            "v_bul":     np.zeros(n_pts),
            "SBdisk":    np.zeros(n_pts),
            "SBbul":     np.zeros(n_pts),
        })
        rc.to_csv(tmp_path / f"{name}_rotmod.dat", sep=" ", index=False,
                  header=False)

    return tmp_path


def _make_env_csv(tmp_path: Path, galaxies: list[str]) -> Path:
    """Create a minimal env_proxy CSV for *galaxies*."""
    df = pd.DataFrame({
        "galaxy": galaxies,
        "env_proxy": np.linspace(0.1, 1.0, len(galaxies)),
    })
    p = tmp_path / "env_proxy.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. load_sparc_properties
# ---------------------------------------------------------------------------


class TestLoadSparcProperties:
    def test_returns_dataframe(self, tmp_path):
        _make_sparc_dir(tmp_path)
        df = load_sparc_properties(tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, tmp_path):
        _make_sparc_dir(tmp_path)
        df = load_sparc_properties(tmp_path)
        assert "galaxy" in df.columns
        assert "logMbar" in df.columns
        assert "Mgas" in df.columns
        assert "Rdisk" in df.columns

    def test_galaxy_count_matches_table(self, tmp_path):
        n_gal = 6
        _make_sparc_dir(tmp_path, n_gal=n_gal)
        df = load_sparc_properties(tmp_path)
        assert len(df) == n_gal

    def test_mgas_computed_from_mhi(self, tmp_path):
        _make_sparc_dir(tmp_path, n_gal=3, include_mhi=True)
        df = load_sparc_properties(tmp_path)
        assert df["Mgas"].notna().all()
        # Mgas = 1.33 × MHI; all values must be positive
        assert (df["Mgas"] > 0).all()

    def test_mgas_nan_when_mhi_absent(self, tmp_path):
        _make_sparc_dir(tmp_path, n_gal=3, include_mhi=False)
        df = load_sparc_properties(tmp_path)
        assert df["Mgas"].isna().all()

    def test_rdisk_present(self, tmp_path):
        _make_sparc_dir(tmp_path, n_gal=3, include_rdisk=True)
        df = load_sparc_properties(tmp_path)
        assert df["Rdisk"].notna().all()

    def test_rdisk_nan_when_absent(self, tmp_path):
        _make_sparc_dir(tmp_path, n_gal=3, include_rdisk=False)
        df = load_sparc_properties(tmp_path)
        assert df["Rdisk"].isna().all()

    def test_logMbar_finite_with_full_columns(self, tmp_path):
        _make_sparc_dir(tmp_path, n_gal=4)
        df = load_sparc_properties(tmp_path)
        assert df["logMbar"].notna().all()

    def test_logMbar_reasonable_range(self, tmp_path):
        _make_sparc_dir(tmp_path, n_gal=4)
        df = load_sparc_properties(tmp_path)
        # 1e9 Msun range: log10(1e9) = 9
        assert (df["logMbar"] > 7.0).all()
        assert (df["logMbar"] < 14.0).all()

    def test_logMbar_nan_when_no_l36(self, tmp_path):
        # Create galaxy table without L36 or MHI
        names = ["A", "B"]
        pd.DataFrame({
            "Galaxy": names,
            "D": [10.0, 20.0],
            "Inc": [45.0, 60.0],
            "Vflat": [2.0, 2.0],
            "e_Vflat": [0.1, 0.1],
        }).to_csv(tmp_path / "SPARC_Lelli2016c.csv", index=False)
        df = load_sparc_properties(tmp_path)
        assert df["logMbar"].isna().all()

    def test_missing_galaxy_table_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="SPARC galaxy table"):
            load_sparc_properties(tmp_path / "nonexistent")

    def test_galaxy_names_are_strings(self, tmp_path):
        _make_sparc_dir(tmp_path, n_gal=3)
        df = load_sparc_properties(tmp_path)
        # dtype may be 'object' or pandas StringDtype depending on pandas version
        assert df["galaxy"].dtype in (object,) or pd.api.types.is_string_dtype(df["galaxy"])

    def test_he_correction_applied(self, tmp_path):
        """Mgas / MHI ≈ HE_CORRECTION for all galaxies."""
        n_gal = 3
        mhi_values = [0.5, 1.0, 2.0]
        names = [f"G{i}" for i in range(n_gal)]
        pd.DataFrame({
            "Galaxy": names,
            "D": [10.0] * n_gal,
            "Inc": [45.0] * n_gal,
            "L36": [1.0] * n_gal,
            "Vflat": [2.0] * n_gal,
            "e_Vflat": [0.1] * n_gal,
            "MHI": mhi_values,
        }).to_csv(tmp_path / "SPARC_Lelli2016c.csv", index=False)
        df = load_sparc_properties(tmp_path)
        expected = pd.Series(mhi_values) * HE_CORRECTION
        np.testing.assert_allclose(df["Mgas"].values, expected.values, rtol=1e-9)


# ---------------------------------------------------------------------------
# 2. compute_rotation_stats
# ---------------------------------------------------------------------------


def _flat_rc(v_flat: float = 2.0, n_pts: int = 40, r_max: float = 20.0) -> pd.DataFrame:
    """Create a perfectly flat rotation curve."""
    r = np.linspace(0.2, r_max, n_pts)
    return pd.DataFrame({
        "r":       r,
        "v_obs":   np.full(n_pts, v_flat),
        "v_obs_err": np.full(n_pts, 0.05),
        "v_gas":   0.3 * v_flat * np.ones(n_pts),
        "v_disk":  0.7 * v_flat * np.ones(n_pts),
        "v_bul":   np.zeros(n_pts),
    })


class TestComputeRotationStats:
    def test_returns_dict_with_correct_keys(self):
        rc = _flat_rc()
        result = compute_rotation_stats(rc)
        assert set(result.keys()) == {
            "Rmax", "Vmax", "slope_tail", "outer_fit_ok", "n_tail_points"
        }

    def test_rmax_is_max_radius(self):
        rc = _flat_rc(r_max=18.0, n_pts=20)
        result = compute_rotation_stats(rc)
        assert math.isclose(result["Rmax"], 18.0, rel_tol=1e-6)

    def test_vmax_is_max_velocity(self):
        rc = _flat_rc(v_flat=3.0)
        result = compute_rotation_stats(rc)
        assert math.isclose(result["Vmax"], 3.0, rel_tol=1e-4)

    def test_slope_tail_finite_for_flat_curve(self):
        rc = _flat_rc()
        result = compute_rotation_stats(rc)
        # flat curve → g_obs ∝ 1/r, g_bar ∝ 1/r → slope ≈ 1.0
        assert math.isfinite(result["slope_tail"])

    def test_slope_tail_near_one_for_flat_curve(self):
        """For a flat rotation curve, β ≈ 1 in the deep regime."""
        rc = _flat_rc(v_flat=1.5, n_pts=60)
        result = compute_rotation_stats(rc)
        assert math.isfinite(result["slope_tail"]), "slope_tail should be finite"
        # g_obs ∝ V²/r, g_bar ∝ V_bar²/r; log-log slope ≈ 1
        assert abs(result["slope_tail"] - 1.0) < 0.3, (
            f"Expected slope_tail ≈ 1, got {result['slope_tail']:.3f}"
        )

    def test_slope_tail_nan_when_too_few_outer_points(self):
        """With only 5 total radial points, at most 2 satisfy r >= 0.7*Rmax,
        which is below MIN_OUTER_POINTS → slope_tail must be NaN."""
        r = np.linspace(1.0, 10.0, 5)
        rc = pd.DataFrame({
            "r":       r,
            "v_obs":   np.full(5, 300.0),   # very high V → g_obs >> a0
            "v_obs_err": np.full(5, 5.0),
            "v_gas":   np.full(5, 100.0),
            "v_disk":  np.full(5, 200.0),
            "v_bul":   np.zeros(5),
        })
        result = compute_rotation_stats(rc)
        assert math.isnan(result["slope_tail"])

    def test_rmax_vmax_types(self):
        rc = _flat_rc()
        result = compute_rotation_stats(rc)
        assert isinstance(result["Rmax"], float)
        assert isinstance(result["Vmax"], float)

    def test_empty_rc_returns_nans(self):
        rc = pd.DataFrame(columns=["r", "v_obs", "v_obs_err",
                                   "v_gas", "v_disk", "v_bul"])
        result = compute_rotation_stats(rc)
        assert math.isnan(result["Rmax"])
        assert math.isnan(result["Vmax"])
        assert math.isnan(result["slope_tail"])

    def test_missing_component_columns_uses_zeros(self):
        """If v_gas/v_disk/v_bul absent, zeros are assumed gracefully."""
        r = np.linspace(0.1, 10.0, 30)
        rc = pd.DataFrame({
            "r":       r,
            "v_obs":   np.full(30, 1.5),
            "v_obs_err": np.full(30, 0.05),
        })
        result = compute_rotation_stats(rc)
        assert isinstance(result["Rmax"], float)
        assert isinstance(result["Vmax"], float)

    def test_outer_fit_ok_true_when_enough_points(self):
        """A curve with many outer points should yield outer_fit_ok=True."""
        rc = _flat_rc(n_pts=40)
        result = compute_rotation_stats(rc)
        assert result["outer_fit_ok"] is True
        assert math.isfinite(result["slope_tail"])

    def test_outer_fit_ok_false_when_too_few_points(self):
        """Fewer than MIN_OUTER_POINTS outer points → outer_fit_ok=False."""
        r = np.linspace(1.0, 10.0, 5)  # 0.7*Rmax=7.0; outer: 2 pts < 4
        rc = pd.DataFrame({
            "r":       r,
            "v_obs":   np.full(5, 2.0),
            "v_obs_err": np.full(5, 0.05),
            "v_gas":   0.5 * np.ones(5),
            "v_disk":  1.5 * np.ones(5),
            "v_bul":   np.zeros(5),
        })
        result = compute_rotation_stats(rc)
        assert result["outer_fit_ok"] is False
        assert math.isnan(result["slope_tail"])

    def test_n_tail_points_correct(self):
        """n_tail_points equals number of valid points with r >= 0.7*Rmax."""
        r = np.linspace(1.0, 10.0, 20)
        rmax = r[-1]
        expected_outer = int((r >= OUTER_FRAC * rmax).sum())
        rc = pd.DataFrame({
            "r":       r,
            "v_obs":   np.full(20, 2.0),
            "v_obs_err": np.full(20, 0.05),
            "v_gas":   0.5 * np.ones(20),
            "v_disk":  1.5 * np.ones(20),
            "v_bul":   np.zeros(20),
        })
        result = compute_rotation_stats(rc)
        assert result["n_tail_points"] == expected_outer

    def test_outer_regime_uses_0_7_rmax(self):
        """Outer regime is exactly r >= 0.7 * Rmax (OUTER_FRAC = 0.7)."""
        assert OUTER_FRAC == 0.7
        r = np.linspace(1.0, 10.0, 30)
        rmax = r[-1]
        threshold = OUTER_FRAC * rmax
        rc = pd.DataFrame({
            "r":       r,
            "v_obs":   np.full(30, 2.0),
            "v_obs_err": np.full(30, 0.05),
            "v_gas":   0.5 * np.ones(30),
            "v_disk":  1.5 * np.ones(30),
            "v_bul":   np.zeros(30),
        })
        result = compute_rotation_stats(rc)
        expected_n = int((r >= threshold).sum())
        assert result["n_tail_points"] == expected_n

    def test_n_tail_points_is_int(self):
        rc = _flat_rc()
        result = compute_rotation_stats(rc)
        assert isinstance(result["n_tail_points"], int)

    def test_upsilon_disk_default_is_1(self):
        """compute_rotation_stats uses upsilon_disk=1.0 by default."""
        rc = _flat_rc()
        r1 = compute_rotation_stats(rc, upsilon_disk=1.0)
        r2 = compute_rotation_stats(rc)
        assert r1["slope_tail"] == r2["slope_tail"]

    def test_upsilon_bulge_default_is_1(self):
        """compute_rotation_stats uses upsilon_bulge=1.0 by default."""
        rc = _flat_rc()
        r1 = compute_rotation_stats(rc, upsilon_bulge=1.0)
        r2 = compute_rotation_stats(rc)
        assert r1["slope_tail"] == r2["slope_tail"]


# ---------------------------------------------------------------------------
# 3. build_signal_table
# ---------------------------------------------------------------------------


class TestBuildSignalTable:
    def test_returns_dataframe(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        assert isinstance(df, pd.DataFrame)

    def test_output_file_created(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        build_signal_table(tmp_path, out)
        assert out.exists()

    def test_output_columns_match_spec(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        assert list(df.columns) == OUTPUT_COLUMNS

    def test_galaxy_count_equals_rotmod_files(self, tmp_path):
        n_gal = 5
        _make_sparc_dir(tmp_path, n_gal=n_gal)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        assert len(df) == n_gal

    def test_rmax_vmax_positive(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        assert (df["Rmax"] > 0).all()
        assert (df["Vmax"] > 0).all()

    def test_delta_f3_is_slope_minus_beta_ref(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        valid = df["slope_tail"].notna()
        np.testing.assert_allclose(
            df.loc[valid, "delta_f3"].values,
            df.loc[valid, "slope_tail"].values - BETA_REF,
            rtol=1e-9,
        )

    def test_width_kpc_is_2_5_times_rdisk(self, tmp_path):
        _make_sparc_dir(tmp_path, include_rdisk=True)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        props = load_sparc_properties(tmp_path)
        props_indexed = props.set_index("galaxy")
        for _, row in df.iterrows():
            gal = row["galaxy"]
            rdisk = props_indexed.loc[gal, "Rdisk"]
            if not np.isnan(rdisk):
                assert math.isclose(row["width_kpc"], 2.5 * rdisk, rel_tol=1e-9)

    def test_thickness_kpc_is_0_1_times_rdisk(self, tmp_path):
        _make_sparc_dir(tmp_path, include_rdisk=True)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        props = load_sparc_properties(tmp_path)
        props_indexed = props.set_index("galaxy")
        for _, row in df.iterrows():
            gal = row["galaxy"]
            rdisk = props_indexed.loc[gal, "Rdisk"]
            if not np.isnan(rdisk):
                assert math.isclose(
                    row["thickness_kpc"], 0.1 * rdisk, rel_tol=1e-9
                )

    def test_width_thickness_nan_when_rdisk_absent(self, tmp_path):
        _make_sparc_dir(tmp_path, include_rdisk=False)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        assert df["width_kpc"].isna().all()
        assert df["thickness_kpc"].isna().all()

    def test_env_proxy_nan_by_default(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        assert df["env_proxy"].isna().all()

    def test_env_proxy_merged_from_csv(self, tmp_path):
        _make_sparc_dir(tmp_path, n_gal=4)
        galaxies = [f"G{i:02d}" for i in range(4)]
        env_csv = _make_env_csv(tmp_path, galaxies)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out, env_csv=env_csv)
        assert df["env_proxy"].notna().all()

    def test_env_proxy_values_match_csv(self, tmp_path):
        n_gal = 4
        _make_sparc_dir(tmp_path, n_gal=n_gal)
        galaxies = [f"G{i:02d}" for i in range(n_gal)]
        env_values = [0.1, 0.4, 0.7, 1.0]
        env_df = pd.DataFrame({"galaxy": galaxies, "env_proxy": env_values})
        env_csv = tmp_path / "env.csv"
        env_df.to_csv(env_csv, index=False)

        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out, env_csv=env_csv)
        env_lookup = dict(zip(env_df["galaxy"], env_df["env_proxy"]))
        for _, row in df.iterrows():
            expected = env_lookup.get(row["galaxy"], np.nan)
            assert math.isclose(row["env_proxy"], expected, rel_tol=1e-9)

    def test_env_csv_missing_columns_raises(self, tmp_path):
        _make_sparc_dir(tmp_path)
        bad_env = tmp_path / "bad_env.csv"
        pd.DataFrame({"galaxy": ["G00"], "wrong_col": [1.0]}).to_csv(bad_env)
        out = tmp_path / "signal.csv"
        with pytest.raises(ValueError, match="env_proxy"):
            build_signal_table(tmp_path, out, env_csv=bad_env)

    def test_output_sorted_by_galaxy(self, tmp_path):
        _make_sparc_dir(tmp_path, n_gal=5)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        assert list(df["galaxy"]) == sorted(df["galaxy"].tolist())

    def test_output_csv_readable(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        build_signal_table(tmp_path, out)
        reloaded = pd.read_csv(out)
        assert list(reloaded.columns) == OUTPUT_COLUMNS

    def test_logMbar_in_reasonable_range(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        valid = df["logMbar"].notna()
        assert valid.any()
        assert (df.loc[valid, "logMbar"] > 7.0).all()
        assert (df.loc[valid, "logMbar"] < 14.0).all()

    def test_mgas_positive_when_available(self, tmp_path):
        _make_sparc_dir(tmp_path, include_mhi=True)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        valid = df["Mgas"].notna()
        assert (df.loc[valid, "Mgas"] > 0).all()

    def test_mgas_nan_without_mhi(self, tmp_path):
        _make_sparc_dir(tmp_path, include_mhi=False)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out)
        assert df["Mgas"].isna().all()

    def test_custom_beta_ref_affects_delta_f3(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        beta_ref_custom = 0.7
        df = build_signal_table(tmp_path, out, beta_ref=beta_ref_custom)
        valid = df["slope_tail"].notna()
        np.testing.assert_allclose(
            df.loc[valid, "delta_f3"].values,
            df.loc[valid, "slope_tail"].values - beta_ref_custom,
            rtol=1e-9,
        )

    def test_missing_galaxy_table_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_signal_table(tmp_path / "missing", tmp_path / "out.csv")

    def test_no_rotmod_files_returns_empty(self, tmp_path):
        """Galaxy table exists but no rotmod files → empty output."""
        pd.DataFrame({
            "Galaxy": ["A", "B"],
            "D": [10.0, 20.0],
            "Inc": [45.0, 60.0],
            "L36": [1.0, 2.0],
            "Vflat": [2.0, 2.0],
            "e_Vflat": [0.1, 0.1],
        }).to_csv(tmp_path / "SPARC_Lelli2016c.csv", index=False)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out, verbose=False)
        assert len(df) == 0
        assert list(df.columns) == OUTPUT_COLUMNS

    def test_output_dir_created_if_missing(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "subdir" / "nested" / "signal.csv"
        build_signal_table(tmp_path, out, verbose=False)
        assert out.exists()

    def test_verbose_false_suppresses_output(self, tmp_path, capsys):
        _make_sparc_dir(tmp_path, n_gal=2)
        out = tmp_path / "signal.csv"
        build_signal_table(tmp_path, out, verbose=False)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_slope_tail_nan_when_no_deep_points(self, tmp_path):
        """Galaxies with too few outer radial points get slope_tail = NaN."""
        names = ["Sparse"]
        pd.DataFrame({
            "Galaxy": names, "D": [10.0], "Inc": [45.0],
            "L36": [1.0], "Vflat": [2.0], "e_Vflat": [0.1],
            "MHI": [0.5], "Rdisk": [2.0],
        }).to_csv(tmp_path / "SPARC_Lelli2016c.csv", index=False)
        # 6 total points: r=linspace(1,10,6); 0.7*Rmax=7; outer: 2 pts < 4
        r = np.linspace(1.0, 10.0, 6)
        rc = pd.DataFrame({
            "r": r, "v_obs": np.full(6, 2.0),
            "v_obs_err": np.full(6, 0.1),
            "v_gas": np.full(6, 0.5), "v_disk": np.full(6, 1.5),
            "v_bul": np.zeros(6),
            "SBdisk": np.zeros(6), "SBbul": np.zeros(6),
        })
        rc.to_csv(tmp_path / "Sparse_rotmod.dat", sep=" ", index=False, header=False)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out, verbose=False)
        assert df["slope_tail"].isna().all()
        assert df["delta_f3"].isna().all()
        assert (df["outer_fit_ok"] == False).all()  # noqa: E712
        assert (df["n_tail_points"] < MIN_OUTER_POINTS).all()

    def test_outer_fit_ok_column_present(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out, verbose=False)
        assert "outer_fit_ok" in df.columns

    def test_n_tail_points_column_present(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out, verbose=False)
        assert "n_tail_points" in df.columns

    def test_outer_fit_ok_true_for_normal_curves(self, tmp_path):
        """Standard 30-point curves should yield outer_fit_ok=True."""
        _make_sparc_dir(tmp_path, n_pts=30)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out, verbose=False)
        assert df["outer_fit_ok"].all()

    def test_n_tail_points_positive_for_normal_curves(self, tmp_path):
        _make_sparc_dir(tmp_path, n_pts=30)
        out = tmp_path / "signal.csv"
        df = build_signal_table(tmp_path, out, verbose=False)
        assert (df["n_tail_points"] >= MIN_OUTER_POINTS).all()

    def test_min_outer_points_is_4(self):
        assert MIN_OUTER_POINTS == 4


# ---------------------------------------------------------------------------
# 4. main()
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_dict(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        result = main([
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--quiet",
        ])
        assert isinstance(result, dict)

    def test_dict_has_required_keys(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        result = main([
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--quiet",
        ])
        assert "n_galaxies" in result
        assert "n_slope" in result
        assert "n_env" in result
        assert "out_path" in result
        assert "table" in result

    def test_n_galaxies_correct(self, tmp_path):
        n_gal = 3
        _make_sparc_dir(tmp_path, n_gal=n_gal)
        out = tmp_path / "signal.csv"
        result = main([
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--quiet",
        ])
        assert result["n_galaxies"] == n_gal

    def test_n_env_zero_without_env_csv(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        result = main([
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--quiet",
        ])
        assert result["n_env"] == 0

    def test_n_env_nonzero_with_env_csv(self, tmp_path):
        n_gal = 3
        _make_sparc_dir(tmp_path, n_gal=n_gal)
        galaxies = [f"G{i:02d}" for i in range(n_gal)]
        env_csv = _make_env_csv(tmp_path, galaxies)
        out = tmp_path / "signal.csv"
        result = main([
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--env-csv", str(env_csv),
            "--quiet",
        ])
        assert result["n_env"] == n_gal

    def test_table_is_dataframe(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        result = main([
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--quiet",
        ])
        assert isinstance(result["table"], pd.DataFrame)

    def test_out_path_in_result(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        result = main([
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--quiet",
        ])
        assert result["out_path"] == str(out)

    def test_custom_beta_ref_via_cli(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        result = main([
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--beta-ref", "0.7",
            "--quiet",
        ])
        df = result["table"]
        valid = df["slope_tail"].notna()
        if valid.any():
            np.testing.assert_allclose(
                df.loc[valid, "delta_f3"].values,
                df.loc[valid, "slope_tail"].values - 0.7,
                rtol=1e-9,
            )


# ---------------------------------------------------------------------------
# 5. CLI (subprocess)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "scripts/build_galaxy_signal_table.py", "--help"],
            capture_output=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0

    def test_cli_creates_output(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        result = subprocess.run(
            [
                sys.executable,
                "scripts/build_galaxy_signal_table.py",
                "--sparc-dir", str(tmp_path),
                "--out", str(out),
                "--quiet",
            ],
            capture_output=True,
            cwd=_REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert out.exists()

    def test_cli_output_has_correct_columns(self, tmp_path):
        _make_sparc_dir(tmp_path)
        out = tmp_path / "signal.csv"
        subprocess.run(
            [
                sys.executable,
                "scripts/build_galaxy_signal_table.py",
                "--sparc-dir", str(tmp_path),
                "--out", str(out),
                "--quiet",
            ],
            capture_output=True,
            cwd=_REPO_ROOT,
        )
        df = pd.read_csv(out)
        assert list(df.columns) == OUTPUT_COLUMNS
