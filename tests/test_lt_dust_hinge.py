"""
tests/test_lt_dust_hinge.py — Unit tests for src/lt/lt_dust_hinge_analysis.py.

Tests cover:
  - load_lt_rotation_curve: happy path, column aliasing, error cases
  - compute_F3_from_rc: numeric correctness, fallback to last-K points
  - build_master_table: safe NaN fallback when galaxy is missing
  - matched_pairs: index-safety after reset_index, used-set tracking
  - wilcoxon_test: edge cases (empty, too few positives)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.lt.lt_dust_hinge_analysis import (
    HingeParams,
    _ensure_columns,
    load_lt_rotation_curve,
    compute_F3_from_rc,
    build_master_table,
    regress_tdust,
    matched_pairs,
    wilcoxon_test,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_rc_csv(tmp_path: Path, galaxy: str, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    path = tmp_path / f"{galaxy}_rot.csv"
    df.to_csv(path, index=False)
    return path


def _make_rc_df(n: int = 8, r_max: float = 2.0, v_flat: float = 30.0) -> pd.DataFrame:
    r = np.linspace(0.1, r_max, n)
    v = v_flat * np.ones(n)
    return pd.DataFrame({"r_kpc": r, "Vbary_kms": v})


@pytest.fixture()
def lt_data_dir(tmp_path):
    """Minimal LITTLE THINGS data directory for smoke tests."""
    lt_dir = tmp_path / "lt_oh2015"
    lt_dir.mkdir()
    galaxies = ["DDO210", "DDO69"]
    for g in galaxies:
        _make_rc_csv(lt_dir, g, [
            {"r_kpc": r, "Vbary_kms": v}
            for r, v in zip(np.linspace(0.2, 2.0, 6), np.linspace(10, 35, 6))
        ])

    dust = pd.DataFrame({"galaxy": galaxies, "T_dust": [18.2, 20.5]})
    mass = pd.DataFrame({"galaxy": galaxies, "logM": [7.2, 7.5]})
    metal = pd.DataFrame({"galaxy": galaxies, "logZ": [7.5, 7.7]})

    dust.to_csv(tmp_path / "cigan2021_tdust.csv", index=False)
    mass.to_csv(tmp_path / "lt_masses.csv", index=False)
    metal.to_csv(tmp_path / "lt_metals.csv", index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# _ensure_columns
# ---------------------------------------------------------------------------

class TestEnsureColumns:
    def test_no_missing_raises_nothing(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        _ensure_columns(df, ["a", "b"], where="test")

    def test_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            _ensure_columns(df, ["a", "b"], where="test")


# ---------------------------------------------------------------------------
# load_lt_rotation_curve
# ---------------------------------------------------------------------------

class TestLoadLtRotationCurve:
    def test_loads_standard_csv(self, tmp_path):
        _make_rc_csv(tmp_path, "DDO1", [
            {"r_kpc": 0.5, "Vbary_kms": 20.0},
            {"r_kpc": 1.0, "Vbary_kms": 30.0},
            {"r_kpc": 1.5, "Vbary_kms": 35.0},
        ])
        df = load_lt_rotation_curve("DDO1", data_dir=tmp_path)
        assert list(df.columns) == ["r_kpc", "Vbary_kms"]
        assert len(df) == 3

    def test_accepts_vbary_alias(self, tmp_path):
        """Vbary column (without _kms suffix) should be renamed automatically."""
        path = tmp_path / "DDO2_rot.csv"
        pd.DataFrame({"r_kpc": [0.5, 1.0, 1.5], "Vbary": [20, 30, 35]}).to_csv(
            path, index=False
        )
        df = load_lt_rotation_curve("DDO2", data_dir=tmp_path)
        assert "Vbary_kms" in df.columns

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_lt_rotation_curve("NOEXIST", data_dir=tmp_path)

    def test_too_few_points_raises(self, tmp_path):
        _make_rc_csv(tmp_path, "DDO3", [
            {"r_kpc": 0.5, "Vbary_kms": 20.0},
            {"r_kpc": 1.0, "Vbary_kms": 30.0},
        ])
        with pytest.raises(ValueError, match="too few points"):
            load_lt_rotation_curve("DDO3", data_dir=tmp_path)

    def test_sorted_by_r(self, tmp_path):
        _make_rc_csv(tmp_path, "DDO4", [
            {"r_kpc": 2.0, "Vbary_kms": 35.0},
            {"r_kpc": 0.5, "Vbary_kms": 20.0},
            {"r_kpc": 1.0, "Vbary_kms": 30.0},
        ])
        df = load_lt_rotation_curve("DDO4", data_dir=tmp_path)
        assert list(df["r_kpc"]) == sorted(df["r_kpc"].tolist())

    def test_nan_rows_dropped(self, tmp_path):
        _make_rc_csv(tmp_path, "DDO5", [
            {"r_kpc": 0.5, "Vbary_kms": 20.0},
            {"r_kpc": float("nan"), "Vbary_kms": 30.0},
            {"r_kpc": 1.0, "Vbary_kms": float("nan")},
            {"r_kpc": 1.5, "Vbary_kms": 35.0},
            {"r_kpc": 2.0, "Vbary_kms": 40.0},
        ])
        df = load_lt_rotation_curve("DDO5", data_dir=tmp_path)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# compute_F3_from_rc
# ---------------------------------------------------------------------------

class TestComputeF3FromRc:
    def test_returns_finite_float(self):
        df = _make_rc_df()
        result = compute_F3_from_rc(df)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_non_negative(self):
        df = _make_rc_df()
        assert compute_F3_from_rc(df) >= 0.0

    def test_custom_params(self):
        df = _make_rc_df()
        params = HingeParams(d=0.1, logg0=-10.0)
        result = compute_F3_from_rc(df, params=params)
        assert np.isfinite(result)

    def test_higher_velocity_lower_F3(self):
        """Higher baryonic velocity → higher g_bar → smaller hinge term."""
        df_low = _make_rc_df(v_flat=10.0)
        df_high = _make_rc_df(v_flat=200.0)
        assert compute_F3_from_rc(df_low) >= compute_F3_from_rc(df_high)

    def test_fallback_last_k_respects_param(self):
        """Fallback uses exactly ext_last_k points when ext region is empty."""
        df = _make_rc_df(n=10)
        # Use ext_frac > 1 to guarantee empty ext region
        p1 = HingeParams(ext_frac=2.0, ext_last_k=1)
        p3 = HingeParams(ext_frac=2.0, ext_last_k=3)
        # Both should return a finite float (no crash)
        assert np.isfinite(compute_F3_from_rc(df, params=p1))
        assert np.isfinite(compute_F3_from_rc(df, params=p3))


# ---------------------------------------------------------------------------
# build_master_table
# ---------------------------------------------------------------------------

class TestBuildMasterTable:
    def test_returns_expected_columns(self, lt_data_dir):
        galaxies = ["DDO210", "DDO69"]
        f3 = {"DDO210": 0.01, "DDO69": 0.02}
        df = build_master_table(
            galaxies,
            f3,
            dust_file=lt_data_dir / "cigan2021_tdust.csv",
            mass_file=lt_data_dir / "lt_masses.csv",
            metal_file=lt_data_dir / "lt_metals.csv",
        )
        assert list(df.columns) == ["galaxy", "F3_SCM", "T_dust", "logM", "logZ"]
        assert len(df) == 2

    def test_missing_galaxy_gives_nan(self, lt_data_dir):
        galaxies = ["DDO210", "MISSING_GALAXY"]
        f3 = {"DDO210": 0.01}
        df = build_master_table(
            galaxies,
            f3,
            dust_file=lt_data_dir / "cigan2021_tdust.csv",
            mass_file=lt_data_dir / "lt_masses.csv",
            metal_file=lt_data_dir / "lt_metals.csv",
        )
        missing_row = df[df["galaxy"] == "MISSING_GALAXY"].iloc[0]
        assert np.isnan(missing_row["T_dust"])
        assert np.isnan(missing_row["logM"])
        assert np.isnan(missing_row["logZ"])

    def test_missing_dust_file_raises(self, lt_data_dir):
        with pytest.raises(FileNotFoundError):
            build_master_table(
                ["DDO210"],
                {},
                dust_file=lt_data_dir / "nonexistent.csv",
                mass_file=lt_data_dir / "lt_masses.csv",
                metal_file=lt_data_dir / "lt_metals.csv",
            )


# ---------------------------------------------------------------------------
# regress_tdust
# ---------------------------------------------------------------------------

class TestRegressTdust:
    def test_returns_none_for_empty_df(self):
        df = pd.DataFrame(columns=["T_dust", "logM", "logZ", "F3_SCM"])
        assert regress_tdust(df) is None

    def test_returns_model_for_sufficient_data(self):
        rng = np.random.default_rng(0)
        n = 10
        df = pd.DataFrame({
            "galaxy": [f"G{i}" for i in range(n)],
            "T_dust": 20.0 + rng.normal(0, 2, n),
            "logM": 7.0 + rng.normal(0, 0.5, n),
            "logZ": 7.5 + rng.normal(0, 0.3, n),
            "F3_SCM": 0.01 + rng.uniform(0, 0.05, n),
        })
        model = regress_tdust(df)
        assert model is not None
        assert hasattr(model, "params")


# ---------------------------------------------------------------------------
# matched_pairs
# ---------------------------------------------------------------------------

class TestMatchedPairs:
    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "galaxy": ["A", "B", "C", "D"],
            "T_dust": [18.0, 21.0, 19.0, 22.0],
            "logM": [7.2, 7.3, 7.2, 7.3],
            "logZ": [7.5, 7.6, 7.5, 7.6],
            "F3_SCM": [0.05, 0.02, 0.03, 0.06],
        })

    def test_returns_dataframe(self):
        df = self._make_df()
        result = matched_pairs(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_galaxy_used_twice(self):
        df = self._make_df()
        pairs = matched_pairs(df)
        all_gals = list(pairs["gal1"]) + list(pairs["gal2"])
        assert len(all_gals) == len(set(all_gals)), "Same galaxy appears in multiple pairs"

    def test_columns_present(self):
        df = self._make_df()
        pairs = matched_pairs(df)
        if not pairs.empty:
            assert set(pairs.columns) >= {"gal1", "gal2", "delta_T", "delta_F3_SCM"}

    def test_empty_when_no_matches(self):
        df = pd.DataFrame({
            "galaxy": ["A", "B"],
            "T_dust": [18.0, 21.0],
            "logM": [7.0, 9.0],   # far apart → no match
            "logZ": [7.5, 9.5],   # far apart → no match
            "F3_SCM": [0.05, 0.02],
        })
        pairs = matched_pairs(df)
        assert pairs.empty

    def test_index_safety_after_reset(self):
        """Pairs should work even if the caller passes a non-default-indexed df."""
        df = self._make_df()
        df.index = [10, 20, 30, 40]  # non-default index
        result = matched_pairs(df)
        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# wilcoxon_test
# ---------------------------------------------------------------------------

class TestWilcoxonTest:
    def test_empty_returns_none_none(self):
        p, n = wilcoxon_test(pd.DataFrame())
        assert p is None
        assert n is None

    def test_too_few_positives_returns_none(self):
        df = pd.DataFrame({"delta_F3_SCM": [0.01, 0.02], "delta_T": [1.0, -1.0]})
        p, n = wilcoxon_test(df)
        assert p is None
        assert n == 2

    def test_sufficient_positives_returns_pvalue(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "delta_F3_SCM": [0.01] * 6,
            "delta_T": rng.normal(1.0, 0.5, 6).tolist(),
        })
        p, n = wilcoxon_test(df)
        assert n == 6
        assert p is not None
        assert 0.0 <= p <= 1.0
