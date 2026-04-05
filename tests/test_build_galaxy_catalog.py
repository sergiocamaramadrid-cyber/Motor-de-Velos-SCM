"""tests/test_build_galaxy_catalog.py — Tests for scripts/build_galaxy_catalog.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_galaxy_catalog import (
    OUTPUT_COLS,
    build_galaxy_catalog,
    write_summary,
    _compute_logm,
    _resolve_beta,
    _resolve_env_proxy,
    _resolve_galaxy_key,
    _resolve_n_deep,
    _compute_rmax_from_contract,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _f3_csv(tmp_path: Path, n: int = 10, beta_col: str = "friction_slope") -> Path:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "galaxy": [f"NGC{i:04d}" for i in range(n)],
            beta_col: rng.uniform(0.3, 0.9, n),
            "n_deep": rng.integers(5, 30, n),
        }
    )
    p = tmp_path / "f3.csv"
    df.to_csv(p, index=False)
    return p


def _f3_csv_beta_col(tmp_path: Path, n: int = 10) -> Path:
    """F3 catalog with 'beta' instead of 'friction_slope'."""
    return _f3_csv(tmp_path, n=n, beta_col="beta")


def _sparc_csv(tmp_path: Path, n: int = 10) -> Path:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "Galaxy": [f"NGC{i:04d}" for i in range(n)],
            "Inc": rng.uniform(20.0, 80.0, n),
            "L36": rng.uniform(0.5, 50.0, n),
            "MHI": rng.uniform(0.1, 10.0, n),
            "Re": rng.uniform(1.0, 20.0, n),
        }
    )
    p = tmp_path / "sparc.csv"
    df.to_csv(p, index=False)
    return p


def _env_csv(tmp_path: Path, n: int = 10) -> Path:
    rng = np.random.default_rng(13)
    df = pd.DataFrame(
        {
            "galaxy": [f"NGC{i:04d}" for i in range(n)],
            "delta_mass_std": rng.uniform(-1.0, 2.0, n),
        }
    )
    p = tmp_path / "env.csv"
    df.to_csv(p, index=False)
    return p


def _contract_csv(tmp_path: Path, n_gal: int = 5, n_pts: int = 6) -> Path:
    rows = []
    for i in range(n_gal):
        for j in range(1, n_pts + 1):
            rows.append({"galaxy": f"NGC{i:04d}", "r_kpc": float(j * 2)})
    df = pd.DataFrame(rows)
    p = tmp_path / "contract.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------


class TestResolveGalaxyKey:
    def test_galaxy_col(self):
        df = pd.DataFrame({"galaxy": ["A", "B"]})
        assert list(_resolve_galaxy_key(df)) == ["A", "B"]

    def test_Galaxy_col(self):
        df = pd.DataFrame({"Galaxy": ["A", "B"]})
        assert list(_resolve_galaxy_key(df)) == ["A", "B"]

    def test_no_col_raises(self):
        df = pd.DataFrame({"x": [1, 2]})
        with pytest.raises(ValueError, match="galaxy-name"):
            _resolve_galaxy_key(df)


class TestResolveBeta:
    def test_friction_slope(self):
        df = pd.DataFrame({"friction_slope": [0.5, 0.6]})
        s = _resolve_beta(df)
        assert list(s) == [0.5, 0.6]

    def test_beta_fallback(self):
        df = pd.DataFrame({"beta": [0.7, 0.8]})
        s = _resolve_beta(df)
        assert list(s) == [0.7, 0.8]

    def test_friction_slope_preferred_over_beta(self):
        df = pd.DataFrame({"friction_slope": [0.5], "beta": [0.9]})
        s = _resolve_beta(df)
        assert list(s) == [0.5]

    def test_missing_returns_nan(self):
        df = pd.DataFrame({"x": [1.0]})
        s = _resolve_beta(df)
        assert np.isnan(s.iloc[0])


class TestResolveNDeep:
    def test_n_deep(self):
        df = pd.DataFrame({"n_deep": [10, 20]})
        assert list(_resolve_n_deep(df)) == [10, 20]

    def test_missing_returns_nan(self):
        df = pd.DataFrame({"x": [1]})
        s = _resolve_n_deep(df)
        assert np.isnan(s.iloc[0])


class TestResolveEnvProxy:
    def test_delta_mass_std(self):
        df = pd.DataFrame({"delta_mass_std": [0.5, -0.3]})
        s = _resolve_env_proxy(df)
        assert list(s) == [0.5, -0.3]

    def test_delta_mass_fallback(self):
        df = pd.DataFrame({"delta_mass": [1.0, 2.0]})
        s = _resolve_env_proxy(df)
        assert list(s) == [1.0, 2.0]


class TestComputeLogM:
    def test_from_L36_MHI(self):
        df = pd.DataFrame({"L36": [10.0], "MHI": [5.0]})
        s = _compute_logm(df)
        M_bar = 0.5 * 10.0 * 1e9 + 1.33 * 5.0 * 1e9
        expected = np.log10(M_bar)
        assert abs(s.iloc[0] - expected) < 1e-10

    def test_preexisting_log_M_bar(self):
        df = pd.DataFrame({"log_M_bar": [10.5]})
        s = _compute_logm(df)
        assert s.iloc[0] == 10.5

    def test_zero_mass_gives_nan(self):
        df = pd.DataFrame({"L36": [0.0], "MHI": [0.0]})
        s = _compute_logm(df)
        assert np.isnan(s.iloc[0])

    def test_missing_cols_returns_nan(self):
        df = pd.DataFrame({"x": [1.0]})
        s = _compute_logm(df)
        assert np.isnan(s.iloc[0])


class TestComputeRmaxFromContract:
    def test_basic(self):
        df = pd.DataFrame({
            "galaxy": ["A", "A", "B", "B"],
            "r_kpc": [1.0, 5.0, 2.0, 3.0],
        })
        rmax = _compute_rmax_from_contract(df)
        assert set(rmax.columns) == {"galaxy", "Rmax"}
        assert rmax.loc[rmax["galaxy"] == "A", "Rmax"].iloc[0] == 5.0
        assert rmax.loc[rmax["galaxy"] == "B", "Rmax"].iloc[0] == 3.0

    def test_no_r_kpc_col(self):
        df = pd.DataFrame({"galaxy": ["A"], "v": [100.0]})
        rmax = _compute_rmax_from_contract(df)
        assert rmax.empty


# ---------------------------------------------------------------------------
# Integration tests — build_galaxy_catalog
# ---------------------------------------------------------------------------


class TestBuildGalaxyCatalog:
    def test_output_columns(self, tmp_path):
        f3 = _f3_csv(tmp_path)
        sparc = _sparc_csv(tmp_path)
        env = _env_csv(tmp_path)
        cat = build_galaxy_catalog(f3, sparc, env)
        assert list(cat.columns) == OUTPUT_COLS

    def test_row_count_matches_f3(self, tmp_path):
        f3 = _f3_csv(tmp_path, n=8)
        cat = build_galaxy_catalog(f3, sparc_path=None, env_path=None)
        assert len(cat) == 8

    def test_slope_tail_from_friction_slope(self, tmp_path):
        f3 = _f3_csv(tmp_path, beta_col="friction_slope")
        cat = build_galaxy_catalog(f3, None, None)
        assert cat["slope_tail"].notna().all()

    def test_slope_tail_from_beta(self, tmp_path):
        f3 = _f3_csv_beta_col(tmp_path)
        cat = build_galaxy_catalog(f3, None, None)
        assert cat["slope_tail"].notna().all()

    def test_inc_deg_from_sparc(self, tmp_path):
        f3 = _f3_csv(tmp_path)
        sparc = _sparc_csv(tmp_path)
        cat = build_galaxy_catalog(f3, sparc, None)
        assert cat["inc_deg"].notna().all()
        assert (cat["inc_deg"] >= 0).all()

    def test_logM_computed(self, tmp_path):
        f3 = _f3_csv(tmp_path)
        sparc = _sparc_csv(tmp_path)
        cat = build_galaxy_catalog(f3, sparc, None)
        assert cat["logM"].notna().all()
        # logM should be in reasonable range for spiral galaxies
        assert (cat["logM"] > 6.0).all()
        assert (cat["logM"] < 14.0).all()

    def test_env_proxy_from_env_catalog(self, tmp_path):
        f3 = _f3_csv(tmp_path)
        env = _env_csv(tmp_path)
        cat = build_galaxy_catalog(f3, None, env)
        assert cat["env_proxy"].notna().all()

    def test_rmax_from_contract(self, tmp_path):
        f3 = _f3_csv(tmp_path, n=5)
        contract = _contract_csv(tmp_path, n_gal=5)
        cat = build_galaxy_catalog(f3, None, None, contract_path=contract)
        # The 5 galaxies in both f3 and contract should have Rmax
        matched = cat[cat["galaxy_id"].isin([f"NGC{i:04d}" for i in range(5)])]
        assert matched["Rmax"].notna().all()
        assert (matched["Rmax"] == 12.0).all()  # max(2,4,6,8,10,12) = 12

    def test_rmax_fallback_to_Re(self, tmp_path):
        f3 = _f3_csv(tmp_path)
        sparc = _sparc_csv(tmp_path)
        cat = build_galaxy_catalog(f3, sparc, None, contract_path=None)
        assert cat["Rmax"].notna().all()  # Re used as fallback

    def test_missing_sparc_gives_nan_inc(self, tmp_path):
        f3 = _f3_csv(tmp_path)
        cat = build_galaxy_catalog(f3, sparc_path=None, env_path=None)
        assert cat["inc_deg"].isna().all()

    def test_missing_env_gives_nan_proxy(self, tmp_path):
        f3 = _f3_csv(tmp_path)
        cat = build_galaxy_catalog(f3, sparc_path=None, env_path=None)
        assert cat["env_proxy"].isna().all()

    def test_partial_join_leaves_nan(self, tmp_path):
        """Galaxies in F3 but not in env catalog get NaN env_proxy."""
        f3 = _f3_csv(tmp_path, n=5)
        # Only 3 galaxies in env catalog
        env = pd.DataFrame({
            "galaxy": [f"NGC{i:04d}" for i in range(3)],
            "delta_mass_std": [0.1, 0.2, 0.3],
        })
        env_path = tmp_path / "env_partial.csv"
        env.to_csv(env_path, index=False)
        cat = build_galaxy_catalog(f3, None, env_path)
        n_with_env = cat["env_proxy"].notna().sum()
        n_without_env = cat["env_proxy"].isna().sum()
        assert n_with_env == 3
        assert n_without_env == 2

    def test_f3_not_found_raises(self, tmp_path):
        with pytest.raises((ValueError, FileNotFoundError)):
            build_galaxy_catalog(
                f3_path=tmp_path / "nonexistent.csv",
                sparc_path=None,
                env_path=None,
            )

    def test_all_sources_full_coverage(self, tmp_path):
        f3 = _f3_csv(tmp_path)
        sparc = _sparc_csv(tmp_path)
        env = _env_csv(tmp_path)
        cat = build_galaxy_catalog(f3, sparc, env)
        # All columns except possibly Rmax should be filled
        for col in ("galaxy_id", "slope_tail", "n_tail_points", "inc_deg", "logM", "env_proxy"):
            assert cat[col].notna().all(), f"Column {col} has unexpected NaN"

    def test_sorted_by_galaxy_id(self, tmp_path):
        f3 = _f3_csv(tmp_path, n=10)
        cat = build_galaxy_catalog(f3, None, None)
        assert list(cat["galaxy_id"]) == sorted(cat["galaxy_id"])

    def test_no_duplicates(self, tmp_path):
        f3 = _f3_csv(tmp_path, n=10)
        cat = build_galaxy_catalog(f3, None, None)
        assert cat["galaxy_id"].nunique() == len(cat)


# ---------------------------------------------------------------------------
# write_summary tests
# ---------------------------------------------------------------------------


class TestWriteSummary:
    def test_creates_file(self, tmp_path):
        df = pd.DataFrame({col: [1.0, np.nan] for col in OUTPUT_COLS})
        df["galaxy_id"] = ["A", "B"]
        out = tmp_path / "summary.txt"
        text = write_summary(df, out)
        assert out.exists()
        assert len(text) > 0

    def test_content_has_total(self, tmp_path):
        df = pd.DataFrame({col: [1.0] for col in OUTPUT_COLS})
        df["galaxy_id"] = ["A"]
        out = tmp_path / "summary.txt"
        text = write_summary(df, out)
        assert "Total galaxies" in text
        assert "1" in text

    def test_completeness_counts(self, tmp_path):
        df = pd.DataFrame({col: [1.0, np.nan] for col in OUTPUT_COLS})
        df["galaxy_id"] = ["A", "B"]
        out = tmp_path / "summary.txt"
        text = write_summary(df, out)
        # galaxy_id is always present (strings, not float NaN)
        assert "galaxy_id" in text

    def test_all_complete_row(self, tmp_path):
        df = pd.DataFrame({col: [1.0, 2.0] for col in OUTPUT_COLS})
        df["galaxy_id"] = ["A", "B"]
        out = tmp_path / "summary.txt"
        text = write_summary(df, out)
        assert "2 / 2" in text

    def test_creates_parent_dir(self, tmp_path):
        df = pd.DataFrame({col: [1.0] for col in OUTPUT_COLS})
        df["galaxy_id"] = ["A"]
        out = tmp_path / "subdir" / "summary.txt"
        write_summary(df, out)
        assert out.exists()
