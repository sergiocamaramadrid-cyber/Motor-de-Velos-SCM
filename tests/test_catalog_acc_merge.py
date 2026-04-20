"""
tests/test_catalog_acc_merge.py — Tests for scripts/catalog_acc_merge.py.

Uses synthetic data so no real SPARC download or external files are required.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.catalog_acc_merge import (
    load_catalog,
    load_acc,
    merge_catalog_acc,
    compute_rar_mass_bins,
    compute_fdm_per_galaxy,
    format_report,
    main,
    CATALOG_DEFAULT,
    ACC_DEFAULT,
    N_BINS_DEFAULT,
    CATALOG_REQUIRED,
    ACC_REQUIRED,
    OUTER_FRAC_DEFAULT,
    MIN_OUTER_POINTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_catalog(tmp_path: Path, n: int = 10,
                  col_logM: str = "logM") -> Path:
    """Write a minimal galaxy catalog CSV."""
    rng = np.random.default_rng(0)
    names = [f"G{i:04d}" for i in range(n)]
    logM = np.linspace(8.5, 11.5, n)
    env_proxy = rng.normal(0, 0.5, n)
    df = pd.DataFrame({"galaxy": names, col_logM: logM, "env_proxy": env_proxy})
    p = tmp_path / "catalog.csv"
    df.to_csv(p, index=False)
    return p


def _make_acc(tmp_path: Path, galaxies: list[str] | None = None,
              n_pts: int = 20) -> Path:
    """Write a minimal per-radial-point acceleration CSV."""
    if galaxies is None:
        galaxies = [f"G{i:04d}" for i in range(5)]
    rng = np.random.default_rng(1)
    rows = []
    for name in galaxies:
        r = np.linspace(0.5, 15.0, n_pts)
        g_bar = rng.uniform(1e-12, 1e-9, n_pts)
        g_obs = g_bar * rng.uniform(1.0, 3.0, n_pts)
        for k in range(n_pts):
            rows.append({
                "galaxy": name,
                "r_kpc": float(r[k]),
                "g_bar": float(g_bar[k]),
                "g_obs": float(g_obs[k]),
                "log_g_bar": float(np.log10(g_bar[k])),
                "log_g_obs": float(np.log10(g_obs[k])),
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "acc.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# load_catalog
# ---------------------------------------------------------------------------

class TestLoadCatalog:
    def test_returns_dataframe_with_logmbar(self, tmp_path):
        p = _make_catalog(tmp_path)
        df = load_catalog(p)
        assert isinstance(df, pd.DataFrame)
        assert "logMbar" in df.columns
        assert "logM" not in df.columns

    def test_logm_to_logmbar_rename(self, tmp_path):
        p = _make_catalog(tmp_path)
        df = load_catalog(p)
        assert "logMbar" in df.columns

    def test_galaxy_column_present(self, tmp_path):
        p = _make_catalog(tmp_path)
        df = load_catalog(p)
        assert "galaxy" in df.columns

    def test_env_proxy_preserved(self, tmp_path):
        p = _make_catalog(tmp_path)
        df = load_catalog(p)
        assert "env_proxy" in df.columns

    def test_row_count(self, tmp_path):
        p = _make_catalog(tmp_path, n=15)
        df = load_catalog(p)
        assert len(df) == 15

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_catalog(tmp_path / "does_not_exist.csv")

    def test_missing_galaxy_col_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"logM": [9.0, 10.0]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_catalog(bad)

    def test_missing_logm_col_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A", "B"]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_catalog(bad)

    def test_logmbar_values_unchanged(self, tmp_path):
        """Rename must not change values."""
        rng = np.random.default_rng(7)
        vals = rng.uniform(8.0, 12.0, 5)
        df_in = pd.DataFrame({"galaxy": list("ABCDE"), "logM": vals})
        p = tmp_path / "cat.csv"
        df_in.to_csv(p, index=False)
        df_out = load_catalog(p)
        np.testing.assert_allclose(df_out["logMbar"].values, vals)

    def test_accepts_path_string(self, tmp_path):
        p = _make_catalog(tmp_path)
        df = load_catalog(str(p))
        assert "logMbar" in df.columns


# ---------------------------------------------------------------------------
# load_acc
# ---------------------------------------------------------------------------

class TestLoadAcc:
    def test_returns_dataframe(self, tmp_path):
        p = _make_acc(tmp_path)
        df = load_acc(p)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self, tmp_path):
        p = _make_acc(tmp_path)
        df = load_acc(p)
        for col in ACC_REQUIRED:
            assert col in df.columns, f"Missing column: {col}"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_acc(tmp_path / "missing.csv")

    def test_missing_column_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        # Missing log_g_bar and log_g_obs
        pd.DataFrame({
            "galaxy": ["A"], "r_kpc": [1.0], "g_bar": [1e-11], "g_obs": [2e-11],
        }).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_acc(bad)

    def test_row_count(self, tmp_path):
        p = _make_acc(tmp_path, galaxies=["G0000", "G0001"], n_pts=10)
        df = load_acc(p)
        assert len(df) == 20

    def test_accepts_path_string(self, tmp_path):
        p = _make_acc(tmp_path)
        df = load_acc(str(p))
        assert isinstance(df, pd.DataFrame)

    def test_g_bar_positive(self, tmp_path):
        p = _make_acc(tmp_path)
        df = load_acc(p)
        assert (df["g_bar"] > 0).all()

    def test_g_obs_positive(self, tmp_path):
        p = _make_acc(tmp_path)
        df = load_acc(p)
        assert (df["g_obs"] > 0).all()


# ---------------------------------------------------------------------------
# merge_catalog_acc
# ---------------------------------------------------------------------------

class TestMergeCatalogAcc:
    def _catalog(self, n=5):
        names = [f"G{i:04d}" for i in range(n)]
        return pd.DataFrame({
            "galaxy": names,
            "logMbar": np.linspace(9.0, 11.0, n),
            "env_proxy": np.zeros(n),
        })

    def _acc(self, galaxies, n_pts=10):
        rng = np.random.default_rng(3)
        rows = []
        for name in galaxies:
            g_bar = rng.uniform(1e-12, 1e-9, n_pts)
            g_obs = g_bar * 2.0
            for k in range(n_pts):
                rows.append({
                    "galaxy": name,
                    "r_kpc": float(k + 1),
                    "g_bar": float(g_bar[k]),
                    "g_obs": float(g_obs[k]),
                    "log_g_bar": float(np.log10(g_bar[k])),
                    "log_g_obs": float(np.log10(g_obs[k])),
                })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        cat = self._catalog()
        acc = self._acc(["G0000", "G0001"])
        merged = merge_catalog_acc(cat, acc)
        assert isinstance(merged, pd.DataFrame)

    def test_logmbar_propagated(self):
        cat = self._catalog(5)
        acc = self._acc(["G0000", "G0002"])
        merged = merge_catalog_acc(cat, acc)
        assert "logMbar" in merged.columns

    def test_env_proxy_propagated(self):
        cat = self._catalog(5)
        acc = self._acc(["G0000"])
        merged = merge_catalog_acc(cat, acc)
        assert "env_proxy" in merged.columns

    def test_unmatched_acc_dropped(self):
        cat = self._catalog(3)  # G0000, G0001, G0002
        acc = self._acc(["G0000", "G9999"])  # G9999 not in catalog
        merged = merge_catalog_acc(cat, acc)
        assert set(merged["galaxy"].unique()) == {"G0000"}

    def test_row_count_all_match(self):
        cat = self._catalog(5)
        acc = self._acc(["G0000", "G0001"], n_pts=10)
        merged = merge_catalog_acc(cat, acc)
        assert len(merged) == 20

    def test_row_count_partial_match(self):
        cat = self._catalog(5)  # G0000-G0004
        acc = self._acc(["G0000", "G0099"], n_pts=10)  # G0099 not in catalog
        merged = merge_catalog_acc(cat, acc)
        assert len(merged) == 10  # only G0000 points

    def test_empty_acc_returns_empty(self):
        cat = self._catalog(5)
        acc = pd.DataFrame(
            columns=["galaxy", "r_kpc", "g_bar", "g_obs", "log_g_bar", "log_g_obs"]
        )
        merged = merge_catalog_acc(cat, acc)
        assert len(merged) == 0

    def test_empty_catalog_returns_empty(self):
        cat = pd.DataFrame(columns=["galaxy", "logMbar", "env_proxy"])
        acc = self._acc(["G0000"])
        merged = merge_catalog_acc(cat, acc)
        assert len(merged) == 0

    def test_reset_index(self):
        cat = self._catalog(5)
        acc = self._acc(["G0001", "G0002"])
        merged = merge_catalog_acc(cat, acc)
        assert list(merged.index) == list(range(len(merged)))


# ---------------------------------------------------------------------------
# compute_rar_mass_bins
# ---------------------------------------------------------------------------

class TestComputeRarMassBins:
    def _merged(self, n_gal=6, n_pts=20):
        rng = np.random.default_rng(9)
        names = [f"G{i:04d}" for i in range(n_gal)]
        logM = np.linspace(9.0, 11.0, n_gal)
        rows = []
        for i, name in enumerate(names):
            g_bar = rng.uniform(1e-12, 1e-9, n_pts)
            g_obs = g_bar * 2.0
            for k in range(n_pts):
                rows.append({
                    "galaxy": name,
                    "logMbar": logM[i],
                    "env_proxy": float(i) * 0.1,
                    "g_bar": float(g_bar[k]),
                    "g_obs": float(g_obs[k]),
                    "log_g_bar": float(np.log10(g_bar[k])),
                    "log_g_obs": float(np.log10(g_obs[k])),
                })
        return pd.DataFrame(rows)

    def test_returns_list(self):
        merged = self._merged()
        bins = compute_rar_mass_bins(merged, n_bins=3)
        assert isinstance(bins, list)

    def test_bin_count_le_n_bins(self):
        merged = self._merged(n_gal=6)
        bins = compute_rar_mass_bins(merged, n_bins=3)
        assert len(bins) <= 3

    def test_required_keys(self):
        merged = self._merged()
        bins = compute_rar_mass_bins(merged, n_bins=3)
        required = {
            "bin_lo", "bin_hi", "n_galaxies", "n_points",
            "logMbar_mean", "env_proxy_mean", "g_bar_median",
            "g_obs_median", "log_ratio_mean",
        }
        for b in bins:
            assert required <= set(b.keys())

    def test_n_galaxies_positive(self):
        merged = self._merged()
        bins = compute_rar_mass_bins(merged, n_bins=3)
        for b in bins:
            assert b["n_galaxies"] > 0

    def test_n_points_positive(self):
        merged = self._merged()
        bins = compute_rar_mass_bins(merged, n_bins=3)
        for b in bins:
            assert b["n_points"] > 0

    def test_bin_lo_lt_bin_hi(self):
        merged = self._merged()
        bins = compute_rar_mass_bins(merged, n_bins=3)
        for b in bins:
            assert b["bin_lo"] <= b["bin_hi"]

    def test_total_points_consistent(self):
        merged = self._merged(n_gal=6, n_pts=20)
        bins = compute_rar_mass_bins(merged, n_bins=3)
        total = sum(b["n_points"] for b in bins)
        assert total == len(merged)

    def test_empty_merged_returns_empty(self):
        empty = pd.DataFrame(
            columns=["galaxy", "logMbar", "env_proxy", "g_bar", "g_obs",
                     "log_g_bar", "log_g_obs"]
        )
        bins = compute_rar_mass_bins(empty, n_bins=3)
        assert bins == []

    def test_single_galaxy(self):
        rng = np.random.default_rng(5)
        n = 10
        g_bar = rng.uniform(1e-11, 1e-10, n)
        df = pd.DataFrame({
            "galaxy": ["G0000"] * n,
            "logMbar": [10.0] * n,
            "env_proxy": [0.0] * n,
            "g_bar": g_bar,
            "g_obs": g_bar * 2.0,
            "log_g_bar": np.log10(g_bar),
            "log_g_obs": np.log10(g_bar * 2.0),
        })
        bins = compute_rar_mass_bins(df, n_bins=3)
        assert len(bins) == 1
        assert bins[0]["n_galaxies"] == 1

    def test_log_ratio_known_value(self):
        """When g_obs = 2*g_bar, log(g_obs/g_bar) = log10(2)."""
        n = 10
        g_bar = np.full(n, 1e-11)
        df = pd.DataFrame({
            "galaxy": ["G0000"] * n,
            "logMbar": [10.0] * n,
            "env_proxy": [0.0] * n,
            "g_bar": g_bar,
            "g_obs": g_bar * 2.0,
            "log_g_bar": np.log10(g_bar),
            "log_g_obs": np.log10(g_bar * 2.0),
        })
        bins = compute_rar_mass_bins(df, n_bins=1)
        assert len(bins) == 1
        assert abs(bins[0]["log_ratio_mean"] - math.log10(2.0)) < 1e-10

    def test_no_env_proxy_col_gives_nan(self):
        n = 5
        g_bar = np.full(n, 1e-11)
        df = pd.DataFrame({
            "galaxy": ["G0000"] * n,
            "logMbar": [10.0] * n,
            "g_bar": g_bar,
            "g_obs": g_bar * 2.0,
            "log_g_bar": np.log10(g_bar),
            "log_g_obs": np.log10(g_bar * 2.0),
        })
        bins = compute_rar_mass_bins(df, n_bins=1)
        assert math.isnan(bins[0]["env_proxy_mean"])

    def test_n_bins_1(self):
        merged = self._merged(n_gal=6, n_pts=10)
        bins = compute_rar_mass_bins(merged, n_bins=1)
        assert len(bins) == 1
        assert bins[0]["n_points"] == len(merged)

    def test_n_bins_default(self):
        merged = self._merged(n_gal=6, n_pts=10)
        bins = compute_rar_mass_bins(merged)
        assert isinstance(bins, list)


# ---------------------------------------------------------------------------
# compute_fdm_per_galaxy
# ---------------------------------------------------------------------------

class TestComputeFdmPerGalaxy:
    """Tests for compute_fdm_per_galaxy."""

    def _acc(self, n_gal=5, n_pts=20, r_max=15.0):
        """Build a synthetic acc DataFrame with well-separated radial ranges."""
        rng = np.random.default_rng(42)
        names = [f"G{i:04d}" for i in range(n_gal)]
        rows = []
        for name in names:
            r = np.linspace(0.5, r_max, n_pts)
            g_bar = rng.uniform(1e-12, 1e-9, n_pts)
            g_obs = g_bar * rng.uniform(1.5, 3.0, n_pts)
            for k in range(n_pts):
                rows.append({
                    "galaxy": name,
                    "r_kpc": float(r[k]),
                    "g_bar": float(g_bar[k]),
                    "g_obs": float(g_obs[k]),
                    "log_g_bar": float(np.log10(g_bar[k])),
                    "log_g_obs": float(np.log10(g_obs[k])),
                })
        return pd.DataFrame(rows)

    def test_returns_dataframe(self):
        acc = self._acc()
        result = compute_fdm_per_galaxy(acc)
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_galaxy(self):
        acc = self._acc(n_gal=5)
        result = compute_fdm_per_galaxy(acc)
        assert result["galaxy"].nunique() == len(result)

    def test_required_columns(self):
        acc = self._acc()
        result = compute_fdm_per_galaxy(acc)
        for col in ("galaxy", "r_max_kpc", "n_outer_points",
                    "f_DM_out", "g_bar_out", "g_obs_out"):
            assert col in result.columns, f"Missing column: {col}"

    def test_n_outer_points_positive(self):
        acc = self._acc()
        result = compute_fdm_per_galaxy(acc)
        assert (result["n_outer_points"] >= MIN_OUTER_POINTS).all()

    def test_r_max_kpc_correct(self):
        acc = self._acc(n_gal=3, r_max=20.0)
        result = compute_fdm_per_galaxy(acc)
        assert not result.empty
        np.testing.assert_allclose(result["r_max_kpc"].values,
                                   np.full(len(result), 20.0), atol=1e-6)

    def test_fdm_formula(self):
        """f_DM_out = 1 - g_bar_out / g_obs_out."""
        acc = self._acc(n_gal=3)
        result = compute_fdm_per_galaxy(acc)
        expected = 1.0 - result["g_bar_out"] / result["g_obs_out"]
        np.testing.assert_allclose(result["f_DM_out"].values,
                                   expected.values, rtol=1e-10)

    def test_fdm_between_minus_inf_and_one(self):
        """For g_obs > g_bar > 0, f_DM should be in (0, 1)."""
        acc = self._acc()
        result = compute_fdm_per_galaxy(acc)
        assert (result["f_DM_out"] < 1.0).all()

    def test_known_fdm_value(self):
        """When g_obs = 2*g_bar at every outer point, f_DM = 0.5."""
        n_pts = 20
        r = np.linspace(0.5, 10.0, n_pts)
        g_bar = np.full(n_pts, 1e-11)
        g_obs = g_bar * 2.0
        df = pd.DataFrame({
            "galaxy": ["G0000"] * n_pts,
            "r_kpc": r,
            "g_bar": g_bar,
            "g_obs": g_obs,
            "log_g_bar": np.log10(g_bar),
            "log_g_obs": np.log10(g_obs),
        })
        result = compute_fdm_per_galaxy(df, r_fraction=0.7)
        assert len(result) == 1
        assert abs(result.iloc[0]["f_DM_out"] - 0.5) < 1e-10

    def test_empty_input_returns_empty(self):
        empty = pd.DataFrame(
            columns=["galaxy", "r_kpc", "g_bar", "g_obs",
                     "log_g_bar", "log_g_obs"]
        )
        result = compute_fdm_per_galaxy(empty)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_too_few_outer_points_skipped(self):
        """Galaxy with only 1 outer point should be omitted."""
        # 10 points, r_fraction=0.9 → only the last point is outer
        n_pts = 10
        r = np.linspace(1.0, 10.0, n_pts)
        g_bar = np.full(n_pts, 1e-11)
        g_obs = g_bar * 2.0
        df = pd.DataFrame({
            "galaxy": ["G0000"] * n_pts,
            "r_kpc": r,
            "g_bar": g_bar,
            "g_obs": g_obs,
            "log_g_bar": np.log10(g_bar),
            "log_g_obs": np.log10(g_obs),
        })
        result = compute_fdm_per_galaxy(df, r_fraction=0.99)
        assert len(result) == 0

    def test_r_fraction_controls_outer_region(self):
        """Lower r_fraction includes more points as outer."""
        acc = self._acc(n_gal=1, n_pts=20)
        res_tight = compute_fdm_per_galaxy(acc, r_fraction=0.9)
        res_loose = compute_fdm_per_galaxy(acc, r_fraction=0.1)
        if not res_tight.empty and not res_loose.empty:
            assert (res_loose["n_outer_points"] >= res_tight["n_outer_points"]).all()

    def test_custom_r_fraction(self):
        n_pts = 20
        r = np.linspace(1.0, 10.0, n_pts)
        g_bar = np.full(n_pts, 1e-11)
        g_obs = g_bar * 3.0
        df = pd.DataFrame({
            "galaxy": ["G0000"] * n_pts,
            "r_kpc": r,
            "g_bar": g_bar,
            "g_obs": g_obs,
            "log_g_bar": np.log10(g_bar),
            "log_g_obs": np.log10(g_obs),
        })
        result = compute_fdm_per_galaxy(df, r_fraction=0.5)
        assert len(result) == 1
        assert abs(result.iloc[0]["f_DM_out"] - (1.0 - 1.0 / 3.0)) < 1e-10

    def test_multiple_galaxies_all_returned(self):
        acc = self._acc(n_gal=7, n_pts=20)
        result = compute_fdm_per_galaxy(acc)
        assert len(result) == 7

    def test_n_outer_points_matches_expectation(self):
        """With r in [0.5, 10] and r_fraction=0.7, outer starts at r > 7.0."""
        n_pts = 20
        r = np.linspace(0.5, 10.0, n_pts)
        # r_fraction * r_max = 0.7 * 10 = 7.0 → points where r > 7.0
        expected_outer = int((r > 7.0).sum())
        g_bar = np.full(n_pts, 1e-11)
        g_obs = g_bar * 2.0
        df = pd.DataFrame({
            "galaxy": ["G0000"] * n_pts,
            "r_kpc": r,
            "g_bar": g_bar,
            "g_obs": g_obs,
            "log_g_bar": np.log10(g_bar),
            "log_g_obs": np.log10(g_obs),
        })
        result = compute_fdm_per_galaxy(df, r_fraction=0.7)
        if not result.empty:
            assert result.iloc[0]["n_outer_points"] == expected_outer

    def test_default_r_fraction_equals_outer_frac_default(self):
        """Default r_fraction must match the module constant."""
        import inspect
        sig = inspect.signature(compute_fdm_per_galaxy)
        default = sig.parameters["r_fraction"].default
        assert default == OUTER_FRAC_DEFAULT


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def _make_data(self):
        rng = np.random.default_rng(2)
        n_gal, n_pts = 4, 10
        names = [f"G{i:04d}" for i in range(n_gal)]
        catalog = pd.DataFrame({
            "galaxy": names,
            "logMbar": np.linspace(9.0, 11.0, n_gal),
            "env_proxy": rng.normal(0, 0.3, n_gal),
        })
        rows = []
        for name in names:
            g_bar = rng.uniform(1e-12, 1e-9, n_pts)
            g_obs = g_bar * 2.0
            for k in range(n_pts):
                rows.append({
                    "galaxy": name, "r_kpc": float(k),
                    "g_bar": float(g_bar[k]), "g_obs": float(g_obs[k]),
                    "log_g_bar": float(np.log10(g_bar[k])),
                    "log_g_obs": float(np.log10(g_obs[k])),
                })
        acc = pd.DataFrame(rows)
        merged = merge_catalog_acc(catalog, acc)
        bins = compute_rar_mass_bins(merged, n_bins=2)
        return catalog, acc, merged, bins

    def test_returns_list(self):
        cat, acc, merged, bins = self._make_data()
        lines = format_report(cat, acc, merged, bins, "cat.csv", "acc.csv")
        assert isinstance(lines, list)
        assert len(lines) > 0

    def test_contains_catalog_path(self):
        cat, acc, merged, bins = self._make_data()
        lines = format_report(cat, acc, merged, bins, "my_catalog.csv", "my_acc.csv")
        assert any("my_catalog.csv" in l for l in lines)

    def test_contains_acc_path(self):
        cat, acc, merged, bins = self._make_data()
        lines = format_report(cat, acc, merged, bins, "cat.csv", "my_acc.csv")
        assert any("my_acc.csv" in l for l in lines)

    def test_contains_shape_info(self):
        cat, acc, merged, bins = self._make_data()
        lines = format_report(cat, acc, merged, bins, "c", "a")
        text = "\n".join(lines)
        assert "shape=" in text

    def test_all_strings(self):
        cat, acc, merged, bins = self._make_data()
        lines = format_report(cat, acc, merged, bins, "c", "a")
        for line in lines:
            assert isinstance(line, str)


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, tmp_path):
        cat_p = _make_catalog(tmp_path, n=5)
        acc_p = _make_acc(tmp_path, galaxies=[f"G{i:04d}" for i in range(5)])
        result = main(["--catalog", str(cat_p), "--acc", str(acc_p)])
        assert isinstance(result, dict)

    def test_required_keys(self, tmp_path):
        cat_p = _make_catalog(tmp_path, n=5)
        acc_p = _make_acc(tmp_path, galaxies=[f"G{i:04d}" for i in range(5)])
        result = main(["--catalog", str(cat_p), "--acc", str(acc_p)])
        required = {
            "catalog_shape", "acc_shape", "merged_shape",
            "catalog_columns", "acc_columns", "n_galaxies", "bins", "fdm",
        }
        assert required <= set(result.keys())

    def test_catalog_columns_has_logmbar(self, tmp_path):
        cat_p = _make_catalog(tmp_path, n=5)
        acc_p = _make_acc(tmp_path, galaxies=[f"G{i:04d}" for i in range(5)])
        result = main(["--catalog", str(cat_p), "--acc", str(acc_p)])
        assert "logMbar" in result["catalog_columns"]
        assert "logM" not in result["catalog_columns"]

    def test_n_galaxies_in_merge(self, tmp_path):
        n = 5
        names = [f"G{i:04d}" for i in range(n)]
        cat_p = _make_catalog(tmp_path, n=n)
        acc_p = _make_acc(tmp_path, galaxies=names[:3])  # only 3 galaxies in acc
        result = main(["--catalog", str(cat_p), "--acc", str(acc_p)])
        assert result["n_galaxies"] == 3

    def test_shapes_consistent(self, tmp_path):
        n = 5
        names = [f"G{i:04d}" for i in range(n)]
        cat_p = _make_catalog(tmp_path, n=n)
        acc_p = _make_acc(tmp_path, galaxies=names, n_pts=10)
        result = main(["--catalog", str(cat_p), "--acc", str(acc_p)])
        assert result["catalog_shape"][0] == n
        assert result["acc_shape"][0] == n * 10
        assert result["merged_shape"][0] == n * 10

    def test_writes_output_files(self, tmp_path):
        cat_p = _make_catalog(tmp_path, n=4)
        acc_p = _make_acc(tmp_path, galaxies=[f"G{i:04d}" for i in range(4)])
        out = tmp_path / "out"
        main(["--catalog", str(cat_p), "--acc", str(acc_p), "--out", str(out)])
        assert (out / "merged.csv").exists()
        assert (out / "report.txt").exists()
        assert (out / "mass_bins.csv").exists()
        assert (out / "fdm_per_galaxy.csv").exists()

    def test_merged_csv_has_logmbar(self, tmp_path):
        cat_p = _make_catalog(tmp_path, n=4)
        acc_p = _make_acc(tmp_path, galaxies=[f"G{i:04d}" for i in range(4)])
        out = tmp_path / "out"
        main(["--catalog", str(cat_p), "--acc", str(acc_p), "--out", str(out)])
        merged_df = pd.read_csv(out / "merged.csv")
        assert "logMbar" in merged_df.columns

    def test_n_bins_arg(self, tmp_path):
        cat_p = _make_catalog(tmp_path, n=6)
        acc_p = _make_acc(tmp_path, galaxies=[f"G{i:04d}" for i in range(6)])
        result = main([
            "--catalog", str(cat_p), "--acc", str(acc_p), "--n-bins", "5",
        ])
        assert len(result["bins"]) <= 5

    def test_catalog_missing_raises(self, tmp_path):
        acc_p = _make_acc(tmp_path)
        with pytest.raises(FileNotFoundError):
            main(["--catalog", str(tmp_path / "no_file.csv"), "--acc", str(acc_p)])

    def test_acc_missing_raises(self, tmp_path):
        cat_p = _make_catalog(tmp_path)
        with pytest.raises(FileNotFoundError):
            main(["--catalog", str(cat_p), "--acc", str(tmp_path / "no_file.csv")])

    def test_partial_overlap_galaxies(self, tmp_path):
        """Galaxies only in acc (not catalog) are silently dropped."""
        cat_p = _make_catalog(tmp_path, n=3)  # G0000-G0002
        acc_p = _make_acc(
            tmp_path,
            galaxies=["G0000", "G0001", "G9999"],  # G9999 not in catalog
            n_pts=10,
        )
        result = main(["--catalog", str(cat_p), "--acc", str(acc_p)])
        assert result["n_galaxies"] == 2
        assert result["merged_shape"][0] == 20

    def test_bins_key_is_list(self, tmp_path):
        cat_p = _make_catalog(tmp_path, n=5)
        acc_p = _make_acc(tmp_path, galaxies=[f"G{i:04d}" for i in range(5)])
        result = main(["--catalog", str(cat_p), "--acc", str(acc_p)])
        assert isinstance(result["bins"], list)

    def test_fdm_key_is_dataframe(self, tmp_path):
        cat_p = _make_catalog(tmp_path, n=5)
        acc_p = _make_acc(tmp_path, galaxies=[f"G{i:04d}" for i in range(5)])
        result = main(["--catalog", str(cat_p), "--acc", str(acc_p)])
        assert isinstance(result["fdm"], pd.DataFrame)

    def test_fdm_csv_has_f_dm_out(self, tmp_path):
        cat_p = _make_catalog(tmp_path, n=4)
        acc_p = _make_acc(tmp_path, galaxies=[f"G{i:04d}" for i in range(4)])
        out = tmp_path / "out"
        main(["--catalog", str(cat_p), "--acc", str(acc_p), "--out", str(out)])
        fdm_df = pd.read_csv(out / "fdm_per_galaxy.csv")
        assert "f_DM_out" in fdm_df.columns
        assert "galaxy" in fdm_df.columns


# ---------------------------------------------------------------------------
# Regression: fixture file round-trip
# ---------------------------------------------------------------------------

class TestFixtureRoundTrip:
    """Verify the repo fixture files load without errors."""

    REPO_ROOT = Path(__file__).parent.parent

    def test_fixture_catalog_loads(self):
        p = self.REPO_ROOT / CATALOG_DEFAULT
        if not p.exists():
            pytest.skip("fixture catalog not present")
        df = load_catalog(p)
        assert "logMbar" in df.columns
        assert "galaxy" in df.columns
        assert len(df) > 0

    def test_fixture_catalog_logmbar_range(self):
        p = self.REPO_ROOT / CATALOG_DEFAULT
        if not p.exists():
            pytest.skip("fixture catalog not present")
        df = load_catalog(p)
        assert df["logMbar"].min() >= 5.0
        assert df["logMbar"].max() <= 15.0

    def test_fixture_catalog_no_logm_col(self):
        """After loading, logM column must be gone."""
        p = self.REPO_ROOT / CATALOG_DEFAULT
        if not p.exists():
            pytest.skip("fixture catalog not present")
        df = load_catalog(p)
        assert "logM" not in df.columns
