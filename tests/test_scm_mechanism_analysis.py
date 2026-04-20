"""
tests/test_scm_mechanism_analysis.py — Tests for scripts/scm_mechanism_analysis.py.

Uses synthetic data so no real SPARC download or external files are required.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.scm_mechanism_analysis import (
    load_catalog,
    compute_fdm_from_rotmods,
    _parse_rotmod,
    _fdm_from_rotmod,
    build_dataset,
    run_correlations,
    run_regressions,
    model_comparison_table,
    plot_h1_diagnostic,
    plot_env_proxy_robustness,
    main,
    CATALOG_DEFAULT,
    SPARC_DIR_DEFAULT,
    OUTER_FRAC_DEFAULT,
    MIN_OUTER_POINTS,
    N_PERM_DEFAULT,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_catalog(
    tmp_path: Path,
    n: int = 20,
    include_rdisk: bool = False,
    logm_col: str = "logMbar",
) -> Path:
    """Write a minimal galaxy catalog CSV with all required columns."""
    rng = np.random.default_rng(42)
    names = [f"G{i:04d}" for i in range(n)]
    logM = np.linspace(8.5, 11.5, n)
    env_proxy = rng.normal(0, 0.5, n)
    slope_tail = -0.05 * env_proxy + rng.normal(0, 0.02, n)

    data: dict = {
        "galaxy": names,
        logm_col: logM,
        "env_proxy": env_proxy,
        "slope_tail": slope_tail,
    }
    if include_rdisk:
        data["Rdisk"] = rng.uniform(1.0, 5.0, n)
        data["Rmax"] = data["Rdisk"] * rng.uniform(2.0, 5.0, n)

    df = pd.DataFrame(data)
    p = tmp_path / "catalog.csv"
    df.to_csv(p, index=False)
    return p


def _make_rotmod_file(tmp_path: Path, galaxy: str, n_pts: int = 30) -> Path:
    """Write a minimal SPARC _rotmod.dat file."""
    rng = np.random.default_rng(hash(galaxy) % (2**32))
    r = np.linspace(0.5, 20.0, n_pts)
    vobs = 100 + 20 * np.tanh(r / 5) + rng.normal(0, 2, n_pts)
    verr = rng.uniform(2, 5, n_pts)
    vgas = rng.uniform(5, 30, n_pts)
    vdisk = rng.uniform(20, 80, n_pts)
    vbul = rng.uniform(0, 20, n_pts)
    sbdisk = rng.uniform(0, 1, n_pts)
    sbbul = rng.uniform(0, 0.5, n_pts)

    lines = []
    for i in range(n_pts):
        lines.append(
            f"{r[i]:.4f}  {vobs[i]:.4f}  {verr[i]:.4f}  "
            f"{vgas[i]:.4f}  {vdisk[i]:.4f}  {vbul[i]:.4f}  "
            f"{sbdisk[i]:.4f}  {sbbul[i]:.4f}"
        )
    content = "\n".join(lines) + "\n"
    p = tmp_path / f"{galaxy}_rotmod.dat"
    p.write_text(content)
    return p


def _make_fdm(n: int = 20) -> pd.DataFrame:
    """Build a synthetic f_DM DataFrame."""
    rng = np.random.default_rng(7)
    names = [f"G{i:04d}" for i in range(n)]
    return pd.DataFrame({
        "galaxy": names,
        "r_max_kpc": rng.uniform(5, 25, n),
        "n_outer_points": rng.integers(3, 10, n),
        "f_DM_out": rng.uniform(0.0, 0.8, n),
        "v_bar_out": rng.uniform(30, 100, n),
        "v_obs_out": rng.uniform(80, 150, n),
    })


def _make_dataset(n: int = 20) -> pd.DataFrame:
    """Build a synthetic merged dataset for regression/correlation tests."""
    rng = np.random.default_rng(3)
    logM = np.linspace(8.5, 11.5, n)
    env_proxy = rng.normal(0, 0.5, n)
    f_DM = 0.3 - 0.03 * logM + rng.normal(0, 0.05, n)
    slope_tail = 0.1 * env_proxy - 0.05 * f_DM + rng.normal(0, 0.02, n)
    return pd.DataFrame({
        "galaxy": [f"G{i:04d}" for i in range(n)],
        "logMbar": logM,
        "env_proxy": env_proxy,
        "slope_tail": slope_tail,
        "f_DM_out": f_DM,
        "Rdisk_Rmax": rng.uniform(0.1, 0.6, n),
    })


# ---------------------------------------------------------------------------
# TestLoadCatalog
# ---------------------------------------------------------------------------

class TestLoadCatalog:
    def test_returns_dataframe(self, tmp_path):
        p = _make_catalog(tmp_path)
        df = load_catalog(p)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self, tmp_path):
        p = _make_catalog(tmp_path)
        df = load_catalog(p)
        for col in ("galaxy", "logMbar", "env_proxy", "slope_tail"):
            assert col in df.columns

    def test_logm_alias_renamed(self, tmp_path):
        p = _make_catalog(tmp_path, logm_col="logM")
        df = load_catalog(p)
        assert "logMbar" in df.columns
        assert "logM" not in df.columns

    def test_rdisk_rmax_derived(self, tmp_path):
        p = _make_catalog(tmp_path, include_rdisk=True)
        df = load_catalog(p)
        assert "Rdisk_Rmax" in df.columns
        assert df["Rdisk_Rmax"].notna().any()

    def test_rdisk_rmax_absent_when_columns_missing(self, tmp_path):
        p = _make_catalog(tmp_path, include_rdisk=False)
        df = load_catalog(p)
        assert "Rdisk_Rmax" not in df.columns

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_catalog(tmp_path / "nonexistent.csv")

    def test_raises_missing_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["G0"]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_catalog(p)

    def test_row_count_matches(self, tmp_path):
        p = _make_catalog(tmp_path, n=15)
        df = load_catalog(p)
        assert len(df) == 15

    def test_accepts_path_string(self, tmp_path):
        p = _make_catalog(tmp_path)
        df = load_catalog(str(p))
        assert len(df) > 0

    def test_rdisk_rmax_nan_when_rmax_zero(self, tmp_path):
        p = _make_catalog(tmp_path, include_rdisk=True)
        data = pd.read_csv(p)
        data.loc[0, "Rmax"] = 0.0
        data.to_csv(p, index=False)
        df = load_catalog(p)
        assert np.isnan(df.loc[0, "Rdisk_Rmax"])


# ---------------------------------------------------------------------------
# TestParseRotmod
# ---------------------------------------------------------------------------

class TestParseRotmod:
    def test_returns_dataframe(self, tmp_path):
        fp = _make_rotmod_file(tmp_path, "G0000")
        df = _parse_rotmod(fp)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, tmp_path):
        fp = _make_rotmod_file(tmp_path, "G0000")
        df = _parse_rotmod(fp)
        for col in ("Rad", "Vobs", "Vgas", "Vdisk", "Vbul"):
            assert col in df.columns

    def test_returns_none_for_empty_file(self, tmp_path):
        fp = tmp_path / "empty_rotmod.dat"
        fp.write_text("")
        result = _parse_rotmod(fp)
        assert result is None

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        result = _parse_rotmod(tmp_path / "missing_rotmod.dat")
        assert result is None

    def test_excludes_zero_radius_rows(self, tmp_path):
        fp = tmp_path / "G_test_rotmod.dat"
        fp.write_text("0.0 100 5 10 80 5 0.5 0.1\n1.0 105 5 10 82 5 0.5 0.1\n")
        df = _parse_rotmod(fp)
        assert (df["Rad"] > 0).all()


# ---------------------------------------------------------------------------
# TestFdmFromRotmod
# ---------------------------------------------------------------------------

class TestFdmFromRotmod:
    def _make_rc(self, n: int = 20) -> pd.DataFrame:
        r = np.linspace(1.0, 20.0, n)
        return pd.DataFrame({
            "Rad": r,
            "Vobs": np.full(n, 100.0),
            "Vgas": np.full(n, 20.0),
            "Vdisk": np.full(n, 70.0),
            "Vbul": np.full(n, 10.0),
        })

    def test_returns_dict(self):
        rc = self._make_rc()
        result = _fdm_from_rotmod(rc, OUTER_FRAC_DEFAULT)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        rc = self._make_rc()
        result = _fdm_from_rotmod(rc, OUTER_FRAC_DEFAULT)
        assert result is not None
        for key in ("r_max_kpc", "n_outer_points", "f_DM_out", "v_bar_out", "v_obs_out"):
            assert key in result

    def test_returns_none_too_few_outer_points(self):
        rc = pd.DataFrame({
            "Rad": [1.0, 2.0, 3.0],
            "Vobs": [100.0, 100.0, 100.0],
            "Vgas": [20.0, 20.0, 20.0],
            "Vdisk": [70.0, 70.0, 70.0],
            "Vbul": [10.0, 10.0, 10.0],
        })
        # With r_fraction=0.7, only the last point qualifies → fewer than MIN_OUTER_POINTS
        result = _fdm_from_rotmod(rc, 0.9)
        assert result is None

    def test_f_dm_out_range(self):
        rc = self._make_rc()
        result = _fdm_from_rotmod(rc, OUTER_FRAC_DEFAULT)
        assert result is not None
        # f_DM can be negative when baryons dominate but should be finite
        assert math.isfinite(result["f_DM_out"])

    def test_n_outer_points_positive(self):
        rc = self._make_rc()
        result = _fdm_from_rotmod(rc, OUTER_FRAC_DEFAULT)
        assert result is not None
        assert result["n_outer_points"] >= MIN_OUTER_POINTS


# ---------------------------------------------------------------------------
# TestComputeFdmFromRotmods
# ---------------------------------------------------------------------------

class TestComputeFdmFromRotmods:
    def test_returns_dataframe(self, tmp_path):
        _make_rotmod_file(tmp_path, "NGC1234")
        df = compute_fdm_from_rotmods(tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, tmp_path):
        _make_rotmod_file(tmp_path, "NGC1234")
        df = compute_fdm_from_rotmods(tmp_path)
        for col in ("galaxy", "r_max_kpc", "n_outer_points", "f_DM_out"):
            assert col in df.columns

    def test_empty_when_no_rotmods(self, tmp_path):
        df = compute_fdm_from_rotmods(tmp_path)
        assert df.empty

    def test_one_row_per_galaxy(self, tmp_path):
        for name in ["NGC1", "NGC2", "NGC3"]:
            _make_rotmod_file(tmp_path, name)
        df = compute_fdm_from_rotmods(tmp_path)
        assert len(df) == 3
        assert df["galaxy"].nunique() == 3

    def test_galaxy_name_extracted(self, tmp_path):
        _make_rotmod_file(tmp_path, "NGC4321")
        df = compute_fdm_from_rotmods(tmp_path)
        assert "NGC4321" in df["galaxy"].values

    def test_searches_raw_subdir(self, tmp_path):
        raw = tmp_path / "raw"
        raw.mkdir()
        _make_rotmod_file(raw, "NGC9999")
        df = compute_fdm_from_rotmods(tmp_path)
        assert "NGC9999" in df["galaxy"].values

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        df = compute_fdm_from_rotmods(tmp_path / "missing")
        assert df.empty

    def test_r_fraction_parameter(self, tmp_path):
        _make_rotmod_file(tmp_path, "NGC1")
        df07 = compute_fdm_from_rotmods(tmp_path, r_fraction=0.7)
        df09 = compute_fdm_from_rotmods(tmp_path, r_fraction=0.9)
        # More restrictive fraction → fewer or equal outer points
        if not df07.empty and not df09.empty:
            assert df09["n_outer_points"].iloc[0] <= df07["n_outer_points"].iloc[0]


# ---------------------------------------------------------------------------
# TestBuildDataset
# ---------------------------------------------------------------------------

class TestBuildDataset:
    def test_returns_dataframe(self, tmp_path):
        p = _make_catalog(tmp_path)
        catalog = load_catalog(p)
        fdm = _make_fdm(n=len(catalog))
        fdm["galaxy"] = catalog["galaxy"]
        ds = build_dataset(catalog, fdm)
        assert isinstance(ds, pd.DataFrame)

    def test_f_dm_out_present_after_merge(self, tmp_path):
        p = _make_catalog(tmp_path, n=10)
        catalog = load_catalog(p)
        fdm = _make_fdm(n=10)
        fdm["galaxy"] = catalog["galaxy"]
        ds = build_dataset(catalog, fdm)
        assert "f_DM_out" in ds.columns

    def test_empty_fdm_adds_nan_column(self, tmp_path):
        p = _make_catalog(tmp_path, n=5)
        catalog = load_catalog(p)
        fdm = pd.DataFrame(columns=["galaxy", "r_max_kpc", "n_outer_points",
                                     "f_DM_out", "v_bar_out", "v_obs_out"])
        ds = build_dataset(catalog, fdm)
        assert "f_DM_out" in ds.columns
        assert ds["f_DM_out"].isna().all()

    def test_catalog_columns_preserved(self, tmp_path):
        p = _make_catalog(tmp_path, n=10)
        catalog = load_catalog(p)
        fdm = _make_fdm(n=10)
        fdm["galaxy"] = catalog["galaxy"]
        ds = build_dataset(catalog, fdm)
        for col in ("galaxy", "logMbar", "env_proxy", "slope_tail"):
            assert col in ds.columns

    def test_partial_fdm_match(self, tmp_path):
        p = _make_catalog(tmp_path, n=10)
        catalog = load_catalog(p)
        # Only 5 galaxies have f_DM
        fdm = _make_fdm(n=5)
        fdm["galaxy"] = catalog["galaxy"].iloc[:5].values
        ds = build_dataset(catalog, fdm)
        assert len(ds) == 10  # all catalog rows preserved (left join)
        assert ds["f_DM_out"].notna().sum() == 5


# ---------------------------------------------------------------------------
# TestRunCorrelations
# ---------------------------------------------------------------------------

class TestRunCorrelations:
    def test_returns_list(self):
        ds = _make_dataset()
        result = run_correlations(ds)
        assert isinstance(result, list)

    def test_four_pairs_returned(self):
        ds = _make_dataset()
        result = run_correlations(ds)
        assert len(result) == 4

    def test_each_dict_has_required_keys(self):
        ds = _make_dataset()
        result = run_correlations(ds)
        for r in result:
            for key in ("pair", "x", "y", "n", "rho", "p_value", "significant"):
                assert key in r

    def test_rho_in_range(self):
        ds = _make_dataset()
        result = run_correlations(ds)
        for r in result:
            if not np.isnan(r["rho"]):
                assert -1.0 <= r["rho"] <= 1.0

    def test_significant_flag_correct(self):
        ds = _make_dataset()
        result = run_correlations(ds)
        for r in result:
            if not np.isnan(r["p_value"]):
                expected = r["p_value"] < 0.05
                assert r["significant"] == expected

    def test_missing_column_gives_nan(self):
        ds = _make_dataset().drop(columns=["Rdisk_Rmax"])
        result = run_correlations(ds)
        rdisk_pair = next(r for r in result if r["x"] == "Rdisk_Rmax")
        assert np.isnan(rdisk_pair["rho"])

    def test_n_matches_valid_rows(self):
        ds = _make_dataset()
        ds.loc[0, "slope_tail"] = np.nan
        result = run_correlations(ds)
        env_pair = next(r for r in result if r["x"] == "env_proxy")
        assert env_pair["n"] == len(ds) - 1


# ---------------------------------------------------------------------------
# TestRunRegressions
# ---------------------------------------------------------------------------

class TestRunRegressions:
    def test_returns_list_of_four(self):
        ds = _make_dataset()
        result = run_regressions(ds)
        assert isinstance(result, list)
        assert len(result) == 4

    def test_model_names(self):
        ds = _make_dataset()
        result = run_regressions(ds)
        names = {r["model"] for r in result}
        assert names == {"M0_base", "M1_H1", "M2_H2H3", "M3_full"}

    def test_r2_in_range(self):
        ds = _make_dataset()
        result = run_regressions(ds)
        for r in result:
            if not np.isnan(r["r2"]):
                assert 0.0 <= r["r2"] <= 1.0

    def test_full_model_r2_gte_base(self):
        ds = _make_dataset(n=30)
        result = run_regressions(ds)
        base = next(r for r in result if r["model"] == "M0_base")
        full = next(r for r in result if r["model"] == "M3_full")
        if not np.isnan(base["r2"]) and not np.isnan(full["r2"]):
            assert full["r2"] >= base["r2"] - 1e-9

    def test_coef_dict_nonempty(self):
        ds = _make_dataset()
        result = run_regressions(ds)
        for r in result:
            if r["n"] > 0:
                assert len(r["coef"]) > 0

    def test_const_in_coef(self):
        ds = _make_dataset()
        result = run_regressions(ds)
        base = next(r for r in result if r["model"] == "M0_base")
        if base["coef"]:
            assert "const" in base["coef"]

    def test_without_rdisk_rmax_still_runs(self):
        ds = _make_dataset().drop(columns=["Rdisk_Rmax"])
        result = run_regressions(ds)
        assert len(result) == 4

    def test_without_fdm_m1_empty(self):
        ds = _make_dataset().drop(columns=["f_DM_out"])
        result = run_regressions(ds)
        m1 = next(r for r in result if r["model"] == "M1_H1")
        assert m1["n"] == 0


# ---------------------------------------------------------------------------
# TestModelComparisonTable
# ---------------------------------------------------------------------------

class TestModelComparisonTable:
    def test_returns_dataframe(self):
        ds = _make_dataset()
        reg = run_regressions(ds)
        ct = model_comparison_table(reg)
        assert isinstance(ct, pd.DataFrame)

    def test_expected_columns(self):
        ds = _make_dataset()
        reg = run_regressions(ds)
        ct = model_comparison_table(reg)
        for col in ("model", "n", "r2", "aic", "delta_aic", "winner"):
            assert col in ct.columns

    def test_winner_has_delta_aic_zero(self):
        ds = _make_dataset()
        reg = run_regressions(ds)
        ct = model_comparison_table(reg)
        winners = ct[ct["winner"]]
        if not winners.empty:
            assert (winners["delta_aic"] == 0.0).all()

    def test_exactly_one_winner_when_valid(self):
        ds = _make_dataset()
        reg = run_regressions(ds)
        ct = model_comparison_table(reg)
        valid = ct[ct["delta_aic"].notna()]
        if not valid.empty:
            assert valid["winner"].sum() == 1

    def test_delta_aic_nonnegative(self):
        ds = _make_dataset()
        reg = run_regressions(ds)
        ct = model_comparison_table(reg)
        valid = ct[ct["delta_aic"].notna()]
        assert (valid["delta_aic"] >= -1e-9).all()

    def test_all_nan_aic_gives_nan_delta(self):
        fake_reg = [{"model": "M", "n": 0, "k": 1,
                     "r2": np.nan, "r2_adj": np.nan, "aic": np.nan,
                     "coef": {}, "pval": {}, "se": {}}]
        ct = model_comparison_table(fake_reg)
        assert ct["delta_aic"].isna().all()


# ---------------------------------------------------------------------------
# TestPlotH1Diagnostic
# ---------------------------------------------------------------------------

class TestPlotH1Diagnostic:
    def test_returns_path(self, tmp_path):
        ds = _make_dataset()
        reg = run_regressions(ds)
        result = plot_h1_diagnostic(ds, reg, tmp_path)
        assert isinstance(result, Path)

    def test_png_file_created(self, tmp_path):
        ds = _make_dataset()
        reg = run_regressions(ds)
        plot_h1_diagnostic(ds, reg, tmp_path)
        assert (tmp_path / "h1_diagnostic.png").exists()

    def test_pdf_file_created(self, tmp_path):
        ds = _make_dataset()
        reg = run_regressions(ds)
        plot_h1_diagnostic(ds, reg, tmp_path)
        assert (tmp_path / "h1_diagnostic.pdf").exists()

    def test_creates_output_dir(self, tmp_path):
        ds = _make_dataset()
        reg = run_regressions(ds)
        new_dir = tmp_path / "subdir" / "figs"
        plot_h1_diagnostic(ds, reg, new_dir)
        assert new_dir.is_dir()

    def test_works_without_fdm_column(self, tmp_path):
        ds = _make_dataset().drop(columns=["f_DM_out"])
        reg = run_regressions(ds)
        result = plot_h1_diagnostic(ds, reg, tmp_path)
        assert result.exists()


# ---------------------------------------------------------------------------
# TestPlotEnvProxyRobustness
# ---------------------------------------------------------------------------

class TestPlotEnvProxyRobustness:
    def test_returns_dict(self):
        ds = _make_dataset()
        result = plot_env_proxy_robustness(ds, n_perm=50, seed=0)
        assert isinstance(result, dict)

    def test_expected_keys(self):
        ds = _make_dataset()
        result = plot_env_proxy_robustness(ds, n_perm=50, seed=0)
        for key in ("rho_obs", "p_perm", "n_perm", "n_galaxies",
                    "rho_null_mean", "rho_null_std"):
            assert key in result

    def test_rho_obs_in_range(self):
        ds = _make_dataset()
        result = plot_env_proxy_robustness(ds, n_perm=50, seed=0)
        assert -1.0 <= result["rho_obs"] <= 1.0

    def test_p_perm_in_range(self):
        ds = _make_dataset()
        result = plot_env_proxy_robustness(ds, n_perm=100, seed=0)
        assert 0.0 <= result["p_perm"] <= 1.0

    def test_n_galaxies_correct(self):
        ds = _make_dataset(n=20)
        result = plot_env_proxy_robustness(ds, n_perm=20, seed=0)
        assert result["n_galaxies"] == 20

    def test_n_perm_respected(self):
        ds = _make_dataset()
        result = plot_env_proxy_robustness(ds, n_perm=77, seed=0)
        assert result["n_perm"] == 77

    def test_creates_figures_when_out_dir(self, tmp_path):
        ds = _make_dataset()
        plot_env_proxy_robustness(ds, n_perm=20, seed=0, out_dir=tmp_path)
        assert (tmp_path / "env_robustness.png").exists()
        assert (tmp_path / "env_robustness.pdf").exists()

    def test_nan_when_too_few_rows(self):
        ds = pd.DataFrame({
            "env_proxy": [0.1, 0.2],
            "slope_tail": [0.3, 0.4],
            "logMbar": [10.0, 10.5],
        })
        result = plot_env_proxy_robustness(ds, n_perm=10, seed=0)
        assert np.isnan(result["rho_obs"])

    def test_seed_reproducibility(self):
        ds = _make_dataset(n=25)
        r1 = plot_env_proxy_robustness(ds, n_perm=50, seed=99)
        r2 = plot_env_proxy_robustness(ds, n_perm=50, seed=99)
        assert r1["p_perm"] == r2["p_perm"]


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------

class TestMain:
    def _write_catalog(self, tmp_path: Path) -> Path:
        return _make_catalog(tmp_path, n=15)

    def test_returns_dict(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        out = tmp_path / "out"
        result = main([
            "--catalog", str(cat),
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--n-perm", "20",
        ])
        assert isinstance(result, dict)

    def test_required_keys(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        out = tmp_path / "out"
        result = main([
            "--catalog", str(cat),
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--n-perm", "20",
        ])
        for key in ("n_catalog", "n_fdm", "n_dataset", "correlations",
                    "regressions", "model_table", "permutation"):
            assert key in result

    def test_n_catalog_correct(self, tmp_path):
        cat = _make_catalog(tmp_path, n=12)
        out = tmp_path / "out"
        result = main([
            "--catalog", str(cat),
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--n-perm", "10",
        ])
        assert result["n_catalog"] == 12

    def test_output_files_written(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        out = tmp_path / "out"
        main([
            "--catalog", str(cat),
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--n-perm", "10",
        ])
        assert (out / "dataset.csv").exists()
        assert (out / "correlations.csv").exists()
        assert (out / "permutation_test.csv").exists()

    def test_figures_written(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        out = tmp_path / "out"
        main([
            "--catalog", str(cat),
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--n-perm", "10",
        ])
        assert (out / "h1_diagnostic.png").exists()
        assert (out / "env_robustness.png").exists()

    def test_with_rotmod_files(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        cat_df = pd.read_csv(cat)
        for gal in cat_df["galaxy"].iloc[:5]:
            _make_rotmod_file(tmp_path, gal)
        out = tmp_path / "out"
        result = main([
            "--catalog", str(cat),
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--n-perm", "10",
        ])
        assert result["n_fdm"] == 5

    def test_correlations_list_length(self, tmp_path):
        cat = self._write_catalog(tmp_path)
        out = tmp_path / "out"
        result = main([
            "--catalog", str(cat),
            "--sparc-dir", str(tmp_path),
            "--out", str(out),
            "--n-perm", "10",
        ])
        assert len(result["correlations"]) == 4
