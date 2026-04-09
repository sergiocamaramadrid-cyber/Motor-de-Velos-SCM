"""tests/test_build_galaxy_catalog_env.py — Tests for
scripts/build_galaxy_catalog_env.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_galaxy_catalog_env import (
    ENV_CSV_DEFAULT,
    OUTPUT_COLUMNS,
    OUTPUT_CSV_DEFAULT,
    SLOPES_CSV_DEFAULT,
    SPARC_CSV_DEFAULT,
    _ENV_REQUIRED,
    _SLOPES_REQUIRED,
    _SPARC_REQUIRED,
    build_catalog,
    load_env,
    load_slopes,
    load_sparc,
    main,
)


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _galaxies(n: int) -> list[str]:
    return [f"NGC{1000 + i}" for i in range(n)]


def _write_sparc(path: Path, galaxies: list[str], seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    mstar = 10 ** rng.uniform(9.0, 11.5, len(galaxies))
    pd.DataFrame({"galaxy": galaxies, "Mstar": mstar}).to_csv(path, index=False)
    return path


def _write_slopes(path: Path, galaxies: list[str], seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    slopes = rng.uniform(-0.4, 0.1, len(galaxies))
    pd.DataFrame({"galaxy": galaxies, "slope_tail": slopes}).to_csv(
        path, index=False
    )
    return path


def _write_env(path: Path, galaxies: list[str], seed: int = 2) -> Path:
    rng = np.random.default_rng(seed)
    env = rng.normal(0, 1, len(galaxies))
    pd.DataFrame({"galaxy": galaxies, "env_proxy": env}).to_csv(
        path, index=False
    )
    return path


def _make_trio(tmp_path: Path, n: int = 20, seed: int = 42):
    """Return (sparc_csv, slopes_csv, env_csv) paths for n matching galaxies."""
    gals = _galaxies(n)
    sparc = _write_sparc(tmp_path / "sparc_basic.csv", gals, seed=seed)
    slopes = _write_slopes(tmp_path / "slope_tail.csv", gals, seed=seed + 1)
    env = _write_env(tmp_path / "env_proxy.csv", gals, seed=seed + 2)
    return sparc, slopes, env


# ---------------------------------------------------------------------------
# load_sparc
# ---------------------------------------------------------------------------

class TestLoadSparc:
    def test_returns_dataframe(self, tmp_path):
        p = _write_sparc(tmp_path / "s.csv", _galaxies(10))
        df = load_sparc(p)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, tmp_path):
        p = _write_sparc(tmp_path / "s.csv", _galaxies(5))
        df = load_sparc(p)
        assert {"galaxy", "Mstar"}.issubset(df.columns)

    def test_row_count(self, tmp_path):
        p = _write_sparc(tmp_path / "s.csv", _galaxies(15))
        assert len(load_sparc(p)) == 15

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="SPARC summary CSV not found"):
            load_sparc(tmp_path / "missing.csv")

    def test_missing_mstar_raises(self, tmp_path):
        p = tmp_path / "s.csv"
        pd.DataFrame({"galaxy": _galaxies(5)}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_sparc(p)

    def test_missing_galaxy_raises(self, tmp_path):
        p = tmp_path / "s.csv"
        pd.DataFrame({"Mstar": [1e10]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_sparc(p)


# ---------------------------------------------------------------------------
# load_slopes
# ---------------------------------------------------------------------------

class TestLoadSlopes:
    def test_returns_dataframe(self, tmp_path):
        p = _write_slopes(tmp_path / "sl.csv", _galaxies(8))
        assert isinstance(load_slopes(p), pd.DataFrame)

    def test_has_required_columns(self, tmp_path):
        p = _write_slopes(tmp_path / "sl.csv", _galaxies(5))
        df = load_slopes(p)
        assert {"galaxy", "slope_tail"}.issubset(df.columns)

    def test_row_count(self, tmp_path):
        p = _write_slopes(tmp_path / "sl.csv", _galaxies(12))
        assert len(load_slopes(p)) == 12

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Slope-tail CSV not found"):
            load_slopes(tmp_path / "nope.csv")

    def test_missing_slope_tail_raises(self, tmp_path):
        p = tmp_path / "sl.csv"
        pd.DataFrame({"galaxy": _galaxies(5)}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_slopes(p)


# ---------------------------------------------------------------------------
# load_env
# ---------------------------------------------------------------------------

class TestLoadEnv:
    def test_returns_dataframe(self, tmp_path):
        p = _write_env(tmp_path / "e.csv", _galaxies(7))
        assert isinstance(load_env(p), pd.DataFrame)

    def test_has_required_columns(self, tmp_path):
        p = _write_env(tmp_path / "e.csv", _galaxies(5))
        df = load_env(p)
        assert {"galaxy", "env_proxy"}.issubset(df.columns)

    def test_row_count(self, tmp_path):
        p = _write_env(tmp_path / "e.csv", _galaxies(9))
        assert len(load_env(p)) == 9

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Environmental proxy CSV not found"):
            load_env(tmp_path / "nope.csv")

    def test_missing_env_proxy_raises(self, tmp_path):
        p = tmp_path / "e.csv"
        pd.DataFrame({"galaxy": _galaxies(5)}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_env(p)


# ---------------------------------------------------------------------------
# build_catalog
# ---------------------------------------------------------------------------

class TestBuildCatalog:
    def test_returns_dataframe(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        df = build_catalog(sparc, slopes, env, out)
        assert isinstance(df, pd.DataFrame)

    def test_output_csv_written(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        build_catalog(sparc, slopes, env, out)
        assert out.exists()

    def test_output_csv_readable(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        build_catalog(sparc, slopes, env, out)
        df = pd.read_csv(out)
        assert isinstance(df, pd.DataFrame)

    def test_has_logm_column(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        df = build_catalog(sparc, slopes, env, out)
        assert "logM" in df.columns

    def test_logm_equals_log10_mstar(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        df = build_catalog(sparc, slopes, env, out)
        np.testing.assert_allclose(df["logM"], np.log10(df["Mstar"]))

    def test_core_columns_present(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        df = build_catalog(sparc, slopes, env, out)
        for col in ["galaxy", "Mstar", "slope_tail", "env_proxy", "logM"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count_inner_join(self, tmp_path):
        """When all three tables have identical galaxies the row count matches."""
        n = 20
        sparc, slopes, env = _make_trio(tmp_path, n=n)
        out = tmp_path / "cat.csv"
        df = build_catalog(sparc, slopes, env, out)
        assert len(df) == n

    def test_partial_overlap_reduces_rows(self, tmp_path):
        """Galaxies missing from env_proxy drop out of the catalog."""
        gals_all = _galaxies(20)
        gals_env = gals_all[:15]   # only 15 in env proxy
        sparc = _write_sparc(tmp_path / "s.csv", gals_all)
        slopes = _write_slopes(tmp_path / "sl.csv", gals_all)
        env = _write_env(tmp_path / "e.csv", gals_env)
        out = tmp_path / "cat.csv"
        df = build_catalog(sparc, slopes, env, out)
        assert len(df) == 15

    def test_galaxy_column_values_correct(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path, n=10)
        out = tmp_path / "cat.csv"
        df = build_catalog(sparc, slopes, env, out)
        expected = set(_galaxies(10))
        assert set(df["galaxy"]) == expected

    def test_creates_parent_directory(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "sub" / "deep" / "cat.csv"
        build_catalog(sparc, slopes, env, out)
        assert out.exists()

    def test_missing_sparc_raises(self, tmp_path):
        _, slopes, env = _make_trio(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_catalog(tmp_path / "nope.csv", slopes, env, tmp_path / "out.csv")

    def test_missing_slopes_raises(self, tmp_path):
        sparc, _, env = _make_trio(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_catalog(sparc, tmp_path / "nope.csv", env, tmp_path / "out.csv")

    def test_missing_env_raises(self, tmp_path):
        sparc, slopes, _ = _make_trio(tmp_path)
        with pytest.raises(FileNotFoundError):
            build_catalog(sparc, slopes, tmp_path / "nope.csv", tmp_path / "out.csv")

    def test_returned_df_matches_written_csv(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        df_returned = build_catalog(sparc, slopes, env, out)
        df_written = pd.read_csv(out)
        pd.testing.assert_frame_equal(
            df_returned.reset_index(drop=True),
            df_written.reset_index(drop=True),
        )

    def test_no_duplicate_galaxy_rows(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path, n=25)
        out = tmp_path / "cat.csv"
        df = build_catalog(sparc, slopes, env, out)
        assert df["galaxy"].nunique() == len(df)

    def test_mstar_values_positive(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        df = build_catalog(sparc, slopes, env, out)
        assert (df["Mstar"] > 0).all()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--env-csv", str(env),
            "--out", str(out),
        ])
        assert isinstance(result, dict)

    def test_required_keys(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--env-csv", str(env),
            "--out", str(out),
        ])
        assert {"catalog", "n", "out_path"}.issubset(result)

    def test_catalog_is_dataframe(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--env-csv", str(env),
            "--out", str(out),
        ])
        assert isinstance(result["catalog"], pd.DataFrame)

    def test_n_matches_catalog_rows(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path, n=18)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--env-csv", str(env),
            "--out", str(out),
        ])
        assert result["n"] == len(result["catalog"])

    def test_out_path_is_str(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        result = main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--env-csv", str(env),
            "--out", str(out),
        ])
        assert isinstance(result["out_path"], str)

    def test_csv_written_to_disk(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "cat.csv"
        main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--env-csv", str(env),
            "--out", str(out),
        ])
        assert out.exists()

    def test_missing_sparc_raises(self, tmp_path):
        _, slopes, env = _make_trio(tmp_path)
        with pytest.raises(FileNotFoundError):
            main([
                "--sparc-csv", str(tmp_path / "nope.csv"),
                "--slopes-csv", str(slopes),
                "--env-csv", str(env),
                "--out", str(tmp_path / "cat.csv"),
            ])

    def test_creates_nested_output_dir(self, tmp_path):
        sparc, slopes, env = _make_trio(tmp_path)
        out = tmp_path / "nested" / "deep" / "cat.csv"
        main([
            "--sparc-csv", str(sparc),
            "--slopes-csv", str(slopes),
            "--env-csv", str(env),
            "--out", str(out),
        ])
        assert out.exists()


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_sparc_csv_default_is_str(self):
        assert isinstance(SPARC_CSV_DEFAULT, str)

    def test_slopes_csv_default_is_str(self):
        assert isinstance(SLOPES_CSV_DEFAULT, str)

    def test_env_csv_default_is_str(self):
        assert isinstance(ENV_CSV_DEFAULT, str)

    def test_output_csv_default_is_str(self):
        assert isinstance(OUTPUT_CSV_DEFAULT, str)

    def test_output_columns_contains_logm(self):
        assert "logM" in OUTPUT_COLUMNS

    def test_output_columns_contains_env_proxy(self):
        assert "env_proxy" in OUTPUT_COLUMNS

    def test_sparc_required_set(self):
        assert {"galaxy", "Mstar"}.issubset(_SPARC_REQUIRED)

    def test_slopes_required_set(self):
        assert {"galaxy", "slope_tail"}.issubset(_SLOPES_REQUIRED)

    def test_env_required_set(self):
        assert {"galaxy", "env_proxy"}.issubset(_ENV_REQUIRED)
