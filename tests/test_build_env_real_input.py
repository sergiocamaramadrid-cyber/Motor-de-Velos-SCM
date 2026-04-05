"""
tests/test_build_env_real_input.py — Tests for scripts/build_env_real_input.py.

Covers: clean_name, load_f3_catalog, load_sparc_basic, load_chae_env,
        build_crossmatch, save_output, main (CLI).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_env_real_input import (
    F3_REQUIRED,
    SPARC_REQUIRED,
    CHAE_REQUIRED,
    build_crossmatch,
    clean_name,
    load_chae_env,
    load_f3_catalog,
    load_sparc_basic,
    main as build_main,
    save_output,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_f3(tmp_path: Path, n: int = 10, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "galaxy": [f"NGC {1000 + i}" for i in range(n)],
        "F3": rng.uniform(0.3, 0.7, n),
    })
    p = tmp_path / "f3.csv"
    df.to_csv(p, index=False)
    return p


def _write_sparc(tmp_path: Path, n: int = 10, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Galaxy": [f"NGC {1000 + i}" for i in range(n)],
        "L36": rng.uniform(1.0, 50.0, n),
        "MHI": rng.uniform(0.1, 10.0, n),
    })
    p = tmp_path / "sparc.csv"
    df.to_csv(p, index=False)
    return p


def _write_chae(tmp_path: Path, n: int = 10, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "galaxy_name": [f"NGC{1000 + i}" for i in range(n)],
        "e_env": rng.uniform(-1.0, 1.0, n),
    })
    p = tmp_path / "chae.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# clean_name
# ---------------------------------------------------------------------------

class TestCleanName:
    def test_strips_whitespace(self):
        assert clean_name("  NGC 253  ") == "NGC253"

    def test_removes_hyphens(self):
        assert clean_name("NGC-253") == "NGC253"

    def test_uppercases(self):
        assert clean_name("ngc253") == "NGC253"

    def test_removes_spaces_and_hyphens(self):
        assert clean_name("ngc - 253") == "NGC253"

    def test_non_string_input(self):
        assert clean_name(253) == "253"

    def test_empty_string(self):
        assert clean_name("") == ""

    def test_idempotent(self):
        name = "NGC253"
        assert clean_name(clean_name(name)) == clean_name(name)


# ---------------------------------------------------------------------------
# load_f3_catalog
# ---------------------------------------------------------------------------

class TestLoadF3Catalog:
    def test_loads_valid_csv(self, tmp_path):
        p = _write_f3(tmp_path, n=5)
        df = load_f3_catalog(p)
        assert len(df) == 5

    def test_columns_galaxy_name_and_delta_f3(self, tmp_path):
        p = _write_f3(tmp_path, n=5)
        df = load_f3_catalog(p)
        assert list(df.columns) == ["galaxy_name", "delta_f3"]

    def test_delta_f3_is_F3_minus_05(self, tmp_path):
        raw = pd.DataFrame({"galaxy": ["NGC1"], "F3": [0.8]})
        p = tmp_path / "f3.csv"
        raw.to_csv(p, index=False)
        df = load_f3_catalog(p)
        assert df["delta_f3"].iloc[0] == pytest.approx(0.3, rel=1e-9)

    def test_galaxy_name_normalised(self, tmp_path):
        raw = pd.DataFrame({"galaxy": ["ngc - 253"], "F3": [0.5]})
        p = tmp_path / "f3.csv"
        raw.to_csv(p, index=False)
        df = load_f3_catalog(p)
        assert df["galaxy_name"].iloc[0] == "NGC253"

    def test_missing_column_raises(self, tmp_path):
        raw = pd.DataFrame({"galaxy": ["NGC1"]})
        p = tmp_path / "f3.csv"
        raw.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_f3_catalog(p)

    def test_required_columns_listed(self):
        assert set(F3_REQUIRED) == {"galaxy", "F3"}

    def test_non_numeric_F3_coerced_to_nan(self, tmp_path):
        raw = pd.DataFrame({"galaxy": ["NGC1", "NGC2"], "F3": ["bad", "0.5"]})
        p = tmp_path / "f3.csv"
        raw.to_csv(p, index=False)
        df = load_f3_catalog(p)
        assert pd.isna(df["delta_f3"].iloc[0])
        assert df["delta_f3"].iloc[1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# load_sparc_basic
# ---------------------------------------------------------------------------

class TestLoadSparcBasic:
    def test_loads_valid_csv(self, tmp_path):
        p = _write_sparc(tmp_path, n=5)
        df = load_sparc_basic(p)
        assert len(df) == 5

    def test_columns_galaxy_name_and_logM(self, tmp_path):
        p = _write_sparc(tmp_path, n=5)
        df = load_sparc_basic(p)
        assert list(df.columns) == ["galaxy_name", "logM"]

    def test_logM_formula(self, tmp_path):
        L36, MHI = 10.0, 5.0
        expected = np.log10(0.5 * L36 * 1e9 + 1.33 * MHI * 1e9)
        raw = pd.DataFrame({"Galaxy": ["NGC1"], "L36": [L36], "MHI": [MHI]})
        p = tmp_path / "sparc.csv"
        raw.to_csv(p, index=False)
        df = load_sparc_basic(p)
        assert df["logM"].iloc[0] == pytest.approx(expected, rel=1e-9)

    def test_galaxy_name_normalised(self, tmp_path):
        raw = pd.DataFrame({"Galaxy": ["ngc 253"], "L36": [10.0], "MHI": [1.0]})
        p = tmp_path / "sparc.csv"
        raw.to_csv(p, index=False)
        df = load_sparc_basic(p)
        assert df["galaxy_name"].iloc[0] == "NGC253"

    def test_missing_column_raises(self, tmp_path):
        raw = pd.DataFrame({"Galaxy": ["NGC1"], "L36": [10.0]})
        p = tmp_path / "sparc.csv"
        raw.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_sparc_basic(p)

    def test_required_columns_listed(self):
        assert set(SPARC_REQUIRED) == {"Galaxy", "L36", "MHI"}

    def test_non_numeric_coerced_to_nan(self, tmp_path):
        raw = pd.DataFrame({"Galaxy": ["NGC1"], "L36": ["bad"], "MHI": [1.0]})
        p = tmp_path / "sparc.csv"
        raw.to_csv(p, index=False)
        df = load_sparc_basic(p)
        assert pd.isna(df["logM"].iloc[0])


# ---------------------------------------------------------------------------
# load_chae_env
# ---------------------------------------------------------------------------

class TestLoadChaeEnv:
    def test_loads_valid_csv(self, tmp_path):
        p = _write_chae(tmp_path, n=5)
        df = load_chae_env(p)
        assert len(df) == 5

    def test_columns_galaxy_name_and_e_env(self, tmp_path):
        p = _write_chae(tmp_path, n=5)
        df = load_chae_env(p)
        assert list(df.columns) == ["galaxy_name", "e_env"]

    def test_galaxy_name_normalised(self, tmp_path):
        raw = pd.DataFrame({"galaxy_name": ["ngc - 253"], "e_env": [0.3]})
        p = tmp_path / "chae.csv"
        raw.to_csv(p, index=False)
        df = load_chae_env(p)
        assert df["galaxy_name"].iloc[0] == "NGC253"

    def test_missing_column_raises(self, tmp_path):
        raw = pd.DataFrame({"galaxy_name": ["NGC1"]})
        p = tmp_path / "chae.csv"
        raw.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_chae_env(p)

    def test_required_columns_listed(self):
        assert set(CHAE_REQUIRED) == {"galaxy_name", "e_env"}

    def test_non_numeric_e_env_coerced_to_nan(self, tmp_path):
        raw = pd.DataFrame({"galaxy_name": ["NGC1", "NGC2"], "e_env": ["bad", "0.5"]})
        p = tmp_path / "chae.csv"
        raw.to_csv(p, index=False)
        df = load_chae_env(p)
        assert pd.isna(df["e_env"].iloc[0])
        assert df["e_env"].iloc[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# build_crossmatch
# ---------------------------------------------------------------------------

class TestBuildCrossmatch:
    def _make_inputs(self, n: int = 5):
        rng = np.random.default_rng(0)
        names = [f"NGC{1000 + i}" for i in range(n)]
        f3 = pd.DataFrame({"galaxy_name": names, "delta_f3": rng.uniform(-0.2, 0.2, n)})
        sparc = pd.DataFrame({"galaxy_name": names, "logM": rng.uniform(9, 11, n)})
        chae = pd.DataFrame({"galaxy_name": names, "e_env": rng.uniform(-1, 1, n)})
        return f3, sparc, chae

    def test_returns_dataframe(self):
        f3, sparc, chae = self._make_inputs()
        result = build_crossmatch(f3, sparc, chae)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns(self):
        f3, sparc, chae = self._make_inputs()
        result = build_crossmatch(f3, sparc, chae)
        assert list(result.columns) == ["galaxy_name", "delta_f3", "logM", "e_env"]

    def test_full_overlap_returns_all_rows(self):
        f3, sparc, chae = self._make_inputs(n=8)
        result = build_crossmatch(f3, sparc, chae)
        assert len(result) == 8

    def test_partial_overlap(self):
        names_all = [f"NGC{1000 + i}" for i in range(5)]
        f3 = pd.DataFrame({"galaxy_name": names_all[:3], "delta_f3": [0.1, 0.2, 0.3]})
        sparc = pd.DataFrame({"galaxy_name": names_all[1:4], "logM": [10.0, 10.1, 10.2]})
        chae = pd.DataFrame({"galaxy_name": names_all[2:5], "e_env": [0.5, 0.6, 0.7]})
        result = build_crossmatch(f3, sparc, chae)
        assert len(result) == 1
        assert result["galaxy_name"].iloc[0] == "NGC1002"

    def test_no_overlap_returns_empty(self):
        f3 = pd.DataFrame({"galaxy_name": ["NGC1"], "delta_f3": [0.1]})
        sparc = pd.DataFrame({"galaxy_name": ["NGC2"], "logM": [10.0]})
        chae = pd.DataFrame({"galaxy_name": ["NGC3"], "e_env": [0.5]})
        result = build_crossmatch(f3, sparc, chae)
        assert len(result) == 0

    def test_nan_rows_dropped(self):
        f3 = pd.DataFrame({"galaxy_name": ["NGC1", "NGC2"], "delta_f3": [float("nan"), 0.1]})
        sparc = pd.DataFrame({"galaxy_name": ["NGC1", "NGC2"], "logM": [10.0, 10.1]})
        chae = pd.DataFrame({"galaxy_name": ["NGC1", "NGC2"], "e_env": [0.5, 0.6]})
        result = build_crossmatch(f3, sparc, chae)
        assert len(result) == 1
        assert result["galaxy_name"].iloc[0] == "NGC2"

    def test_index_reset(self):
        f3, sparc, chae = self._make_inputs(n=5)
        result = build_crossmatch(f3, sparc, chae)
        assert list(result.index) == list(range(len(result)))

    def test_column_order(self):
        f3, sparc, chae = self._make_inputs(n=3)
        result = build_crossmatch(f3, sparc, chae)
        assert result.columns.tolist() == ["galaxy_name", "delta_f3", "logM", "e_env"]


# ---------------------------------------------------------------------------
# save_output
# ---------------------------------------------------------------------------

class TestSaveOutput:
    def _make_df(self):
        return pd.DataFrame({
            "galaxy_name": ["NGC1", "NGC2"],
            "delta_f3": [0.1, -0.1],
            "logM": [10.0, 10.5],
            "e_env": [0.3, -0.3],
        })

    def test_creates_file(self, tmp_path):
        df = self._make_df()
        out = tmp_path / "out.csv"
        save_output(df, out)
        assert out.exists()

    def test_roundtrip(self, tmp_path):
        df = self._make_df()
        out = tmp_path / "out.csv"
        save_output(df, out)
        loaded = pd.read_csv(out)
        pd.testing.assert_frame_equal(df, loaded)

    def test_creates_parent_dirs(self, tmp_path):
        df = self._make_df()
        out = tmp_path / "deep" / "sub" / "out.csv"
        save_output(df, out)
        assert out.exists()

    def test_accepts_string_path(self, tmp_path):
        df = self._make_df()
        out = str(tmp_path / "out.csv")
        save_output(df, out)
        assert Path(out).exists()


# ---------------------------------------------------------------------------
# Integration (main CLI)
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_returns_dataframe(self, tmp_path):
        f3_path = _write_f3(tmp_path, n=5)
        sparc_path = _write_sparc(tmp_path, n=5)
        chae_path = _write_chae(tmp_path, n=5)
        out_path = tmp_path / "out.csv"
        result = build_main([
            "--f3-catalog", str(f3_path),
            "--sparc-basic", str(sparc_path),
            "--chae-env", str(chae_path),
            "--out", str(out_path),
        ])
        assert isinstance(result, pd.DataFrame)

    def test_main_writes_file(self, tmp_path):
        f3_path = _write_f3(tmp_path, n=5)
        sparc_path = _write_sparc(tmp_path, n=5)
        chae_path = _write_chae(tmp_path, n=5)
        out_path = tmp_path / "out.csv"
        build_main([
            "--f3-catalog", str(f3_path),
            "--sparc-basic", str(sparc_path),
            "--chae-env", str(chae_path),
            "--out", str(out_path),
        ])
        assert out_path.exists()

    def test_main_output_columns(self, tmp_path):
        f3_path = _write_f3(tmp_path, n=5)
        sparc_path = _write_sparc(tmp_path, n=5)
        chae_path = _write_chae(tmp_path, n=5)
        out_path = tmp_path / "out.csv"
        result = build_main([
            "--f3-catalog", str(f3_path),
            "--sparc-basic", str(sparc_path),
            "--chae-env", str(chae_path),
            "--out", str(out_path),
        ])
        assert set(result.columns) == {"galaxy_name", "delta_f3", "logM", "e_env"}

    def test_main_full_overlap_count(self, tmp_path):
        f3_path = _write_f3(tmp_path, n=8)
        sparc_path = _write_sparc(tmp_path, n=8)
        chae_path = _write_chae(tmp_path, n=8)
        out_path = tmp_path / "out.csv"
        result = build_main([
            "--f3-catalog", str(f3_path),
            "--sparc-basic", str(sparc_path),
            "--chae-env", str(chae_path),
            "--out", str(out_path),
        ])
        assert len(result) == 8

    def test_main_partial_overlap(self, tmp_path):
        """Only galaxies present in all three catalogs should survive."""
        def _write_csv(path, galaxies, col, values):
            pd.DataFrame({"galaxy" if col == "F3" else ("Galaxy" if col == "logM" else "galaxy_name"): galaxies,
                          col: values}).to_csv(path, index=False)

        f3 = pd.DataFrame({"galaxy": [f"NGC{i}" for i in range(5)], "F3": [0.5] * 5})
        sparc = pd.DataFrame({"Galaxy": [f"NGC{i}" for i in range(3)], "L36": [10.0] * 3, "MHI": [1.0] * 3})
        chae = pd.DataFrame({"galaxy_name": [f"NGC{i}" for i in range(2)], "e_env": [0.3, 0.4]})

        f3_path = tmp_path / "f3.csv"
        sparc_path = tmp_path / "sparc.csv"
        chae_path = tmp_path / "chae.csv"
        f3.to_csv(f3_path, index=False)
        sparc.to_csv(sparc_path, index=False)
        chae.to_csv(chae_path, index=False)

        result = build_main([
            "--f3-catalog", str(f3_path),
            "--sparc-basic", str(sparc_path),
            "--chae-env", str(chae_path),
            "--out", str(tmp_path / "out.csv"),
        ])
        assert len(result) == 2

    def test_default_arg_values(self, tmp_path):
        from scripts.build_env_real_input import _parse_args, DEFAULT_F3_CATALOG, DEFAULT_SPARC_BASIC, DEFAULT_CHAE_ENV, DEFAULT_OUT
        args = _parse_args([])
        assert args.f3_catalog == DEFAULT_F3_CATALOG
        assert args.sparc_basic == DEFAULT_SPARC_BASIC
        assert args.chae_env == DEFAULT_CHAE_ENV
        assert args.out == DEFAULT_OUT
