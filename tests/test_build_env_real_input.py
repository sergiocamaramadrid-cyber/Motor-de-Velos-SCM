"""
tests/test_build_env_real_input.py — Tests for scripts/build_env_real_input.py.

Covers:
  1. clean_name()           — name normalisation
  2. load_f3_catalog()      — F3 catalog loader
  3. load_sparc_basic()     — SPARC basic table loader
  4. load_chae_env()        — Chae environment catalog loader
  5. compute_logM()         — baryonic mass proxy formula
  6. merge_catalogs()       — three-way inner join
  7. build_env_real_input() — end-to-end pipeline (writes CSV)
  8. main()                 — CLI and keyword-argument API
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_env_real_input import (
    MOND_REF,
    UPSILON_36,
    ALPHA_HI,
    clean_name,
    load_f3_catalog,
    load_sparc_basic,
    load_chae_env,
    compute_logM,
    merge_catalogs,
    build_env_real_input,
    main,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _write_f3(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    """Write a minimal F3 catalog CSV and return its path."""
    if rows is None:
        rows = [
            {"galaxy": "NGC1234", "F3": 0.52},
            {"galaxy": "UGC5678", "F3": 0.48},
            {"galaxy": "IC910",   "F3": 0.50},
        ]
    p = tmp_path / "f3_catalog.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _write_sparc(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    """Write a minimal SPARC basic table CSV and return its path."""
    if rows is None:
        rows = [
            {"Galaxy": "NGC1234", "L36": 4.0,  "MHI": 2.0},
            {"Galaxy": "UGC5678", "L36": 1.5,  "MHI": 0.8},
            {"Galaxy": "IC910",   "L36": 2.0,  "MHI": 1.0},
        ]
    p = tmp_path / "sparc_basic.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _write_chae(tmp_path: Path, rows: list[dict] | None = None,
                with_err: bool = False) -> Path:
    """Write a minimal Chae environment catalog CSV and return its path."""
    if rows is None:
        rows = [
            {"galaxy_name": "NGC1234", "e_env": -0.3},
            {"galaxy_name": "UGC5678", "e_env":  0.1},
            {"galaxy_name": "IC910",   "e_env":  0.4},
        ]
    if with_err:
        for r in rows:
            r.setdefault("e_env_err", 0.05)
    p = tmp_path / "chae_env.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. clean_name()
# ---------------------------------------------------------------------------

class TestCleanName:
    def test_lowercase(self):
        assert clean_name("NGC1234") == "ngc1234"

    def test_strips_leading_trailing_spaces(self):
        assert clean_name("  ngc1234  ") == "ngc1234"

    def test_collapses_internal_whitespace(self):
        assert clean_name("NGC  1234") == "ngc1234"

    def test_removes_hyphens(self):
        assert clean_name("NGC-1234") == "ngc1234"

    def test_removes_underscores(self):
        assert clean_name("NGC_1234") == "ngc1234"

    def test_removes_dots(self):
        assert clean_name("NGC.1234") == "ngc1234"

    def test_identical_after_normalisation(self):
        assert clean_name("UGC 5678") == clean_name("ugc5678")

    def test_numeric_string(self):
        assert clean_name("  42  ") == "42"


# ---------------------------------------------------------------------------
# 2. load_f3_catalog()
# ---------------------------------------------------------------------------

class TestLoadF3Catalog:
    def test_loads_f3_column(self, tmp_path):
        p = _write_f3(tmp_path)
        df = load_f3_catalog(p)
        assert "F3" in df.columns
        assert "galaxy" in df.columns

    def test_accepts_beta_column(self, tmp_path):
        p = tmp_path / "f3.csv"
        pd.DataFrame({"galaxy": ["NGC1"], "beta": [0.5]}).to_csv(p, index=False)
        df = load_f3_catalog(p)
        assert "F3" in df.columns
        assert df["F3"].iloc[0] == pytest.approx(0.5)

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_f3_catalog(tmp_path / "missing.csv")

    def test_raises_missing_galaxy_column(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"name": ["A"], "F3": [0.5]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="galaxy"):
            load_f3_catalog(p)

    def test_raises_missing_f3_and_beta_column(self, tmp_path):
        p = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"], "slope": [0.5]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="F3"):
            load_f3_catalog(p)

    def test_returns_only_galaxy_and_f3_columns(self, tmp_path):
        p = tmp_path / "f3.csv"
        pd.DataFrame({
            "galaxy": ["A"], "F3": [0.5], "extra": [99]
        }).to_csv(p, index=False)
        df = load_f3_catalog(p)
        assert set(df.columns) == {"galaxy", "F3"}


# ---------------------------------------------------------------------------
# 3. load_sparc_basic()
# ---------------------------------------------------------------------------

class TestLoadSparcBasic:
    def test_loads_required_columns(self, tmp_path):
        p = _write_sparc(tmp_path)
        df = load_sparc_basic(p)
        assert {"Galaxy", "L36", "MHI"}.issubset(df.columns)

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_sparc_basic(tmp_path / "missing.csv")

    def test_raises_missing_column(self, tmp_path):
        p = tmp_path / "sparc.csv"
        pd.DataFrame({"Galaxy": ["A"], "L36": [1.0]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="MHI"):
            load_sparc_basic(p)

    def test_returns_only_three_columns(self, tmp_path):
        p = tmp_path / "sparc.csv"
        pd.DataFrame({
            "Galaxy": ["A"], "L36": [1.0], "MHI": [0.5], "extra": [9]
        }).to_csv(p, index=False)
        df = load_sparc_basic(p)
        assert set(df.columns) == {"Galaxy", "L36", "MHI"}

    def test_row_count_preserved(self, tmp_path):
        p = _write_sparc(tmp_path)
        df = load_sparc_basic(p)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# 4. load_chae_env()
# ---------------------------------------------------------------------------

class TestLoadChaeEnv:
    def test_loads_required_columns(self, tmp_path):
        p = _write_chae(tmp_path)
        df = load_chae_env(p)
        assert {"galaxy_name", "e_env"}.issubset(df.columns)

    def test_includes_e_env_err_when_present(self, tmp_path):
        p = _write_chae(tmp_path, with_err=True)
        df = load_chae_env(p)
        assert "e_env_err" in df.columns

    def test_excludes_e_env_err_when_absent(self, tmp_path):
        p = _write_chae(tmp_path, with_err=False)
        df = load_chae_env(p)
        assert "e_env_err" not in df.columns

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_chae_env(tmp_path / "missing.csv")

    def test_raises_missing_e_env_column(self, tmp_path):
        p = tmp_path / "chae.csv"
        pd.DataFrame({"galaxy_name": ["A"]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="e_env"):
            load_chae_env(p)

    def test_row_count_preserved(self, tmp_path):
        p = _write_chae(tmp_path)
        df = load_chae_env(p)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# 5. compute_logM()
# ---------------------------------------------------------------------------

class TestComputeLogM:
    def test_formula_correctness(self):
        L36, MHI = 4.0, 2.0
        expected = math.log10(UPSILON_36 * L36 * 1e9 + ALPHA_HI * MHI * 1e9)
        result = compute_logM(np.array([L36]), np.array([MHI]))
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_array_input(self):
        L36 = np.array([1.0, 2.0, 4.0])
        MHI = np.array([0.5, 1.0, 2.0])
        result = compute_logM(L36, MHI)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_zero_inputs_returns_finite(self):
        result = compute_logM(np.array([0.0]), np.array([0.0]))
        assert np.isfinite(result[0])

    def test_monotone_in_L36(self):
        MHI = np.full(5, 1.0)
        L36 = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        logM = compute_logM(L36, MHI)
        assert np.all(np.diff(logM) > 0)

    def test_monotone_in_MHI(self):
        L36 = np.full(5, 1.0)
        MHI = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
        logM = compute_logM(L36, MHI)
        assert np.all(np.diff(logM) > 0)

    def test_physical_range(self):
        L36 = np.array([0.1, 1.0, 10.0, 100.0])
        MHI = np.array([0.1, 0.5, 5.0,  50.0])
        logM = compute_logM(L36, MHI)
        # Expect baryonic masses roughly between 10^7 and 10^12 M_sun
        assert np.all(logM > 6.0)
        assert np.all(logM < 14.0)


# ---------------------------------------------------------------------------
# 6. merge_catalogs()
# ---------------------------------------------------------------------------

class TestMergeCatalogs:
    def _make_frames(self):
        df_f3 = pd.DataFrame({
            "galaxy": ["NGC1234", "UGC5678", "IC910"],
            "F3":     [0.52,      0.48,       0.50],
        })
        df_sparc = pd.DataFrame({
            "Galaxy": ["NGC1234", "UGC5678", "IC910"],
            "L36":    [4.0, 1.5, 2.0],
            "MHI":    [2.0, 0.8, 1.0],
        })
        df_chae = pd.DataFrame({
            "galaxy_name": ["NGC1234", "UGC5678", "IC910"],
            "e_env":       [-0.3, 0.1, 0.4],
        })
        return df_f3, df_sparc, df_chae

    def test_returns_dataframe(self):
        df_f3, df_sparc, df_chae = self._make_frames()
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        assert isinstance(result, pd.DataFrame)

    def test_output_columns_no_err(self):
        df_f3, df_sparc, df_chae = self._make_frames()
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        assert set(result.columns) == {"galaxy_name", "delta_f3", "logM", "e_env"}

    def test_output_columns_with_err(self):
        df_f3, df_sparc, df_chae = self._make_frames()
        df_chae["e_env_err"] = 0.05
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        assert "e_env_err" in result.columns

    def test_row_count_full_overlap(self):
        df_f3, df_sparc, df_chae = self._make_frames()
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        assert len(result) == 3

    def test_row_count_partial_overlap(self):
        df_f3 = pd.DataFrame({
            "galaxy": ["NGC1234", "UGC5678"],
            "F3":     [0.52,       0.48],
        })
        df_sparc = pd.DataFrame({
            "Galaxy": ["NGC1234", "UGC5678", "IC910"],
            "L36": [4.0, 1.5, 2.0],
            "MHI": [2.0, 0.8, 1.0],
        })
        df_chae = pd.DataFrame({
            "galaxy_name": ["NGC1234", "IC910"],
            "e_env": [-0.3, 0.4],
        })
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        assert len(result) == 1
        assert result["galaxy_name"].iloc[0] == "ngc1234"

    def test_delta_f3_equals_f3_minus_half(self):
        df_f3, df_sparc, df_chae = self._make_frames()
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        row = result[result["galaxy_name"] == "ngc1234"].iloc[0]
        assert row["delta_f3"] == pytest.approx(0.52 - MOND_REF, rel=1e-9)

    def test_name_normalisation_enables_match(self):
        df_f3 = pd.DataFrame({"galaxy": ["NGC-1234"], "F3": [0.50]})
        df_sparc = pd.DataFrame({"Galaxy": ["NGC 1234"], "L36": [2.0], "MHI": [1.0]})
        df_chae = pd.DataFrame({"galaxy_name": ["ngc1234"], "e_env": [0.2]})
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        assert len(result) == 1

    def test_empty_result_when_no_overlap(self):
        df_f3 = pd.DataFrame({"galaxy": ["NGC1"], "F3": [0.5]})
        df_sparc = pd.DataFrame({"Galaxy": ["NGC2"], "L36": [1.0], "MHI": [0.5]})
        df_chae = pd.DataFrame({"galaxy_name": ["NGC3"], "e_env": [0.0]})
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        assert len(result) == 0

    def test_logM_finite(self):
        df_f3, df_sparc, df_chae = self._make_frames()
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        assert result["logM"].notna().all()
        assert np.isfinite(result["logM"].values).all()

    def test_galaxy_name_is_normalised(self):
        df_f3, df_sparc, df_chae = self._make_frames()
        result = merge_catalogs(df_f3, df_sparc, df_chae)
        for name in result["galaxy_name"]:
            assert name == clean_name(name)


# ---------------------------------------------------------------------------
# 7. build_env_real_input()
# ---------------------------------------------------------------------------

class TestBuildEnvRealInput:
    def test_writes_csv(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "out" / "merged.csv"
        build_env_real_input(f3_p, sp_p, ch_p, out)
        assert out.exists()

    def test_creates_parent_directory(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "deep" / "nested" / "merged.csv"
        build_env_real_input(f3_p, sp_p, ch_p, out)
        assert out.exists()

    def test_returns_dataframe(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "merged.csv"
        result = build_env_real_input(f3_p, sp_p, ch_p, out)
        assert isinstance(result, pd.DataFrame)

    def test_csv_readable_and_has_required_columns(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "merged.csv"
        build_env_real_input(f3_p, sp_p, ch_p, out)
        df = pd.read_csv(out)
        assert {"galaxy_name", "delta_f3", "logM", "e_env"}.issubset(df.columns)

    def test_row_count_matches_overlap(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "merged.csv"
        df = build_env_real_input(f3_p, sp_p, ch_p, out)
        assert len(df) == 3

    def test_includes_e_env_err_when_present(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path, with_err=True)
        out = tmp_path / "merged.csv"
        df = build_env_real_input(f3_p, sp_p, ch_p, out)
        assert "e_env_err" in df.columns


# ---------------------------------------------------------------------------
# 8. main() — CLI and keyword-argument API
# ---------------------------------------------------------------------------

class TestMain:
    def test_keyword_args_run_without_argv(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "merged.csv"
        result = main(
            f3_catalog=str(f3_p),
            sparc_basic=str(sp_p),
            chae_env=str(ch_p),
            out=str(out),
        )
        assert isinstance(result, pd.DataFrame)
        assert out.exists()

    def test_argv_list_works(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "merged.csv"
        result = main([
            "--f3-catalog", str(f3_p),
            "--sparc-basic", str(sp_p),
            "--chae-env", str(ch_p),
            "--out", str(out),
        ])
        assert isinstance(result, pd.DataFrame)

    def test_keyword_args_override_argv(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out_argv = tmp_path / "argv_out.csv"
        out_kwarg = tmp_path / "kwarg_out.csv"
        main(
            ["--f3-catalog", str(f3_p),
             "--sparc-basic", str(sp_p),
             "--chae-env", str(ch_p),
             "--out", str(out_argv)],
            out=str(out_kwarg),
        )
        assert out_kwarg.exists()
        assert not out_argv.exists()

    def test_missing_f3_catalog_raises(self, tmp_path):
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "out.csv"
        with pytest.raises(ValueError, match="--f3-catalog"):
            main(
                sparc_basic=str(sp_p),
                chae_env=str(ch_p),
                out=str(out),
            )

    def test_missing_sparc_basic_raises(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "out.csv"
        with pytest.raises(ValueError, match="--sparc-basic"):
            main(
                f3_catalog=str(f3_p),
                chae_env=str(ch_p),
                out=str(out),
            )

    def test_missing_chae_env_raises(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        out = tmp_path / "out.csv"
        with pytest.raises(ValueError, match="--chae-env"):
            main(
                f3_catalog=str(f3_p),
                sparc_basic=str(sp_p),
                out=str(out),
            )

    def test_default_out_path_used_when_no_kwarg(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        # --out is not supplied; argparse default is used
        result = main([
            "--f3-catalog", str(f3_p),
            "--sparc-basic", str(sp_p),
            "--chae-env", str(ch_p),
        ])
        assert isinstance(result, pd.DataFrame)
        # Clean up the default output location
        default_out = Path("results/env_real/sparc_f3_chae_merged.csv")
        if default_out.exists():
            default_out.unlink()

    def test_returns_correct_row_count(self, tmp_path):
        f3_p = _write_f3(tmp_path)
        sp_p = _write_sparc(tmp_path)
        ch_p = _write_chae(tmp_path)
        out = tmp_path / "merged.csv"
        result = main(
            f3_catalog=str(f3_p),
            sparc_basic=str(sp_p),
            chae_env=str(ch_p),
            out=str(out),
        )
        assert len(result) == 3
