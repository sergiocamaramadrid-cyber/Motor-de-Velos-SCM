"""
tests/test_build_delta_f3_from_sparc.py — Tests for build_delta_f3_from_sparc.py.

Covers:
  1. compute_logMbar()      — baryonic mass calculation.
  2. compute_logRd()        — disk scale length calculation.
  3. compute_env_proxy()    — environmental proxy (delta_mass).
  4. build_catalog()        — full merge pipeline.
  5. main() CLI             — end-to-end invocation.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_delta_f3_from_sparc import (
    BETA_REF_DEFAULT,
    compute_logMbar,
    compute_logRd,
    compute_env_proxy,
    build_catalog,
    main,
    _parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_f3_catalog(tmp_path: Path, n: int = 10, beta: float = 0.6) -> Path:
    """Write a minimal F3 catalog CSV."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "galaxy": [f"G{i:02d}" for i in range(n)],
        "beta": beta + rng.normal(0, 0.02, n),
        "beta_err": np.full(n, 0.01),
        "reliable": np.ones(n, dtype=bool),
    })
    path = tmp_path / "f3_catalog.csv"
    df.to_csv(path, index=False)
    return path


def _make_sparc_table(tmp_path: Path, n: int = 10) -> Path:
    """Write a minimal SPARC-like table CSV."""
    rng = np.random.default_rng(99)
    df = pd.DataFrame({
        "Galaxy": [f"G{i:02d}" for i in range(n)],
        "L36": rng.uniform(0.5, 10.0, n),
        "MHI": rng.uniform(0.1, 5.0, n),
        "Re": rng.uniform(1.0, 10.0, n),
        "D": rng.uniform(5.0, 50.0, n),
        "Inc": rng.uniform(30.0, 80.0, n),
    })
    path = tmp_path / "sparc_table.csv"
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# 1. compute_logMbar()
# ---------------------------------------------------------------------------

class TestComputeLogMbar:
    def test_basic_computation(self):
        L36 = np.array([1.0])
        MHI = np.array([1.0])
        result = compute_logMbar(L36, MHI)
        expected = math.log10(0.5 * 1.0 * 1e9 + 1.33 * 1.0 * 1e9)
        assert result[0] == pytest.approx(expected, rel=1e-9)

    def test_zero_l36_zero_mhi_returns_nan(self):
        result = compute_logMbar(np.array([0.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_both_zero_returns_nan(self):
        # 0.5*0*1e9 + 1.33*0*1e9 = 0 → log10(0) → nan
        result = compute_logMbar(np.array([0.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_array_length_preserved(self):
        L36 = np.array([1.0, 2.0, 3.0])
        MHI = np.array([0.5, 1.0, 1.5])
        result = compute_logMbar(L36, MHI)
        assert len(result) == 3

    def test_larger_L36_gives_larger_logMbar(self):
        L36_a = np.array([1.0])
        L36_b = np.array([10.0])
        MHI = np.array([0.0])
        # With MHI=0 the 1.33*MHI term is 0, mass = 0.5*L36*1e9
        # But 0.5*0*1e9 = 0 → logMbar would be nan for MHI=0 if L36 also ~0
        MHI_pos = np.array([0.1])
        r_a = compute_logMbar(L36_a, MHI_pos)
        r_b = compute_logMbar(L36_b, MHI_pos)
        assert r_b[0] > r_a[0]

    def test_returns_ndarray(self):
        result = compute_logMbar(np.array([1.0]), np.array([1.0]))
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# 2. compute_logRd()
# ---------------------------------------------------------------------------

class TestComputeLogRd:
    def test_basic_computation(self):
        Re = np.array([1.678])
        result = compute_logRd(Re)
        # Rd = 1.678 / 1.678 = 1.0 → log10(1.0) = 0.0
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_zero_re_returns_nan(self):
        result = compute_logRd(np.array([0.0]))
        assert np.isnan(result[0])

    def test_negative_re_returns_nan(self):
        result = compute_logRd(np.array([-1.0]))
        assert np.isnan(result[0])

    def test_array_length_preserved(self):
        result = compute_logRd(np.array([1.0, 2.0, 3.0]))
        assert len(result) == 3

    def test_larger_re_gives_larger_logRd(self):
        r_small = compute_logRd(np.array([1.0]))[0]
        r_large = compute_logRd(np.array([5.0]))[0]
        assert r_large > r_small


# ---------------------------------------------------------------------------
# 3. compute_env_proxy()
# ---------------------------------------------------------------------------

class TestComputeEnvProxy:
    def test_basic_computation(self):
        # log10(MHI/L36) = log10(2/1) = log10(2)
        result = compute_env_proxy(np.array([1.0]), np.array([2.0]))
        assert result[0] == pytest.approx(math.log10(2.0), rel=1e-9)

    def test_equal_values_give_zero(self):
        result = compute_env_proxy(np.array([5.0]), np.array([5.0]))
        assert result[0] == pytest.approx(0.0, abs=1e-9)

    def test_zero_l36_returns_nan(self):
        result = compute_env_proxy(np.array([0.0]), np.array([1.0]))
        assert np.isnan(result[0])

    def test_zero_mhi_returns_nan(self):
        result = compute_env_proxy(np.array([1.0]), np.array([0.0]))
        assert np.isnan(result[0])

    def test_array_length_preserved(self):
        result = compute_env_proxy(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert len(result) == 2


# ---------------------------------------------------------------------------
# 4. build_catalog()
# ---------------------------------------------------------------------------

class TestBuildCatalog:
    def test_returns_dict_with_three_keys(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)
        result = build_catalog(f3, sparc, out_dir=tmp_path / "out")
        assert set(result.keys()) == {"catalog", "catalog_with_env", "delta_mass_proxy"}

    def test_catalog_has_required_columns(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)
        result = build_catalog(f3, sparc, out_dir=tmp_path / "out")
        df = result["catalog"]
        assert {"galaxy", "beta", "delta_f3", "logMbar", "logRd"}.issubset(df.columns)

    def test_catalog_with_env_has_delta_mass(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)
        result = build_catalog(f3, sparc, out_dir=tmp_path / "out")
        df = result["catalog_with_env"]
        assert "delta_mass" in df.columns

    def test_delta_mass_proxy_has_two_columns(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)
        result = build_catalog(f3, sparc, out_dir=tmp_path / "out")
        df = result["delta_mass_proxy"]
        assert set(df.columns) == {"galaxy", "delta_mass"}

    def test_delta_f3_equals_beta_minus_ref(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path, beta=0.7)
        sparc = _make_sparc_table(tmp_path)
        result = build_catalog(f3, sparc, out_dir=tmp_path / "out", beta_ref=0.5)
        df = result["catalog"]
        expected = df["beta"] - 0.5
        pd.testing.assert_series_equal(
            df["delta_f3"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )

    def test_custom_beta_ref(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path, beta=1.0)
        sparc = _make_sparc_table(tmp_path)
        result = build_catalog(f3, sparc, out_dir=tmp_path / "out", beta_ref=0.8)
        df = result["catalog"]
        expected_mean = pytest.approx(0.2, abs=0.1)
        assert df["delta_f3"].mean() == expected_mean

    def test_output_csv_files_written(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)
        out = tmp_path / "out"
        build_catalog(f3, sparc, out_dir=out)
        assert (out / "galaxy_catalog.csv").exists()
        assert (out / "galaxy_catalog_with_env.csv").exists()
        assert (out / "delta_mass_proxy.csv").exists()

    def test_output_csv_readable(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)
        out = tmp_path / "out"
        build_catalog(f3, sparc, out_dir=out)
        df = pd.read_csv(out / "galaxy_catalog.csv")
        assert len(df) > 0
        assert "galaxy" in df.columns

    def test_missing_f3_catalog_raises(self, tmp_path):
        sparc = _make_sparc_table(tmp_path)
        with pytest.raises(FileNotFoundError, match="F3 catalog"):
            build_catalog(tmp_path / "nonexistent.csv", sparc, out_dir=tmp_path / "out")

    def test_missing_sparc_table_raises(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        with pytest.raises(FileNotFoundError, match="SPARC table"):
            build_catalog(f3, tmp_path / "nonexistent.csv", out_dir=tmp_path / "out")

    def test_f3_catalog_missing_beta_column_raises(self, tmp_path):
        bad_f3 = tmp_path / "bad_f3.csv"
        pd.DataFrame({"galaxy": ["G0"], "other": [1.0]}).to_csv(bad_f3, index=False)
        sparc = _make_sparc_table(tmp_path)
        with pytest.raises(ValueError, match="missing"):
            build_catalog(bad_f3, sparc, out_dir=tmp_path / "out")

    def test_sparc_table_missing_l36_raises(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        bad_sparc = tmp_path / "bad_sparc.csv"
        pd.DataFrame({"Galaxy": ["G0"], "MHI": [1.0], "Re": [2.0]}).to_csv(
            bad_sparc, index=False
        )
        with pytest.raises(ValueError, match="missing"):
            build_catalog(f3, bad_sparc, out_dir=tmp_path / "out")

    def test_friction_slope_alias_accepted(self, tmp_path):
        """F3 catalog using friction_slope instead of beta must be accepted."""
        sparc = _make_sparc_table(tmp_path, n=5)
        alias_f3 = tmp_path / "alias_f3.csv"
        pd.DataFrame({
            "galaxy": [f"G{i:02d}" for i in range(5)],
            "friction_slope": [0.6] * 5,
            "reliable": [True] * 5,
        }).to_csv(alias_f3, index=False)
        result = build_catalog(alias_f3, sparc, out_dir=tmp_path / "out")
        assert len(result["catalog"]) > 0

    def test_only_inner_join_galaxies_kept(self, tmp_path):
        """Galaxies present only in one table must be dropped."""
        f3 = _make_f3_catalog(tmp_path, n=5)
        sparc_path = tmp_path / "sparc_extra.csv"
        # Extra galaxy not in f3 catalog
        pd.DataFrame({
            "Galaxy": [f"G{i:02d}" for i in range(3)] + ["EXTRA"],
            "L36": [1.0] * 4,
            "MHI": [1.0] * 4,
            "Re": [2.0] * 4,
        }).to_csv(sparc_path, index=False)
        result = build_catalog(f3, sparc_path, out_dir=tmp_path / "out")
        galaxies = set(result["catalog"]["galaxy"])
        assert "EXTRA" not in galaxies

    def test_no_nan_in_logmbar_after_valid_inputs(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)  # all positive values
        result = build_catalog(f3, sparc, out_dir=tmp_path / "out")
        df = result["catalog"]
        assert df["logMbar"].notna().all()

    def test_out_dir_created_if_missing(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)
        new_dir = tmp_path / "new" / "nested" / "out"
        build_catalog(f3, sparc, out_dir=new_dir)
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# 5. main() CLI
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_returns_dict_with_three_keys(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)
        result = main([
            "--f3-catalog", str(f3),
            "--sparc-table", str(sparc),
            "--out-dir", str(tmp_path / "out"),
        ])
        assert set(result.keys()) == {"catalog", "catalog_with_env", "delta_mass_proxy"}

    def test_kwargs_override_argv(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path)
        sparc = _make_sparc_table(tmp_path)
        result = main(
            f3_catalog=str(f3),
            sparc_table=str(sparc),
            out_dir=str(tmp_path / "out"),
        )
        assert isinstance(result["catalog"], pd.DataFrame)

    def test_default_beta_ref_is_half(self):
        args = _parse_args(["--f3-catalog", "x.csv", "--sparc-table", "y.csv"])
        assert args.beta_ref == pytest.approx(BETA_REF_DEFAULT)

    def test_custom_beta_ref_via_cli(self, tmp_path):
        f3 = _make_f3_catalog(tmp_path, beta=1.0)
        sparc = _make_sparc_table(tmp_path)
        result = main([
            "--f3-catalog", str(f3),
            "--sparc-table", str(sparc),
            "--out-dir", str(tmp_path / "out"),
            "--beta-ref", "0.8",
        ])
        df = result["catalog"]
        # All delta_f3 should be near 0.2 (1.0 - 0.8)
        assert df["delta_f3"].mean() == pytest.approx(0.2, abs=0.1)
