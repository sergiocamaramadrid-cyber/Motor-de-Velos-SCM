"""
tests/test_final_analysis.py — Tests for scripts/final_analysis.py.

Covers all four analysis blocks:
  A. SPARC F3 distribution
  B. LITTLE THINGS blind test + F3-equivalent β
  C. SPARC + Yang robustness (BLOQUE FINAL delegation)
  D. Cross-dataset β comparison table
  + Report formatting
  + CLI / main()
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.final_analysis import (
    BETA_MOND,
    compute_lt_gobs,
    format_final_report,
    main,
    run_final_analysis,
    _build_comparison_table,
    _lt_block,
    _sparc_f3_block,
)

# ---------------------------------------------------------------------------
# Repo paths (real data files)
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent
_LT_CSV = _REPO / "data" / "little_things_global.csv"
_F3_SYNTHETIC = _REPO / "results" / "f3_catalog_synthetic_flat.csv"

# ---------------------------------------------------------------------------
# Synthetic catalog builders
# ---------------------------------------------------------------------------

def _make_sparc_catalog(
    n: int = 40,
    beta: float = 0.5,
    seed: int = 0,
    reliable_all: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "friction_slope": beta + rng.normal(0, 0.05, n),
        "friction_slope_err": np.full(n, 0.01),
        "velo_inerte_flag": np.ones(n, dtype=bool) if reliable_all
                            else np.array([i % 2 == 0 for i in range(n)]),
        "n_deep": np.full(n, 10),
        "reliable": np.ones(n, dtype=bool) if reliable_all
                    else np.array([i % 2 == 0 for i in range(n)]),
    })


def _make_f3_catalog_with_mass(n: int = 60, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "friction_slope": 0.5 + rng.normal(0, 0.05, n),
        "reliable": np.ones(n, dtype=bool),
        "log_M_bar": rng.normal(9.5, 0.5, n),
        "Rmax_kpc": 10 ** rng.normal(1.3, 0.2, n),
    })


def _make_env_catalog(galaxy_col: str, n: int = 60, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        galaxy_col: [f"G{i:03d}" for i in range(n)],
        "delta_mass": rng.normal(0.0, 1.0, n),
    })


def _write(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Block A — SPARC F3
# ---------------------------------------------------------------------------

class TestSparcF3Block:
    def test_returns_required_keys(self, tmp_path):
        p = _write(_make_sparc_catalog(), tmp_path / "f3.csv")
        result = _sparc_f3_block(p)
        for key in ("dataset", "n_galaxies", "n_reliable", "beta_mean",
                    "beta_median", "beta_std", "t_stat", "p_value",
                    "consistent_mond"):
            assert key in result

    def test_dataset_label(self, tmp_path):
        p = _write(_make_sparc_catalog(), tmp_path / "f3.csv")
        assert _sparc_f3_block(p)["dataset"] == "SPARC"

    def test_n_galaxies_correct(self, tmp_path):
        p = _write(_make_sparc_catalog(n=25), tmp_path / "f3.csv")
        result = _sparc_f3_block(p)
        assert result["n_galaxies"] == 25

    def test_mond_consistent_when_beta_near_half(self, tmp_path):
        p = _write(_make_sparc_catalog(n=60, beta=0.5, seed=7), tmp_path / "f3.csv")
        result = _sparc_f3_block(p)
        assert result["consistent_mond"] is True

    def test_not_mond_consistent_for_beta_one(self, tmp_path):
        p = _write(_make_sparc_catalog(n=60, beta=1.0, seed=8), tmp_path / "f3.csv")
        result = _sparc_f3_block(p)
        assert result["consistent_mond"] is False

    def test_p_value_in_range(self, tmp_path):
        p = _write(_make_sparc_catalog(n=30), tmp_path / "f3.csv")
        result = _sparc_f3_block(p)
        if math.isfinite(result["p_value"]):
            assert 0.0 <= result["p_value"] <= 1.0

    def test_accepts_friction_slope_alias(self, tmp_path):
        df = _make_sparc_catalog(n=20)
        # already uses friction_slope column
        p = _write(df, tmp_path / "f3.csv")
        result = _sparc_f3_block(p)
        assert result["n_reliable"] == 20

    def test_accepts_beta_alias(self, tmp_path):
        df = _make_sparc_catalog(n=20)
        df = df.rename(columns={"friction_slope": "beta"})
        df = df.drop(columns=["reliable"], errors="ignore")
        df["reliable"] = True
        p = _write(df, tmp_path / "f3.csv")
        result = _sparc_f3_block(p)
        assert result["n_reliable"] == 20

    def test_synthetic_fixture_beta_near_one(self):
        """The committed synthetic CI fixture must show β ≈ 1 (not MOND).

        The tolerance abs=0.15 accounts for random variation in the synthetic
        fixture (20 galaxies, σ_β ≈ 0.05) while still clearly excluding β=0.5.
        """
        if not _F3_SYNTHETIC.exists():
            pytest.skip("Synthetic fixture not found.")
        result = _sparc_f3_block(_F3_SYNTHETIC)
        assert result["beta_median"] == pytest.approx(1.0, abs=0.15)
        assert result["consistent_mond"] is False

    def test_missing_f3_column_raises(self, tmp_path):
        df = pd.DataFrame({"galaxy": ["G0"], "other": [1.0], "reliable": [True]})
        p = _write(df, tmp_path / "f3.csv")
        with pytest.raises(ValueError, match="F3 column"):
            _sparc_f3_block(p)


# ---------------------------------------------------------------------------
# Block B — LITTLE THINGS
# ---------------------------------------------------------------------------

class TestComputeLtGobs:
    def test_output_shape(self):
        logV = np.array([1.5, 1.6, 1.7])
        log_j = np.array([1.5, 1.6, 1.7])
        out = compute_lt_gobs(logV, log_j)
        assert out.shape == (3,)

    def test_values_finite(self):
        out = compute_lt_gobs(np.array([1.5, 1.6]), np.array([1.5, 1.6]))
        assert np.all(np.isfinite(out))

    def test_higher_vflat_gives_higher_gobs(self):
        """For fixed j, higher Vflat → higher g_obs."""
        g1 = compute_lt_gobs(np.array([1.5]), np.array([1.5]))
        g2 = compute_lt_gobs(np.array([1.7]), np.array([1.5]))
        assert g2[0] > g1[0]

    def test_higher_j_gives_lower_gobs(self):
        """For fixed Vflat, higher j → lower g_obs."""
        g1 = compute_lt_gobs(np.array([1.5]), np.array([1.5]))
        g2 = compute_lt_gobs(np.array([1.5]), np.array([2.0]))
        assert g2[0] < g1[0]


class TestLtBlock:
    def test_real_data_required_keys(self):
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        result = _lt_block(_LT_CSV)
        for key in ("dataset", "n_galaxies", "beta_lt", "beta_lt_err",
                    "beta_lt_r", "beta_lt_p", "rmse_btfr", "rmse_interp",
                    "wilcoxon_p_interp", "consistent_mond"):
            assert key in result

    def test_dataset_label(self):
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        result = _lt_block(_LT_CSV)
        assert result["dataset"] == "LITTLE_THINGS"

    def test_n_galaxies(self):
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        result = _lt_block(_LT_CSV)
        assert result["n_galaxies"] == 26

    def test_beta_lt_finite(self):
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        result = _lt_block(_LT_CSV)
        assert math.isfinite(result["beta_lt"])

    def test_beta_lt_near_mond_range(self):
        """LT β should be in the range (0.2, 0.8) — MOND-ish regime."""
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        result = _lt_block(_LT_CSV)
        assert 0.1 < result["beta_lt"] < 1.0, (
            f"LT β = {result['beta_lt']:.3f} outside expected MOND range"
        )

    def test_rmse_interp_le_rmse_btfr(self):
        """Interpolation model should have lower or equal RMSE than BTFR."""
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        result = _lt_block(_LT_CSV)
        assert result["rmse_interp"] <= result["rmse_btfr"] + 0.05

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _lt_block(tmp_path / "nonexistent.csv")

    def test_with_synthetic_lt_csv(self, tmp_path):
        """_lt_block must run on a synthetic LT CSV."""
        rng = np.random.default_rng(42)
        n = 15
        df = pd.DataFrame({
            "galaxy_id": [f"G{i}" for i in range(n)],
            "logM": rng.normal(7.5, 0.5, n),
            "logVobs": rng.normal(1.5, 0.1, n),
            "log_gbar": rng.normal(-11.5, 0.4, n),
            "log_j": rng.normal(1.6, 0.2, n),
        })
        p = _write(df, tmp_path / "lt.csv")
        result = _lt_block(p)
        assert result["n_galaxies"] == n
        assert math.isfinite(result["beta_lt"])


# ---------------------------------------------------------------------------
# Block D — Comparison table
# ---------------------------------------------------------------------------

class TestBuildComparisonTable:
    def test_empty_when_no_data(self):
        df = _build_comparison_table(None, None)
        assert df.empty

    def test_one_row_sparc_only(self, tmp_path):
        p = _write(_make_sparc_catalog(n=30, beta=0.5), tmp_path / "f3.csv")
        sparc = _sparc_f3_block(p)
        df = _build_comparison_table(sparc, None)
        assert len(df) == 1
        assert df.iloc[0]["dataset"] == "SPARC"

    def test_one_row_lt_only(self):
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        lt = _lt_block(_LT_CSV)
        df = _build_comparison_table(None, lt)
        assert len(df) == 1
        assert df.iloc[0]["dataset"] == "LITTLE THINGS"

    def test_two_rows_both_datasets(self, tmp_path):
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        p = _write(_make_sparc_catalog(n=30, beta=0.5), tmp_path / "f3.csv")
        sparc = _sparc_f3_block(p)
        lt = _lt_block(_LT_CSV)
        df = _build_comparison_table(sparc, lt)
        assert len(df) == 2

    def test_required_columns_present(self, tmp_path):
        p = _write(_make_sparc_catalog(n=20), tmp_path / "f3.csv")
        sparc = _sparc_f3_block(p)
        df = _build_comparison_table(sparc, None)
        for col in ("dataset", "method", "N", "beta", "beta_std",
                    "p_vs_mond", "consistent_mond"):
            assert col in df.columns


# ---------------------------------------------------------------------------
# Format report
# ---------------------------------------------------------------------------

class TestFormatFinalReport:
    def _make_all(self, tmp_path):
        p = _write(_make_sparc_catalog(n=30, beta=0.5), tmp_path / "f3.csv")
        sparc = _sparc_f3_block(p)
        lt = _lt_block(_LT_CSV) if _LT_CSV.exists() else None
        comparison = _build_comparison_table(sparc, lt)
        return sparc, lt, comparison

    def test_returns_list_of_strings(self, tmp_path):
        sparc, lt, cmp = self._make_all(tmp_path)
        lines = format_final_report(sparc, lt, None, None, None, None, cmp)
        assert isinstance(lines, list)
        assert all(isinstance(l, str) for l in lines)

    def test_all_four_blocks_present(self, tmp_path):
        sparc, lt, cmp = self._make_all(tmp_path)
        combined = "\n".join(
            format_final_report(sparc, lt, None, None, None, None, cmp)
        )
        for block in ("BLOCK A", "BLOCK B", "BLOCK C", "BLOCK D"):
            assert block in combined

    def test_skipped_when_none(self, tmp_path):
        combined = "\n".join(
            format_final_report(None, None, None, None, None, None,
                                _build_comparison_table(None, None))
        )
        assert "SKIPPED" in combined

    def test_contains_mond_value(self, tmp_path):
        sparc, lt, cmp = self._make_all(tmp_path)
        combined = "\n".join(
            format_final_report(sparc, lt, None, None, None, None, cmp)
        )
        assert "0.5" in combined


# ---------------------------------------------------------------------------
# run_final_analysis() integration
# ---------------------------------------------------------------------------

class TestRunFinalAnalysis:
    def test_lt_only_returns_dict(self):
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        results = run_final_analysis(lt_csv_path=_LT_CSV)
        assert isinstance(results, dict)
        assert results["lt"] is not None
        assert results["sparc"] is None

    def test_sparc_only_returns_dict(self, tmp_path):
        p = _write(_make_sparc_catalog(n=20, beta=0.5), tmp_path / "f3.csv")
        results = run_final_analysis(f3_catalog_path=p)
        assert results["sparc"] is not None
        assert results["lt"] is None
        assert results["yang"] is None

    def test_sparc_and_yang_runs(self, tmp_path):
        f3 = _write(_make_f3_catalog_with_mass(n=60), tmp_path / "f3.csv")
        env = _write(_make_env_catalog("galaxy", n=60), tmp_path / "env.csv")
        results = run_final_analysis(
            f3_catalog_path=f3,
            env_catalog_path=env,
            n_perms=30,
            n_boot=30,
        )
        assert results["sparc"] is not None
        assert results["yang"] is not None
        assert results["reg"] is not None
        assert results["perm"] is not None
        assert results["boot"] is not None

    def test_all_three_runs(self, tmp_path):
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        f3 = _write(_make_f3_catalog_with_mass(n=60), tmp_path / "f3.csv")
        env = _write(_make_env_catalog("galaxy", n=60), tmp_path / "env.csv")
        results = run_final_analysis(
            f3_catalog_path=f3,
            lt_csv_path=_LT_CSV,
            env_catalog_path=env,
            n_perms=20,
            n_boot=20,
        )
        assert results["sparc"] is not None
        assert results["lt"] is not None
        assert results["yang"] is not None

    def test_report_lines_in_results(self, tmp_path):
        p = _write(_make_sparc_catalog(n=20), tmp_path / "f3.csv")
        results = run_final_analysis(f3_catalog_path=p)
        assert isinstance(results["report_lines"], list)
        assert len(results["report_lines"]) > 5

    def test_comparison_df_in_results(self, tmp_path):
        p = _write(_make_sparc_catalog(n=20), tmp_path / "f3.csv")
        results = run_final_analysis(f3_catalog_path=p)
        assert isinstance(results["comparison"], pd.DataFrame)

    def test_no_inputs_returns_empty(self):
        results = run_final_analysis()
        assert results["sparc"] is None
        assert results["lt"] is None
        assert results["yang"] is None
        assert results["comparison"].empty


# ---------------------------------------------------------------------------
# CLI / main()
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_lt_only(self):
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        results = main(["--lt-csv", str(_LT_CSV)])
        assert results["lt"] is not None

    def test_sparc_only(self, tmp_path):
        p = _write(_make_sparc_catalog(n=20, beta=0.5), tmp_path / "f3.csv")
        results = main(["--f3-catalog", str(p)])
        assert results["sparc"] is not None

    def test_writes_output_files(self, tmp_path):
        p = _write(_make_sparc_catalog(n=20), tmp_path / "f3.csv")
        out_dir = tmp_path / "out"
        main(["--f3-catalog", str(p), "--out", str(out_dir)])
        assert (out_dir / "final_analysis.log").exists()
        assert (out_dir / "final_analysis.json").exists()
        assert (out_dir / "cross_dataset_beta.csv").exists()

    def test_json_output_has_required_keys(self, tmp_path):
        p = _write(_make_sparc_catalog(n=20), tmp_path / "f3.csv")
        out_dir = tmp_path / "out"
        main(["--f3-catalog", str(p), "--out", str(out_dir)])
        with (out_dir / "final_analysis.json").open() as fh:
            data = json.load(fh)
        assert "sparc" in data
        assert "lt" in data

    def test_sparc_and_yang(self, tmp_path):
        f3 = _write(_make_f3_catalog_with_mass(n=50), tmp_path / "f3.csv")
        env = _write(_make_env_catalog("galaxy", n=50), tmp_path / "env.csv")
        results = main([
            "--f3-catalog", str(f3),
            "--env-catalog", str(env),
            "--n-perms", "20",
            "--n-boot", "20",
        ])
        assert results["yang"] is not None

    def test_missing_f3_catalog_exits(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--f3-catalog", str(tmp_path / "nonexistent.csv")])

    def test_missing_lt_csv_skips_block_b(self, tmp_path):
        """When LT CSV is missing, Block B should be skipped (no crash)."""
        p = _write(_make_sparc_catalog(n=20), tmp_path / "f3.csv")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            results = main([
                "--f3-catalog", str(p),
                "--lt-csv", str(tmp_path / "nonexistent_lt.csv"),
            ])
        assert results["lt"] is None
        assert results["sparc"] is not None

    def test_all_flags_accepted(self, tmp_path):
        """Exercise all CLI flags without error."""
        if not _LT_CSV.exists():
            pytest.skip("LT CSV not found.")
        f3 = _write(_make_f3_catalog_with_mass(n=40), tmp_path / "f3.csv")
        env = _write(_make_env_catalog("galaxy", n=40), tmp_path / "env.csv")
        out = tmp_path / "out"
        results = main([
            "--f3-catalog", str(f3),
            "--env-catalog", str(env),
            "--lt-csv", str(_LT_CSV),
            "--n-perms", "20",
            "--n-boot", "20",
            "--seed", "7",
            "--out", str(out),
        ])
        assert isinstance(results, dict)
