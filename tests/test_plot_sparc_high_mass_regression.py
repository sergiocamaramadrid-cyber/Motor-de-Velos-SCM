"""
tests/test_plot_sparc_high_mass_regression.py — Tests for
scripts/plot_sparc_high_mass_regression.py.

Covers:
  1. compute_stats()   — Spearman + OLS (HC3) statistics.
  2. generate_figure() — figure creation and PNG/PDF output.
  3. main() CLI        — end-to-end invocation and regression guard.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.plot_sparc_high_mass_regression import (
    BETA_REF,
    M_CRIT_DEFAULT,
    compute_stats,
    generate_figure,
    main,
    _parse_args,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_catalog(
    n: int = 30,
    seed: int = 0,
    logM_range: tuple[float, float] = (9.0, 11.5),
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    logM = rng.uniform(*logM_range, n)
    delta_mass_std = rng.normal(0.0, 1.0, n)
    # introduce mild negative slope so signal is present
    slope_tail = 0.5 - 0.1 * delta_mass_std + rng.normal(0.0, 0.05, n)
    return pd.DataFrame(
        {
            "galaxy": [f"NGC{i:04d}" for i in range(n)],
            "logM": logM,
            "delta_mass_std": delta_mass_std,
            "slope_tail": slope_tail,
        }
    )


def _write_catalog(df: pd.DataFrame, tmp_path: Path) -> Path:
    p = tmp_path / "catalog.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# 1. compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 20)
        y = rng.normal(0, 1, 20)
        result = compute_stats(x, y)
        assert set(result.keys()) == {
            "rho", "p_spear", "ols_slope", "ols_intercept", "ols_pval", "n"
        }

    def test_n_equals_input_length(self):
        x = np.arange(10, dtype=float)
        y = np.arange(10, dtype=float)
        assert compute_stats(x, y)["n"] == 10

    def test_perfect_positive_correlation(self):
        x = np.linspace(0, 1, 20)
        y = 2.0 * x + 1.0
        stats = compute_stats(x, y)
        assert math.isclose(stats["rho"], 1.0, abs_tol=1e-6)
        assert stats["p_spear"] < 1e-10

    def test_perfect_negative_correlation(self):
        x = np.linspace(0, 1, 20)
        y = -3.0 * x + 5.0
        stats = compute_stats(x, y)
        assert math.isclose(stats["rho"], -1.0, abs_tol=1e-6)

    def test_ols_slope_sign_matches_rho(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 40)
        y = -0.5 * x + rng.normal(0, 0.1, 40)
        stats = compute_stats(x, y)
        assert stats["rho"] < 0
        assert stats["ols_slope"] < 0

    def test_ols_slope_recovers_true_value(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, 200)
        y = 3.0 * x + rng.normal(0, 0.1, 200)
        stats = compute_stats(x, y)
        assert abs(stats["ols_slope"] - 3.0) < 0.1

    def test_pval_is_finite_float(self):
        x = np.linspace(-2, 2, 15)
        y = np.random.default_rng(99).normal(0, 1, 15)
        stats = compute_stats(x, y)
        assert math.isfinite(stats["p_spear"])
        assert math.isfinite(stats["ols_pval"])

    def test_accepts_list_input(self):
        stats = compute_stats([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        assert stats["n"] == 3

    def test_rho_bounded(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, 50)
        y = rng.normal(0, 1, 50)
        stats = compute_stats(x, y)
        assert -1.0 <= stats["rho"] <= 1.0


# ---------------------------------------------------------------------------
# 2. generate_figure
# ---------------------------------------------------------------------------


class TestGenerateFigure:
    def test_creates_png_and_pdf(self, tmp_path):
        df = _make_catalog(n=30)
        out = tmp_path / "fig.png"
        generate_figure(df, out, m_crit=9.0)
        assert out.exists()
        assert out.with_suffix(".pdf").exists()

    def test_returns_figure_object(self, tmp_path):
        import matplotlib.pyplot as plt
        df = _make_catalog(n=30)
        fig = generate_figure(df, tmp_path / "f.png", m_crit=9.0)
        assert isinstance(fig, plt.Figure)

    def test_respects_m_crit(self, tmp_path):
        df = _make_catalog(n=40, logM_range=(9.0, 11.5))
        # Only high-mass galaxies should enter the figure; no error if enough remain
        generate_figure(df, tmp_path / "f.png", m_crit=10.0)

    def test_computes_delta_f3_internally(self, tmp_path):
        """generate_figure must derive delta_f3 from slope_tail, not require it."""
        df = _make_catalog(n=25)
        assert "delta_f3" not in df.columns
        generate_figure(df, tmp_path / "f.png", m_crit=9.0)  # should not raise

    def test_raises_when_too_few_galaxies(self, tmp_path):
        df = _make_catalog(n=10, logM_range=(7.0, 9.0))  # all below 10.05
        with pytest.raises(ValueError, match="need at least 2"):
            generate_figure(df, tmp_path / "f.png", m_crit=10.05)

    def test_default_m_crit_constant(self, tmp_path):
        df = _make_catalog(n=30, logM_range=(10.0, 11.5))
        generate_figure(df, tmp_path / "f.png")  # uses M_CRIT_DEFAULT
        assert (tmp_path / "f.png").exists()

    def test_png_nonzero(self, tmp_path):
        df = _make_catalog(n=30)
        out = tmp_path / "x.png"
        generate_figure(df, out, m_crit=9.0)
        assert out.stat().st_size > 0

    def test_does_not_mutate_input_df(self, tmp_path):
        df = _make_catalog(n=25)
        cols_before = list(df.columns)
        generate_figure(df, tmp_path / "f.png", m_crit=9.0)
        assert list(df.columns) == cols_before


# ---------------------------------------------------------------------------
# 3. main() CLI
# ---------------------------------------------------------------------------


class TestMain:
    def test_returns_expected_keys(self, tmp_path):
        df = _make_catalog(n=40)
        csv = _write_catalog(df, tmp_path)
        out = tmp_path / "out.png"
        result = main(["--csv", str(csv), "--m-crit", "9.0", "--out", str(out)])
        assert set(result.keys()) == {"stats", "m_crit", "figure_path", "pdf_path"}

    def test_figure_path_exists(self, tmp_path):
        df = _make_catalog(n=40)
        csv = _write_catalog(df, tmp_path)
        out = tmp_path / "out.png"
        result = main(["--csv", str(csv), "--m-crit", "9.0", "--out", str(out)])
        assert Path(result["figure_path"]).exists()
        assert Path(result["pdf_path"]).exists()

    def test_m_crit_propagated(self, tmp_path):
        df = _make_catalog(n=40)
        csv = _write_catalog(df, tmp_path)
        out = tmp_path / "out.png"
        result = main(["--csv", str(csv), "--m-crit", "9.5", "--out", str(out)])
        assert result["m_crit"] == pytest.approx(9.5)

    def test_stats_n_matches_filtered_count(self, tmp_path):
        df = _make_catalog(n=40, logM_range=(9.0, 11.5))
        csv = _write_catalog(df, tmp_path)
        out = tmp_path / "out.png"
        m_crit = 10.0
        result = main(["--csv", str(csv), "--m-crit", str(m_crit), "--out", str(out)])
        expected_n = int((df["logM"] >= m_crit).sum())
        assert result["stats"]["n"] == expected_n

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "missing.csv")])

    def test_raises_missing_columns(self, tmp_path):
        bad_df = pd.DataFrame({"galaxy": ["A"], "logM": [10.5]})
        csv = tmp_path / "bad.csv"
        bad_df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            main(["--csv", str(csv)])

    def test_raises_too_few_high_mass(self, tmp_path):
        df = _make_catalog(n=20, logM_range=(7.0, 9.0))
        csv = _write_catalog(df, tmp_path)
        with pytest.raises(ValueError, match="need at least 2"):
            main(["--csv", str(csv), "--m-crit", "10.05"])

    def test_stats_keys(self, tmp_path):
        df = _make_catalog(n=40)
        csv = _write_catalog(df, tmp_path)
        result = main(["--csv", str(csv), "--m-crit", "9.0",
                       "--out", str(tmp_path / "f.png")])
        assert set(result["stats"].keys()) == {
            "rho", "p_spear", "ols_slope", "ols_intercept", "ols_pval", "n"
        }

    def test_beta_ref_is_0_5(self):
        assert BETA_REF == 0.5

    def test_m_crit_default_is_10_05(self):
        assert M_CRIT_DEFAULT == pytest.approx(10.05)

    # ------------------------------------------------------------------
    # Regression guard — real data
    # ------------------------------------------------------------------

    def test_real_data_rho_negative(self, tmp_path):
        """On sparc_subset.csv the signal must be negative (ρ < 0)."""
        real_csv = Path(__file__).parent.parent / "data" / "sparc_subset.csv"
        if not real_csv.exists():
            pytest.skip("sparc_subset.csv not available")
        result = main([
            "--csv", str(real_csv),
            "--m-crit", "10.05",
            "--out", str(tmp_path / "real.png"),
        ])
        assert result["stats"]["rho"] < 0

    def test_real_data_n_56(self, tmp_path):
        """High-mass subsample should contain 56 galaxies for M_CRIT=10.05."""
        real_csv = Path(__file__).parent.parent / "data" / "sparc_subset.csv"
        if not real_csv.exists():
            pytest.skip("sparc_subset.csv not available")
        result = main([
            "--csv", str(real_csv),
            "--m-crit", "10.05",
            "--out", str(tmp_path / "real.png"),
        ])
        assert result["stats"]["n"] == 56


# ---------------------------------------------------------------------------
# 4. _parse_args
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.m_crit == pytest.approx(M_CRIT_DEFAULT)
        assert args.out == "results/scm_high_mass_regression.png"

    def test_custom_m_crit(self):
        args = _parse_args(["--m-crit", "10.5"])
        assert args.m_crit == pytest.approx(10.5)

    def test_custom_out(self):
        args = _parse_args(["--out", "/tmp/fig.png"])
        assert args.out == "/tmp/fig.png"
