"""
tests/test_deep_slope.py — Tests for scripts/deep_slope_test.py.

Uses synthetic per-radial-point data with known slope to verify the
deep-slope computation is correct.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.deep_slope_test import (
    deep_slope,
    format_report,
    main,
    EXPECTED_SLOPE,
    G0_DEFAULT,
    DEEP_THRESHOLD_DEFAULT,
    MIN_DEEP_POINTS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mond_csv(tmp_path: Path, n_deep: int = 50, n_shallow: int = 50,
                   g0: float = G0_DEFAULT,
                   noise_std: float = 0.005,
                   planted_slope: float = 0.5) -> Path:
    """Create a synthetic per-point CSV with known slope in deep regime.

    Deep points: g_bar = uniform in [0.001·g0, 0.3·g0]
    Shallow points: g_bar = uniform in [g0, 10·g0]
    log_g_obs = planted_slope·log_g_bar + 0.5·log(g0) + Gaussian noise

    Note: this helper constructs MOND-compliant data directly in (log_g_bar,
    log_g_obs) space.  Flat-rotation-curve synthetic data from run_pipeline()
    will produce slope ≈ 1.0 in the deep regime because both g_obs and g_bar
    scale as V²/r with the same V → the ratio is constant, not the MOND
    sqrt(g_bar·g0) relation.  Real SPARC LSB galaxies follow the MOND form.
    """
    rng = np.random.default_rng(42)

    g_bar_deep = rng.uniform(0.001 * g0, 0.29 * g0, n_deep)
    g_bar_shallow = rng.uniform(g0, 10.0 * g0, n_shallow)
    g_bar_all = np.concatenate([g_bar_deep, g_bar_shallow])

    # Pure MOND: g_obs = sqrt(g_bar * g0) → log_g_obs = 0.5*log_g_bar + 0.5*log(g0)
    log_g_bar = np.log10(g_bar_all)
    log_g_obs = (planted_slope * log_g_bar
                 + 0.5 * np.log10(g0)
                 + rng.normal(0, noise_std, len(g_bar_all)))

    df = pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(len(g_bar_all))],
        "r_kpc": rng.uniform(1.0, 20.0, len(g_bar_all)),
        "g_bar_SCM": g_bar_all,
        "g_obs_SCM": 10.0 ** log_g_obs,
        "log_g_bar_SCM": log_g_bar,
        "log_g_obs_SCM": log_g_obs,
    })
    p = tmp_path / "universal_term_comparison_full.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Unit tests for deep_slope()
# ---------------------------------------------------------------------------

class TestDeepSlopeFunction:
    def test_returns_required_keys(self):
        rng = np.random.default_rng(0)
        log_gbar = np.log10(rng.uniform(1e-12, 1e-9, 100))
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(G0_DEFAULT)
        result = deep_slope(log_gbar, log_gobs)
        required = {"n_total", "n_deep", "deep_frac", "slope", "intercept",
                    "stderr", "r_value", "p_value", "delta_from_mond",
                    "verdict", "log_g0_pred"}
        assert required.issubset(set(result.keys()))

    def test_exact_mond_slope(self):
        """Pure MOND data must recover slope ≈ 0.5."""
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 200)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(g0)
        result = deep_slope(log_gbar, log_gobs, g0=g0)
        assert result["slope"] == pytest.approx(0.5, abs=1e-4)
        assert result["delta_from_mond"] == pytest.approx(0.0, abs=1e-4)

    def test_slope_recovers_planted_value(self):
        """Test that a planted slope of 0.44 is recovered."""
        rng = np.random.default_rng(7)
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 300)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.44 * log_gbar + 0.5 * np.log10(g0) + rng.normal(0, 0.001, 300)
        result = deep_slope(log_gbar, log_gobs, g0=g0)
        assert result["slope"] == pytest.approx(0.44, abs=0.02)

    def test_nan_when_zero_deep_points(self):
        """Slope must be NaN when no deep-regime points exist."""
        g0 = G0_DEFAULT
        # All points are in the Newtonian regime
        g_bar = np.linspace(2.0 * g0, 10.0 * g0, 50)
        log_gbar = np.log10(g_bar)
        log_gobs = log_gbar  # trivial
        result = deep_slope(log_gbar, log_gobs, g0=g0)
        assert np.isnan(result["slope"])
        assert result["n_deep"] == 0

    def test_n_deep_matches_mask(self):
        """n_deep must equal count of points with g_bar < threshold * g0."""
        g0 = G0_DEFAULT
        threshold = 0.3
        rng = np.random.default_rng(1)
        g_bar = rng.uniform(0.0 * g0, 2.0 * g0, 100)
        expected_deep = int((g_bar < threshold * g0).sum())
        log_gbar = np.log10(np.maximum(g_bar, 1e-30))
        log_gobs = 0.5 * log_gbar
        result = deep_slope(log_gbar, log_gobs, g0=g0, deep_threshold=threshold)
        assert result["n_deep"] == expected_deep

    def test_implied_g0_from_mond_data(self):
        """Under pure MOND the implied g0 should match the planted value."""
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 500)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(g0)
        result = deep_slope(log_gbar, log_gobs, g0=g0)
        # log10(g0_pred) should be close to log10(G0_DEFAULT)
        assert result["log_g0_pred"] == pytest.approx(np.log10(g0), abs=0.01)

    def test_deep_frac_between_zero_and_one(self):
        rng = np.random.default_rng(99)
        g0 = G0_DEFAULT
        g_bar = rng.uniform(0.001 * g0, 5.0 * g0, 200)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar
        result = deep_slope(log_gbar, log_gobs, g0=g0)
        assert 0.0 <= result["deep_frac"] <= 1.0


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_contains_beta_line(self):
        g0 = G0_DEFAULT
        g_bar = np.logspace(np.log10(0.001 * g0), np.log10(0.29 * g0), 100)
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(g0)
        result = deep_slope(log_gbar, log_gobs)
        lines = format_report(result, g0, DEEP_THRESHOLD_DEFAULT, "test.csv")
        combined = "\n".join(lines)
        assert "Slope β" in combined
        assert "Expected (MOND)" in combined

    def test_warning_when_no_deep_points(self):
        result = {
            "n_total": 100, "n_deep": 0, "deep_frac": 0.0,
            "slope": float("nan"), "intercept": float("nan"),
            "stderr": float("nan"), "r_value": float("nan"),
            "p_value": float("nan"), "delta_from_mond": float("nan"),
            "verdict": "⚠️  Insufficient deep-regime points",
            "log_g0_pred": float("nan"),
        }
        lines = format_report(result, G0_DEFAULT, 0.3, "test.csv")
        combined = "\n".join(lines)
        assert "Verdict" in combined


# ---------------------------------------------------------------------------
# CLI (main) integration tests
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_recovers_mond_slope(self, tmp_path):
        csv = _make_mond_csv(tmp_path, n_deep=100, n_shallow=50, planted_slope=0.5)
        result = main(["--csv", str(csv)])
        assert result["slope"] == pytest.approx(0.5, abs=0.05)

    def test_recovers_planted_slope_044(self, tmp_path):
        csv = _make_mond_csv(tmp_path, n_deep=200, n_shallow=50, planted_slope=0.44,
                              noise_std=0.001)
        result = main(["--csv", str(csv)])
        assert result["slope"] == pytest.approx(0.44, abs=0.02)

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "nonexistent.csv")])

    def test_missing_columns_raises(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"], "g_bar": [1e-11]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            main(["--csv", str(bad_csv)])

    def test_writes_output_files(self, tmp_path):
        csv = _make_mond_csv(tmp_path, n_deep=50, n_shallow=50)
        out = tmp_path / "out"
        main(["--csv", str(csv), "--out", str(out)])
        assert (out / "deep_slope_test.csv").exists()
        assert (out / "deep_slope_test.log").exists()

    def test_insufficient_deep_points(self, tmp_path):
        """When deep_threshold is tiny, n_deep=0 → slope=NaN."""
        csv = _make_mond_csv(tmp_path, n_deep=50, n_shallow=50,
                              g0=G0_DEFAULT)
        result = main(["--csv", str(csv), "--deep-threshold", "1e-10"])
        assert np.isnan(result["slope"])
        assert result["n_deep"] == 0


# ---------------------------------------------------------------------------
# Integration: run_pipeline → deep_slope_test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Verify that run_pipeline() writes log_g_bar/log_g_obs columns that
    deep_slope_test can consume."""

    @pytest.fixture(scope="class")
    def pipeline_csv(self, tmp_path_factory):
        """Run the pipeline on a 10-galaxy synthetic dataset and return
        the path to universal_term_comparison_full.csv."""
        from src.scm_analysis import run_pipeline
        import numpy as np, pandas as pd

        root = tmp_path_factory.mktemp("pipe_intg")
        rng = np.random.default_rng(5)
        n_gal, n_pts = 10, 20
        names = [f"H{i:02d}" for i in range(n_gal)]
        v_flats = np.linspace(100.0, 250.0, n_gal)

        pd.DataFrame({
            "Galaxy": names,
            "D": np.linspace(5, 40, n_gal),
            "Inc": np.linspace(30, 70, n_gal),
            "L36": 1e9 * np.arange(1, n_gal + 1, dtype=float),
            "Vflat": v_flats,
            "e_Vflat": np.full(n_gal, 5.0),
        }).to_csv(root / "SPARC_Lelli2016c.csv", index=False)

        for name, vf in zip(names, v_flats):
            r = np.linspace(0.5, 15.0, n_pts)
            rc = pd.DataFrame({
                "r": r,
                "v_obs": np.full(n_pts, vf) + rng.normal(0, 3, n_pts),
                "v_obs_err": np.full(n_pts, 5.0),
                "v_gas": 0.3 * vf * np.ones(n_pts),
                "v_disk": 0.75 * vf * np.ones(n_pts),
                "v_bul": np.zeros(n_pts),
                "SBdisk": np.zeros(n_pts),
                "SBbul": np.zeros(n_pts),
            })
            rc.to_csv(root / f"{name}_rotmod.dat", sep=" ", index=False, header=False)

        out = tmp_path_factory.mktemp("pipe_out")
        run_pipeline(root, out, verbose=False)
        return out / "universal_term_comparison_full.csv"

    def test_has_log_columns(self, pipeline_csv):
        df = pd.read_csv(pipeline_csv)
        assert "log_g_bar_SCM" in df.columns
        assert "log_g_obs_SCM" in df.columns

    def test_columns_are_finite(self, pipeline_csv):
        df = pd.read_csv(pipeline_csv)
        assert df["log_g_bar_SCM"].apply(np.isfinite).all()
        assert df["log_g_obs_SCM"].apply(np.isfinite).all()

    def test_deep_slope_consumes_csv(self, pipeline_csv):
        result = main(["--csv", str(pipeline_csv)])
        # With synthetic high-velocity data g_bar >> a0, so n_deep may be 0 —
        # either outcome is valid; key is no exception is raised.
        # Note: flat rotation curves yield slope ≈ 1.0 in the deep regime
        # (g_obs and g_bar both scale as V²/r with constant V), not the MOND
        # value of 0.5.  Real SPARC LSB galaxies follow the MOND deep form.
        assert "n_deep" in result

    def test_pipeline_csv_has_mond_slope_when_fed_mond_data(self, tmp_path_factory):
        """Direct test: CSV built from MOND-compliant (g_bar, g_obs) pairs
        must recover slope ≈ 0.5 independent of run_pipeline."""
        import numpy as np
        root = tmp_path_factory.mktemp("mond_direct")
        g0 = G0_DEFAULT
        rng = np.random.default_rng(77)
        g_bar = np.concatenate([
            rng.uniform(0.001 * g0, 0.29 * g0, 200),  # deep
            rng.uniform(g0, 5.0 * g0, 50),              # shallow
        ])
        log_gbar = np.log10(g_bar)
        log_gobs = 0.5 * log_gbar + 0.5 * np.log10(g0) + rng.normal(0, 0.003, 250)
        csv = root / "universal_term_comparison_full.csv"
        pd.DataFrame({
            "galaxy": "X", "r_kpc": 5.0,
            "g_bar_SCM": g_bar, "g_obs_SCM": 10**log_gobs,
            "log_g_bar_SCM": log_gbar, "log_g_obs_SCM": log_gobs,
        }).to_csv(csv, index=False)
        result = main(["--csv", str(csv)])
        assert result["slope"] == pytest.approx(0.5, abs=0.05)
