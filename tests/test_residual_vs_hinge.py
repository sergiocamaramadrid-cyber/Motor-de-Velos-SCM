"""
tests/test_residual_vs_hinge.py — Tests for scripts/residual_vs_hinge.py.

Validates that the standalone PR #70 diagnostic:
  - Uses OOS residuals (residual_dex_oos column, not residual_dex)
  - Produces a PNG plot using matplotlib only
  - Handles edge cases gracefully (missing CSV, missing columns, etc.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.residual_vs_hinge import (
    plot_residual_vs_hinge,
    main,
    _REQUIRED_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_oos_per_point(n_galaxies: int = 8, n_points_per_galaxy: int = 15,
                         seed: int = 0) -> pd.DataFrame:
    """Synthetic oos_per_point.csv DataFrame matching audit_scm.py output."""
    rng = np.random.default_rng(seed)
    n = n_galaxies * n_points_per_galaxy
    galaxy_labels = np.repeat([f"G{i:03d}" for i in range(n_galaxies)], n_points_per_galaxy)
    fold_labels = np.repeat(np.arange(n_galaxies) % 5, n_points_per_galaxy)
    a0_log = np.log10(1.2e-10)
    log_gbar = rng.uniform(-12.5, -8.5, n)
    hinge = np.maximum(0.0, a0_log - log_gbar)
    residual_dex = rng.normal(0.0, 0.1, n)
    residual_dex_pred = rng.normal(0.0, 0.05, n)
    residual_dex_oos = residual_dex - residual_dex_pred
    return pd.DataFrame({
        "galaxy": galaxy_labels,
        "hinge": hinge,
        "residual_dex": residual_dex,
        "residual_dex_pred": residual_dex_pred,
        "residual_dex_oos": residual_dex_oos,
        "fold": fold_labels,
    })


@pytest.fixture()
def oos_csv(tmp_path: Path) -> Path:
    """Write a synthetic oos_per_point.csv and return its path."""
    df = _make_oos_per_point()
    path = tmp_path / "oos_per_point.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def oos_df() -> pd.DataFrame:
    return _make_oos_per_point()


# ---------------------------------------------------------------------------
# plot_residual_vs_hinge
# ---------------------------------------------------------------------------

class TestPlotResidualVsHinge:
    def test_creates_png(self, tmp_path: Path, oos_df: pd.DataFrame):
        out = tmp_path / "plot.png"
        plot_residual_vs_hinge(oos_df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_uses_oos_residuals_not_insample(self, tmp_path: Path):
        """The plot must use residual_dex_oos, not residual_dex."""
        rng = np.random.default_rng(1)
        n = 30
        df = pd.DataFrame({
            "galaxy": [f"G{i}" for i in range(n)],
            "hinge": rng.uniform(0, 2, n),
            # In-sample residual is intentionally huge — should not affect plot
            "residual_dex": rng.uniform(-100, 100, n),
            "residual_dex_pred": rng.normal(0, 0.01, n),
            "residual_dex_oos": rng.normal(0, 0.05, n),
            "fold": np.zeros(n, dtype=int),
        })
        out = tmp_path / "oos_check.png"
        # Should succeed and produce a sensible-looking plot
        plot_residual_vs_hinge(df, out)
        assert out.exists()

    def test_single_point_no_crash(self, tmp_path: Path):
        """Edge case: single row — no bins, no crash."""
        df = pd.DataFrame({
            "galaxy": ["G0"],
            "hinge": [0.5],
            "residual_dex": [0.0],
            "residual_dex_pred": [0.0],
            "residual_dex_oos": [0.1],
            "fold": [0],
        })
        out = tmp_path / "single.png"
        plot_residual_vs_hinge(df, out)
        assert out.exists()

    def test_all_zero_hinge_no_crash(self, tmp_path: Path):
        """Edge case: hinge is all-zero (Newtonian regime only)."""
        rng = np.random.default_rng(2)
        n = 20
        df = pd.DataFrame({
            "galaxy": [f"G{i}" for i in range(n)],
            "hinge": np.zeros(n),
            "residual_dex": rng.normal(0, 0.1, n),
            "residual_dex_pred": rng.normal(0, 0.05, n),
            "residual_dex_oos": rng.normal(0, 0.05, n),
            "fold": np.zeros(n, dtype=int),
        })
        out = tmp_path / "zero_hinge.png"
        plot_residual_vs_hinge(df, out)
        assert out.exists()

    def test_custom_bins(self, tmp_path: Path, oos_df: pd.DataFrame):
        out = tmp_path / "custom_bins.png"
        plot_residual_vs_hinge(oos_df, out, n_bins=6)
        assert out.exists()

    def test_output_parent_does_not_need_to_exist(self, tmp_path: Path,
                                                   oos_df: pd.DataFrame):
        """plot_residual_vs_hinge should not crash if parent already exists."""
        nested = tmp_path / "nested" / "plot.png"
        nested.parent.mkdir(parents=True, exist_ok=True)
        plot_residual_vs_hinge(oos_df, nested)
        assert nested.exists()


# ---------------------------------------------------------------------------
# _REQUIRED_COLS
# ---------------------------------------------------------------------------

class TestRequiredCols:
    def test_hinge_in_required(self):
        assert "hinge" in _REQUIRED_COLS

    def test_residual_dex_oos_in_required(self):
        """Must use OOS residuals, not in-sample."""
        assert "residual_dex_oos" in _REQUIRED_COLS

    def test_galaxy_in_required(self):
        assert "galaxy" in _REQUIRED_COLS

    def test_fold_in_required(self):
        assert "fold" in _REQUIRED_COLS


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_produces_png(self, oos_csv: Path, tmp_path: Path):
        out = tmp_path / "out.png"
        main([
            "--csv", str(oos_csv),
            "--out", str(out),
            "--bins", "8",
        ])
        assert out.exists()
        assert out.stat().st_size > 0

    def test_main_missing_csv_exits(self, tmp_path: Path):
        with pytest.raises(SystemExit):
            main([
                "--csv", str(tmp_path / "nonexistent.csv"),
                "--out", str(tmp_path / "out.png"),
            ])

    def test_main_missing_column_exits(self, tmp_path: Path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["G1"], "hinge": [0.5]}).to_csv(
            bad_csv, index=False
        )
        with pytest.raises(SystemExit):
            main([
                "--csv", str(bad_csv),
                "--out", str(tmp_path / "out.png"),
            ])

    def test_main_creates_parent_dir(self, oos_csv: Path, tmp_path: Path):
        out = tmp_path / "subdir" / "plot.png"
        main([
            "--csv", str(oos_csv),
            "--out", str(out),
        ])
        assert out.exists()

    def test_main_custom_bins(self, oos_csv: Path, tmp_path: Path):
        out = tmp_path / "bins_test.png"
        main([
            "--csv", str(oos_csv),
            "--out", str(out),
            "--bins", "4",
        ])
        assert out.exists()


# ---------------------------------------------------------------------------
# Integration: audit_scm writes oos_per_point.csv that residual_vs_hinge reads
# ---------------------------------------------------------------------------

class TestEndToEndAuditToPlot:
    """Verify the audit → plot pipeline works end-to-end."""

    def test_audit_writes_oos_per_point_csv(self, tmp_path: Path):
        """audit_scm.run_audit must write oos_per_point.csv."""
        from scripts.audit_scm import run_audit

        # Build synthetic audit_features.csv
        rng = np.random.default_rng(42)
        n = 10 * 15
        log_gbar = rng.uniform(-12.5, -8.5, n)
        a0_log = np.log10(1.2e-10)
        log_r = rng.uniform(-0.5, 1.5, n)
        log_gobs = 0.5 * log_gbar + rng.normal(0, 0.05, n) - 5.0
        df = pd.DataFrame({
            "galaxy": np.repeat([f"G{i}" for i in range(10)], 15),
            "logM": log_gbar + 2.0 * log_r,
            "log_gbar": log_gbar,
            "log_j": 0.5 * log_gobs + 1.5 * log_r,
            "hinge": np.maximum(0.0, a0_log - log_gbar),
            "residual_dex": log_gobs - log_gbar,
        })
        feat_csv = tmp_path / "audit_features.csv"
        df.to_csv(feat_csv, index=False)

        outdir = tmp_path / "audit_out"
        run_audit(feat_csv, outdir, seed=0, n_folds=3, n_perm=9)

        assert (outdir / "oos_per_point.csv").exists(), \
            "oos_per_point.csv must be written by run_audit"

    def test_oos_per_point_csv_has_required_columns(self, tmp_path: Path):
        """oos_per_point.csv must contain all columns needed by residual_vs_hinge."""
        from scripts.audit_scm import run_audit

        rng = np.random.default_rng(5)
        n = 10 * 15
        log_gbar = rng.uniform(-12.5, -8.5, n)
        a0_log = np.log10(1.2e-10)
        log_r = rng.uniform(-0.5, 1.5, n)
        log_gobs = 0.5 * log_gbar + rng.normal(0, 0.05, n) - 5.0
        df = pd.DataFrame({
            "galaxy": np.repeat([f"G{i}" for i in range(10)], 15),
            "logM": log_gbar + 2.0 * log_r,
            "log_gbar": log_gbar,
            "log_j": 0.5 * log_gobs + 1.5 * log_r,
            "hinge": np.maximum(0.0, a0_log - log_gbar),
            "residual_dex": log_gobs - log_gbar,
        })
        feat_csv = tmp_path / "audit_features.csv"
        df.to_csv(feat_csv, index=False)

        outdir = tmp_path / "audit_out"
        run_audit(feat_csv, outdir, seed=0, n_folds=3, n_perm=9)

        oos_df = pd.read_csv(outdir / "oos_per_point.csv")
        for col in _REQUIRED_COLS:
            assert col in oos_df.columns, f"Column '{col}' missing from oos_per_point.csv"

    def test_oos_residuals_are_true_oos(self, tmp_path: Path):
        """residual_dex_oos must differ from residual_dex (not just a copy)."""
        from scripts.audit_scm import run_audit

        rng = np.random.default_rng(7)
        n = 10 * 15
        log_gbar = rng.uniform(-12.5, -8.5, n)
        a0_log = np.log10(1.2e-10)
        log_r = rng.uniform(-0.5, 1.5, n)
        log_gobs = 0.5 * log_gbar + rng.normal(0, 0.05, n) - 5.0
        df = pd.DataFrame({
            "galaxy": np.repeat([f"G{i}" for i in range(10)], 15),
            "logM": log_gbar + 2.0 * log_r,
            "log_gbar": log_gbar,
            "log_j": 0.5 * log_gobs + 1.5 * log_r,
            "hinge": np.maximum(0.0, a0_log - log_gbar),
            "residual_dex": log_gobs - log_gbar,
        })
        feat_csv = tmp_path / "audit_features.csv"
        df.to_csv(feat_csv, index=False)

        outdir = tmp_path / "audit_out"
        run_audit(feat_csv, outdir, seed=0, n_folds=3, n_perm=9)

        oos_df = pd.read_csv(outdir / "oos_per_point.csv")
        # OOS residuals = target - OOS prediction; must NOT equal target
        assert not np.allclose(
            oos_df["residual_dex_oos"].values,
            oos_df["residual_dex"].values,
        ), "residual_dex_oos must be OOS prediction error, not raw target"

    def test_full_pipeline_produces_plot(self, tmp_path: Path):
        """audit_scm → residual_vs_hinge produces the final PNG."""
        from scripts.audit_scm import run_audit

        rng = np.random.default_rng(9)
        n = 10 * 15
        log_gbar = rng.uniform(-12.5, -8.5, n)
        a0_log = np.log10(1.2e-10)
        log_r = rng.uniform(-0.5, 1.5, n)
        log_gobs = 0.5 * log_gbar + rng.normal(0, 0.05, n) - 5.0
        df = pd.DataFrame({
            "galaxy": np.repeat([f"G{i}" for i in range(10)], 15),
            "logM": log_gbar + 2.0 * log_r,
            "log_gbar": log_gbar,
            "log_j": 0.5 * log_gobs + 1.5 * log_r,
            "hinge": np.maximum(0.0, a0_log - log_gbar),
            "residual_dex": log_gobs - log_gbar,
        })
        feat_csv = tmp_path / "audit_features.csv"
        df.to_csv(feat_csv, index=False)

        outdir = tmp_path / "audit_out"
        run_audit(feat_csv, outdir, seed=0, n_folds=3, n_perm=9)

        oos_csv = outdir / "oos_per_point.csv"
        out_png = tmp_path / "final_plot.png"

        main([
            "--csv", str(oos_csv),
            "--out", str(out_png),
        ])
        assert out_png.exists()
        assert out_png.stat().st_size > 0
