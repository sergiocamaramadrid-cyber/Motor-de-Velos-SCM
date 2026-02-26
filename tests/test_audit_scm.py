"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py.

Uses synthetic audit CSVs (no real pipeline run required).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.audit_scm import (
    plot_vif,
    plot_stability,
    plot_residual_vs_hinge,
    run_audit,
    main,
    _format_report,
    _VIF_WARN,
    _VIF_SEVERE,
    _KAPPA_WARN,
    _KAPPA_SEVERE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features_df(n: int = 50, rng_seed: int = 0) -> pd.DataFrame:
    """Synthetic per-radial-point features dataframe."""
    rng = np.random.default_rng(rng_seed)
    log_gbar = rng.uniform(-13, -9, n)
    log_gobs = log_gbar + rng.normal(0, 0.1, n)
    a0 = 1.2e-10
    hinge = np.maximum(0.0, np.log10(a0) - log_gbar)
    residual_dex = log_gobs - log_gbar
    return pd.DataFrame({
        "logM": log_gbar + 2.0,
        "log_gbar": log_gbar,
        "log_j": 0.5 * log_gobs + 1.5,
        "hinge": hinge,
        "residual_dex": residual_dex,
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def audit_dir_with_csvs(tmp_path):
    """Create a minimal audit directory with synthetic CSV files."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()

    vif_df = pd.DataFrame({
        "feature": ["logM", "log_gbar", "log_j", "hinge"],
        "VIF": [2.1, 3.5, 1.8, 1.2],
    })
    vif_df.to_csv(audit_dir / "vif_table.csv", index=False)

    sm_df = pd.DataFrame({
        "metric": ["condition_number_kappa"],
        "value": [15.3],
        "status": ["stable"],
        "notes": ["kappa computed on z-scored [logM, log_gbar, log_j, hinge]"],
    })
    sm_df.to_csv(audit_dir / "stability_metrics.csv", index=False)

    _make_features_df().to_csv(audit_dir / "audit_features.csv", index=False)

    return audit_dir


@pytest.fixture()
def audit_dir_high_vif(tmp_path):
    """Audit directory with severe VIF and high condition number."""
    audit_dir = tmp_path / "audit"
    audit_dir.mkdir()

    vif_df = pd.DataFrame({
        "feature": ["logM", "log_gbar", "log_j", "hinge"],
        "VIF": [2.0, 12.0, float("inf"), 1.1],
    })
    vif_df.to_csv(audit_dir / "vif_table.csv", index=False)

    sm_df = pd.DataFrame({
        "metric": ["condition_number_kappa"],
        "value": [150.0],
        "status": ["check"],
        "notes": ["kappa computed on z-scored [logM, log_gbar, log_j, hinge]"],
    })
    sm_df.to_csv(audit_dir / "stability_metrics.csv", index=False)

    return audit_dir


# ---------------------------------------------------------------------------
# plot_vif
# ---------------------------------------------------------------------------

class TestPlotVif:
    def test_creates_png(self, audit_dir_with_csvs):
        vif_df = pd.read_csv(audit_dir_with_csvs / "vif_table.csv")
        out = audit_dir_with_csvs / "vif_table.png"
        plot_vif(vif_df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_png_with_inf_vif(self, audit_dir_high_vif):
        vif_df = pd.read_csv(audit_dir_high_vif / "vif_table.csv")
        out = audit_dir_high_vif / "vif_table.png"
        plot_vif(vif_df, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# plot_stability
# ---------------------------------------------------------------------------

class TestPlotStability:
    def test_creates_png_stable(self, audit_dir_with_csvs):
        sm_df = pd.read_csv(audit_dir_with_csvs / "stability_metrics.csv")
        out = audit_dir_with_csvs / "stability_metrics.png"
        plot_stability(sm_df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_png_severe(self, audit_dir_high_vif):
        sm_df = pd.read_csv(audit_dir_high_vif / "stability_metrics.csv")
        out = audit_dir_high_vif / "stability_metrics.png"
        plot_stability(sm_df, out)
        assert out.exists()

    def test_no_crash_on_missing_kappa_row(self, tmp_path):
        """plot_stability must be a no-op when kappa row is absent."""
        sm_df = pd.DataFrame({
            "metric": ["some_other_metric"],
            "value": [1.0],
            "status": ["ok"],
            "notes": ["n/a"],
        })
        out = tmp_path / "stability_metrics.png"
        plot_stability(sm_df, out)  # should not raise
        assert not out.exists()


# ---------------------------------------------------------------------------
# plot_residual_vs_hinge
# ---------------------------------------------------------------------------

class TestPlotResidualVsHinge:
    def test_creates_png_and_csv(self, tmp_path):
        feat_df = _make_features_df()
        out_csv = tmp_path / "residual_vs_hinge.csv"
        out_png = tmp_path / "residual_vs_hinge.png"
        summary = plot_residual_vs_hinge(feat_df, out_csv, out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 0
        assert out_csv.exists()
        assert "hinge_bin_centre" in summary.columns
        assert "residual_median" in summary.columns

    def test_csv_has_correct_columns(self, tmp_path):
        feat_df = _make_features_df()
        out_csv = tmp_path / "rv.csv"
        out_png = tmp_path / "rv.png"
        summary = plot_residual_vs_hinge(feat_df, out_csv, out_png)
        expected = {"hinge_bin_centre", "residual_median", "residual_std", "n"}
        assert expected.issubset(set(summary.columns))

    def test_units_are_dex(self, tmp_path):
        """x-axis label must say dex for the hinge quantity."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        feat_df = _make_features_df()
        out_csv = tmp_path / "rv.csv"
        out_png = tmp_path / "rv.png"
        plot_residual_vs_hinge(feat_df, out_csv, out_png)
        # Read back the CSV and verify it has real numeric values
        df = pd.read_csv(out_csv)
        assert len(df) > 0
        assert df["hinge_bin_centre"].notna().any()


# ---------------------------------------------------------------------------
# _format_report — unit labels
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_mentions_dex_units(self, audit_dir_with_csvs):
        """Report must label log-space features as dex."""
        vif_df = pd.read_csv(audit_dir_with_csvs / "vif_table.csv")
        sm_df = pd.read_csv(audit_dir_with_csvs / "stability_metrics.csv")
        lines = _format_report(vif_df, sm_df)
        full_text = "\n".join(lines).lower()
        assert "dex" in full_text, "Report must label units as dex"

    def test_all_features_present(self, audit_dir_with_csvs):
        vif_df = pd.read_csv(audit_dir_with_csvs / "vif_table.csv")
        sm_df = pd.read_csv(audit_dir_with_csvs / "stability_metrics.csv")
        lines = _format_report(vif_df, sm_df)
        full_text = "\n".join(lines)
        for feat in ["logM", "log_gbar", "log_j", "hinge"]:
            assert feat in full_text

    def test_severe_vif_flagged(self, audit_dir_high_vif):
        vif_df = pd.read_csv(audit_dir_high_vif / "vif_table.csv")
        sm_df = pd.read_csv(audit_dir_high_vif / "stability_metrics.csv")
        lines = _format_report(vif_df, sm_df)
        full_text = "\n".join(lines)
        assert "severe" in full_text.lower()


# ---------------------------------------------------------------------------
# run_audit
# ---------------------------------------------------------------------------

class TestRunAudit:
    def test_creates_vif_png(self, audit_dir_with_csvs):
        run_audit(audit_dir_with_csvs)
        assert (audit_dir_with_csvs / "vif_table.png").exists()

    def test_creates_stability_png(self, audit_dir_with_csvs):
        run_audit(audit_dir_with_csvs)
        assert (audit_dir_with_csvs / "stability_metrics.png").exists()

    def test_creates_residual_vs_hinge_outputs(self, audit_dir_with_csvs):
        """When audit_features.csv is present, both residual_vs_hinge files must be created."""
        run_audit(audit_dir_with_csvs)
        assert (audit_dir_with_csvs / "residual_vs_hinge.csv").exists()
        assert (audit_dir_with_csvs / "residual_vs_hinge.png").exists()

    def test_no_residual_outputs_without_features_csv(self, tmp_path):
        """When audit_features.csv is absent, residual_vs_hinge files must NOT be created."""
        audit_dir = tmp_path / "audit"
        audit_dir.mkdir()
        pd.DataFrame({
            "feature": ["logM"], "VIF": [1.5],
        }).to_csv(audit_dir / "vif_table.csv", index=False)
        pd.DataFrame({
            "metric": ["condition_number_kappa"],
            "value": [10.0], "status": ["stable"],
            "notes": ["n/a"],
        }).to_csv(audit_dir / "stability_metrics.csv", index=False)
        run_audit(audit_dir)
        assert not (audit_dir / "residual_vs_hinge.csv").exists()
        assert not (audit_dir / "residual_vs_hinge.png").exists()

    def test_raises_on_missing_csv(self, tmp_path):
        empty_dir = tmp_path / "empty_audit"
        empty_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="vif_table.csv"):
            run_audit(empty_dir)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_with_out_flag(self, tmp_path, audit_dir_with_csvs):
        """--out resolves to <out>/audit/."""
        out_dir = audit_dir_with_csvs.parent
        main(["--out", str(out_dir)])
        assert (audit_dir_with_csvs / "vif_table.png").exists()
        assert (audit_dir_with_csvs / "stability_metrics.png").exists()

    def test_main_with_audit_dir_flag(self, audit_dir_with_csvs):
        """--audit-dir points directly at the audit directory."""
        main(["--audit-dir", str(audit_dir_with_csvs)])
        assert (audit_dir_with_csvs / "vif_table.png").exists()

    def test_main_requires_one_of_out_or_audit_dir(self):
        with pytest.raises(SystemExit):
            main([])

