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
    run_audit,
    main,
    _format_report,
    _VIF_WARN,
    _VIF_SEVERE,
    _KAPPA_WARN,
    _KAPPA_SEVERE,
)


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
