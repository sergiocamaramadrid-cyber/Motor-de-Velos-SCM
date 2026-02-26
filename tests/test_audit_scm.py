"""
tests/test_audit_scm.py — Tests for scripts/audit_scm.py

Validates that the OOS audit script correctly:
  - Computes per-radial-point residuals and hinge values.
  - Writes residual_vs_hinge.csv with the expected columns.
  - Writes residual_vs_hinge.png.
  - Writes the full set of audit artefacts (VIF, stability, quality, features).
  - CLI entry-point works with --data-dir and --csv modes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.audit_scm as audit_scm
from src.scm_analysis import run_pipeline


# ---------------------------------------------------------------------------
# Shared fixture — tiny synthetic SPARC dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sparc_tiny_dir(tmp_path_factory):
    """Synthetic 5-galaxy SPARC-like dataset."""
    root = tmp_path_factory.mktemp("sparc_tiny")
    rng = np.random.default_rng(7)
    galaxy_names = [f"T{i:03d}" for i in range(5)]
    v_flats = [120.0, 160.0, 200.0, 240.0, 280.0]

    pd.DataFrame({
        "Galaxy": galaxy_names,
        "D": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Inc": [45.0, 55.0, 65.0, 75.0, 85.0],
        "L36": [1e9, 2e9, 3e9, 4e9, 5e9],
        "Vflat": v_flats,
        "e_Vflat": [5.0] * 5,
    }).to_csv(root / "SPARC_Lelli2016c.csv", index=False)

    n_pts = 20
    for name, vf in zip(galaxy_names, v_flats):
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

    return root


# ---------------------------------------------------------------------------
# compute_oos_residuals
# ---------------------------------------------------------------------------

class TestComputeOosResiduals:
    def test_returns_dataframe(self, sparc_tiny_dir):
        df = audit_scm.compute_oos_residuals(sparc_tiny_dir, verbose=False)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, sparc_tiny_dir):
        df = audit_scm.compute_oos_residuals(sparc_tiny_dir, verbose=False)
        for col in ["galaxy", "r_kpc", "v_hinge", "residual_scm",
                    "residual_bary", "improvement"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_row_count(self, sparc_tiny_dir):
        df = audit_scm.compute_oos_residuals(sparc_tiny_dir, verbose=False)
        # 5 galaxies × 20 radial points each
        assert len(df) == 100

    def test_v_hinge_non_negative(self, sparc_tiny_dir):
        df = audit_scm.compute_oos_residuals(sparc_tiny_dir, verbose=False)
        assert (df["v_hinge"] >= 0).all(), "v_hinge must be non-negative"

    def test_v_hinge_increases_with_radius(self, sparc_tiny_dir):
        df = audit_scm.compute_oos_residuals(sparc_tiny_dir, verbose=False)
        g0 = df[df["galaxy"] == df["galaxy"].iloc[0]].sort_values("r_kpc")
        assert g0["v_hinge"].is_monotonic_increasing, (
            "v_hinge should increase with r for a fixed galaxy"
        )


# ---------------------------------------------------------------------------
# write_residual_vs_hinge
# ---------------------------------------------------------------------------

class TestWriteResidualVsHinge:
    def test_creates_csv(self, sparc_tiny_dir, tmp_path):
        oos_df = audit_scm.compute_oos_residuals(sparc_tiny_dir, verbose=False)
        audit_dir = tmp_path / "audit"
        audit_scm.write_residual_vs_hinge(oos_df, audit_dir)
        assert (audit_dir / "residual_vs_hinge.csv").exists()

    def test_creates_png(self, sparc_tiny_dir, tmp_path):
        oos_df = audit_scm.compute_oos_residuals(sparc_tiny_dir, verbose=False)
        audit_dir = tmp_path / "audit"
        audit_scm.write_residual_vs_hinge(oos_df, audit_dir)
        assert (audit_dir / "residual_vs_hinge.png").exists()
        assert (audit_dir / "residual_vs_hinge.png").stat().st_size > 0

    def test_csv_columns(self, sparc_tiny_dir, tmp_path):
        oos_df = audit_scm.compute_oos_residuals(sparc_tiny_dir, verbose=False)
        audit_dir = tmp_path / "audit"
        audit_scm.write_residual_vs_hinge(oos_df, audit_dir)
        df = pd.read_csv(audit_dir / "residual_vs_hinge.csv")
        for col in ["galaxy", "r_kpc", "v_hinge", "residual_scm",
                    "residual_bary", "improvement"]:
            assert col in df.columns, f"CSV missing column: {col}"

    def test_csv_row_count_matches(self, sparc_tiny_dir, tmp_path):
        oos_df = audit_scm.compute_oos_residuals(sparc_tiny_dir, verbose=False)
        audit_dir = tmp_path / "audit"
        audit_scm.write_residual_vs_hinge(oos_df, audit_dir)
        df = pd.read_csv(audit_dir / "residual_vs_hinge.csv")
        assert len(df) == len(oos_df)


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------

class TestAuditScmCli:
    def test_data_dir_mode(self, sparc_tiny_dir, tmp_path):
        out = tmp_path / "final_audit"
        audit_scm.main(["--data-dir", str(sparc_tiny_dir),
                        "--outdir", str(out), "--quiet"])
        audit_dir = out / "audit"
        assert (audit_dir / "residual_vs_hinge.csv").exists()
        assert (audit_dir / "residual_vs_hinge.png").exists()
        assert (audit_dir / "vif_table.csv").exists()
        assert (audit_dir / "stability_metrics.csv").exists()
        assert (audit_dir / "quality_status.txt").exists()
        assert (audit_dir / "audit_features.csv").exists()

    def test_csv_mode(self, sparc_tiny_dir, tmp_path):
        # First generate the compare CSV via run_pipeline
        pipeline_out = tmp_path / "pipeline"
        run_pipeline(sparc_tiny_dir, pipeline_out, verbose=False)
        csv_file = pipeline_out / "universal_term_comparison_full.csv"
        assert csv_file.exists()

        audit_out = tmp_path / "audit_csv_mode"
        audit_scm.main(["--csv", str(csv_file),
                        "--outdir", str(audit_out), "--quiet"])
        # csv-mode now produces all artefacts including residual_vs_hinge
        audit_dir = audit_out / "audit"
        assert (audit_dir / "vif_table.csv").exists()
        assert (audit_dir / "stability_metrics.csv").exists()
        assert (audit_dir / "quality_status.txt").exists()
        assert (audit_dir / "residual_vs_hinge.csv").exists()
        assert (audit_dir / "residual_vs_hinge.png").exists()

    def test_input_flag_alias(self, sparc_tiny_dir, tmp_path):
        """--input is an alias for --csv and produces identical artefacts."""
        pipeline_out = tmp_path / "pipeline_input"
        run_pipeline(sparc_tiny_dir, pipeline_out, verbose=False)
        sparc_global = pipeline_out / "audit" / "sparc_global.csv"
        assert sparc_global.exists(), "sparc_global.csv must be written by run_pipeline"

        audit_out = tmp_path / "audit_input_mode"
        audit_scm.main(["--input", str(sparc_global),
                        "--outdir", str(audit_out), "--quiet"])
        audit_dir = audit_out / "audit"
        assert (audit_dir / "residual_vs_hinge.csv").exists()
        assert (audit_dir / "residual_vs_hinge.png").exists()
        assert (audit_dir / "vif_table.csv").exists()
        assert (audit_dir / "quality_status.txt").exists()

    def test_seed_flag_accepted(self, sparc_tiny_dir, tmp_path):
        """--seed is accepted without error."""
        pipeline_out = tmp_path / "pipeline_seed"
        run_pipeline(sparc_tiny_dir, pipeline_out, verbose=False)
        csv_file = pipeline_out / "universal_term_comparison_full.csv"
        audit_out = tmp_path / "audit_seed"
        # should not raise
        audit_scm.main(["--csv", str(csv_file),
                        "--outdir", str(audit_out),
                        "--seed", "20260211",
                        "--quiet"])


# ---------------------------------------------------------------------------
# scm_analysis --outdir argument
# ---------------------------------------------------------------------------

class TestOutdirArgument:
    def test_outdir_creates_audit_subdir(self, sparc_tiny_dir, tmp_path):
        from src.scm_analysis import main as scm_main
        out = tmp_path / "audit_v0_6"
        scm_main(["--data-dir", str(sparc_tiny_dir),
                  "--outdir", str(out), "--quiet"])
        assert (out / "audit" / "vif_table.csv").exists()
        assert (out / "audit" / "stability_metrics.csv").exists()
        assert (out / "audit" / "quality_status.txt").exists()
        assert (out / "audit" / "sparc_global.csv").exists()

    def test_out_legacy_still_works(self, sparc_tiny_dir, tmp_path):
        from src.scm_analysis import main as scm_main
        out = tmp_path / "legacy_out"
        scm_main(["--data-dir", str(sparc_tiny_dir),
                  "--out", str(out), "--quiet"])
        assert (out / "per_galaxy_summary.csv").exists()
        assert (out / "audit" / "vif_table.csv").exists()
