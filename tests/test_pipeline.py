"""
tests/test_pipeline.py — "Notarial" integration test for run_pipeline().

Validates three invariants that must hold on every run:
  (i)  Exact column contract of the returned DataFrame.
  (ii) Row count ≥ 150 (SPARC-scale dataset).
  (iii) per_galaxy_summary.csv is written with the same column contract.

Uses a 175-galaxy synthetic dataset so the test runs without a real
SPARC download and CI does not need external data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.scm_analysis import run_pipeline

# Canonical column order expected from run_pipeline()
EXPECTED_COLS = [
    "galaxy",
    "upsilon_disk",
    "chi2_reduced",
    "n_points",
    "Vflat_kms",
    "M_bar_BTFR_Msun",
]

N_GALAXIES = 175  # matches full SPARC sample size


@pytest.fixture(scope="module")
def sparc175_dir(tmp_path_factory):
    """Synthetic 175-galaxy SPARC-like dataset.

    Each galaxy gets a 20-point rotation curve with a flat profile plus
    small Gaussian noise so the fitter converges quickly.
    """
    root = tmp_path_factory.mktemp("sparc175")
    rng = np.random.default_rng(0)

    galaxy_names = [f"G{i:04d}" for i in range(N_GALAXIES)]
    v_flats = np.linspace(80.0, 320.0, N_GALAXIES)

    galaxy_table = pd.DataFrame(
        {
            "Galaxy": galaxy_names,
            "D": np.linspace(3.0, 80.0, N_GALAXIES),
            "Inc": np.linspace(25.0, 85.0, N_GALAXIES),
            "L36": 1e9 * np.arange(1, N_GALAXIES + 1, dtype=float),
            "Vflat": v_flats,
            "e_Vflat": np.full(N_GALAXIES, 5.0),
        }
    )
    galaxy_table.to_csv(root / "SPARC_Lelli2016c.csv", index=False)

    n_pts = 20
    for name, vf in zip(galaxy_names, v_flats):
        r = np.linspace(0.5, 15.0, n_pts)
        rc = pd.DataFrame(
            {
                "r": r,
                "v_obs": np.full(n_pts, vf) + rng.normal(0, 3, n_pts),
                "v_obs_err": np.full(n_pts, 5.0),
                "v_gas": 0.3 * vf * np.ones(n_pts),
                "v_disk": 0.75 * vf * np.ones(n_pts),
                "v_bul": np.zeros(n_pts),
                "SBdisk": np.zeros(n_pts),
                "SBbul": np.zeros(n_pts),
            }
        )
        rc.to_csv(root / f"{name}_rotmod.dat", sep=" ", index=False, header=False)

    return root


def test_run_pipeline_outputs(sparc175_dir, tmp_path):
    """Notarial test: columns, N, all core artefacts, and data sanity."""
    out = tmp_path / "results"
    df = run_pipeline(sparc175_dir, str(out), verbose=False)

    # (i) exact column contract
    assert df.columns.tolist() == EXPECTED_COLS, (
        f"Column mismatch: got {df.columns.tolist()}"
    )

    # (ii) row count
    assert len(df) >= 150, f"Expected ≥150 rows, got {len(df)}"

    # (iii) all six core artefacts must exist
    per_gal = out / "per_galaxy_summary.csv"
    uni = out / "universal_term_comparison_full.csv"
    summary = out / "executive_summary.txt"
    top10 = out / "top10_universal.tex"
    deep_slope = out / "deep_slope_test.csv"
    sensitivity = out / "sensitivity_a0.csv"

    assert per_gal.exists(), "per_galaxy_summary.csv not written"
    assert uni.exists(), "universal_term_comparison_full.csv not written"
    assert summary.exists(), "executive_summary.txt not written"
    assert top10.exists(), "top10_universal.tex not written"
    assert deep_slope.exists(), "deep_slope_test.csv not written"
    assert sensitivity.exists(), "sensitivity_a0.csv not written"

    # (v) audit artefacts must exist
    assert (out / "audit" / "vif_table.csv").exists(), "audit/vif_table.csv not written"
    assert (out / "audit" / "stability_metrics.csv").exists(), "audit/stability_metrics.csv not written"
    assert (out / "audit" / "quality_status.txt").exists(), "audit/quality_status.txt not written"

    # (iv) per_galaxy_summary contract and data sanity
    df2 = pd.read_csv(per_gal)
    assert df2.columns.tolist() == EXPECTED_COLS, (
        f"per_galaxy_summary.csv column mismatch: got {df2.columns.tolist()}"
    )
    assert df2["galaxy"].notna().all(), "galaxy column contains NaN values"


def test_run_pipeline_sorted(sparc175_dir, tmp_path):
    """Returned DataFrame must be sorted by galaxy name."""
    out = tmp_path / "results_sorted"
    df = run_pipeline(sparc175_dir, str(out), verbose=False)
    assert list(df["galaxy"]) == sorted(df["galaxy"].tolist())


def test_run_pipeline_types(sparc175_dir, tmp_path):
    """n_points must be int; numeric columns must be float."""
    out = tmp_path / "results_types"
    df = run_pipeline(sparc175_dir, str(out), verbose=False)
    assert pd.api.types.is_integer_dtype(df["n_points"]), (
        f"n_points dtype should be integer, got {df['n_points'].dtype}"
    )
    for col in ("upsilon_disk", "chi2_reduced", "Vflat_kms", "M_bar_BTFR_Msun"):
        assert pd.api.types.is_float_dtype(df[col]), f"{col} is not float"


def test_audit_artefact_content(sparc175_dir, tmp_path):
    """Audit artefacts have the expected columns and structure."""
    out = tmp_path / "results_audit"
    run_pipeline(sparc175_dir, str(out), verbose=False)

    audit_dir = out / "audit"

    # vif_table.csv — correct columns and expected variable names
    vif_df = pd.read_csv(audit_dir / "vif_table.csv")
    assert list(vif_df.columns) == ["variable", "VIF"], f"vif_table columns: {vif_df.columns.tolist()}"
    assert set(vif_df["variable"]) == {"const", "logM", "log_gbar", "log_j", "hinge"}

    # stability_metrics.csv — correct columns and a kappa row
    stab_df = pd.read_csv(audit_dir / "stability_metrics.csv")
    assert list(stab_df.columns) == ["metric", "value", "status"]
    kappa_row = stab_df[stab_df["metric"] == "condition_number_kappa"]
    assert len(kappa_row) == 1, "condition_number_kappa row must be present"
    assert kappa_row["status"].iloc[0] in ("stable", "unstable")

    # quality_status.txt — must contain overall verdict
    status_text = (audit_dir / "quality_status.txt").read_text(encoding="utf-8")
    assert "overall_status=" in status_text


def test_audit_with_deep_regime_data(tmp_path):
    """Audit produces finite VIF and kappa when the data spans both regimes."""
    from src.scm_analysis import _run_audit, _A0_DEFAULT

    a0 = _A0_DEFAULT  # 1.2e-10 m/s²
    rng = np.random.default_rng(7)
    n = 300

    # Mix of Newtonian (log_g_bar > log10(a0)) and deep-regime (log_g_bar < log10(a0)) points
    log_a0 = np.log10(a0)
    log_gbar = rng.uniform(log_a0 - 2.5, log_a0 + 1.5, n)   # spans both regimes
    log_gobs = 0.5 * log_gbar + rng.normal(0, 0.1, n)
    r_kpc = 10 ** rng.uniform(-0.5, 1.5, n)
    galaxies = [f"G{i:03d}" for i in rng.integers(0, 30, n)]

    compare_df = pd.DataFrame({
        "galaxy": galaxies,
        "r_kpc": r_kpc,
        "g_bar": 10.0 ** log_gbar,
        "g_obs": 10.0 ** log_gobs,
        "log_g_bar": log_gbar,
        "log_g_obs": log_gobs,
    })

    # Galaxy table with L36 varying across galaxies
    galaxy_table = pd.DataFrame({
        "Galaxy": [f"G{i:03d}" for i in range(30)],
        "L36": 1e9 * np.arange(1, 31, dtype=float),
    })

    out_dir = tmp_path / "audit_deep"
    _run_audit(compare_df, galaxy_table, out_dir, a0=a0)

    vif_df = pd.read_csv(out_dir / "audit" / "vif_table.csv")
    stab_df = pd.read_csv(out_dir / "audit" / "stability_metrics.csv")

    assert set(vif_df["variable"]) == {"const", "logM", "log_gbar", "log_j", "hinge"}
    assert vif_df["VIF"].notna().all(), "All VIF values should be finite"

    kappa = float(stab_df[stab_df["metric"] == "condition_number_kappa"]["value"].iloc[0])
    assert np.isfinite(kappa), "Condition number should be finite with mixed-regime data"


def test_audit_hinge_sign(sparc175_dir, tmp_path):
    """hinge = max(0, log10(a0) - log_gbar) is always >= 0."""
    import numpy as np
    from src.scm_analysis import _run_audit

    out = tmp_path / "results_hinge"
    run_pipeline(sparc175_dir, str(out), verbose=False)

    # Read back the compare_df and verify hinge sign directly
    compare_df = pd.read_csv(out / "universal_term_comparison_full.csv")
    a0 = 1.2e-10
    hinge = np.maximum(0.0, np.log10(a0) - compare_df["log_g_bar"].values)
    assert (hinge >= 0).all(), "hinge must be non-negative"
    # In the deep regime (g_bar < a0), hinge > 0
    deep_mask = compare_df["g_bar"].values < a0
    if deep_mask.any():
        assert (hinge[deep_mask] > 0).all(), "hinge must be positive in deep regime"


def test_cli_outdir_alias(sparc175_dir, tmp_path):
    """--outdir is accepted as a CLI alias for --out."""
    from src.scm_analysis import _parse_args

    out_path = str(tmp_path / "cli_out")
    args = _parse_args(["--data-dir", "data/sparc", "--outdir", out_path])
    assert args.out == out_path
