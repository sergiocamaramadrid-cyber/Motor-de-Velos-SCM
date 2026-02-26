"""
tests/test_audit_scm.py — Smoke test for scripts/audit_scm.py.

Validates:
  - main() runs end-to-end with a tiny synthetic dataset.
  - All six artifact files are written.
  - Physical hinge constraint: d >= 0 in master_coeffs.json.
  - Permutation p_empirical is in [0, 1].
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _make_tiny_df() -> pd.DataFrame:
    """Synthetic dataset with 6 galaxies, 3 rows each."""
    rng = np.random.default_rng(0)
    rows = []
    for g in range(6):
        gid = f"G{g:02d}"
        logM = 9.5 + 0.2 * g
        for _ in range(3):
            log_gbar = -11.0 + 0.1 * rng.normal()
            log_j = 2.0 + 0.1 * rng.normal()
            y = 0.33 * logM + 0.02 * log_j - 0.01 * log_gbar + rng.normal(scale=0.02)
            rows.append(
                {
                    "galaxy_id": gid,
                    "logM": logM,
                    "log_gbar": log_gbar,
                    "log_j": log_j,
                    "log_v_obs": y,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def audit_mod():
    """Load scripts/audit_scm.py as a module."""
    import sys
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "audit_scm.py"
    spec = importlib.util.spec_from_file_location("audit_scm", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules["audit_scm"] = mod  # needed for @dataclass with string annotations
    spec.loader.exec_module(mod)
    return mod


def test_audit_scm_smoke(tmp_path: Path, audit_mod):
    """End-to-end smoke test: artifacts exist, hinge >= 0, p in [0,1]."""
    df = _make_tiny_df()
    in_csv = tmp_path / "tiny.csv"
    df.to_csv(in_csv, index=False)

    outdir = tmp_path / "out"

    rc = audit_mod.main(
        [
            "--input", str(in_csv),
            "--outdir", str(outdir),
            "--seed", "123",
            "--kfold", "3",
            "--permutations", "10",
            "--y-col", "log_v_obs",
            "--logM-col", "logM",
            "--log-gbar-col", "log_gbar",
            "--log-j-col", "log_j",
        ]
    )
    assert rc == 0

    # All six artifacts must exist
    for fname in [
        "groupkfold_metrics.csv",
        "groupkfold_per_galaxy.csv",
        "coeffs_by_fold.csv",
        "permutation_summary.json",
        "master_coeffs.json",
        "audit_summary.json",
    ]:
        assert (outdir / fname).exists(), f"Missing artifact: {fname}"

    # Physical hinge constraint: d >= 0 in master_coeffs.json
    master = json.loads((outdir / "master_coeffs.json").read_text(encoding="utf-8"))
    d = master["scm_full"]["d"]
    assert d >= 0.0, f"Hinge coefficient d={d} is negative (physical constraint violated)"

    # Permutation p_empirical in [0, 1]
    perm = json.loads((outdir / "permutation_summary.json").read_text(encoding="utf-8"))
    p = perm["p_empirical"]
    assert 0.0 <= p <= 1.0, f"p_empirical={p} out of [0,1]"


def test_audit_scm_fold_metrics_columns(tmp_path: Path, audit_mod):
    """groupkfold_metrics.csv must have the expected columns."""
    df = _make_tiny_df()
    in_csv = tmp_path / "tiny2.csv"
    df.to_csv(in_csv, index=False)

    outdir = tmp_path / "out2"
    audit_mod.main(
        [
            "--input", str(in_csv),
            "--outdir", str(outdir),
            "--seed", "7",
            "--kfold", "3",
            "--permutations", "5",
            "--y-col", "log_v_obs",
        ]
    )

    fold_df = pd.read_csv(outdir / "groupkfold_metrics.csv")
    for col in ("fold", "rmse_btfr", "rmse_no_hinge", "rmse_full"):
        assert col in fold_df.columns, f"Missing column '{col}' in groupkfold_metrics.csv"
    assert len(fold_df) == 3  # kfold=3
