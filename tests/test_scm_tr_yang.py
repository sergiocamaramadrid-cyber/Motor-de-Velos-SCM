"""
tests/test_scm_tr_yang.py — Tests for scripts/scm_tr_yang.py.

Creates synthetic Yang-like data to verify:
- YANG_ENV_COL constant
- main() end-to-end output and column structure
- Regime column presence and valid labels
- Correlation direction in high-mass regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.stats import spearmanr

from scripts.scm_tr_yang import (
    YANG_ENV_COL,
    main,
)
from scripts.scm_tr_regime_test import (
    LOGM_THRESHOLD_DEFAULT,
    MASS_COL,
    SLOPE_COL,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_yang_df(n_low: int = 22, n_high: int = 54, seed: int = 13) -> pd.DataFrame:
    """Synthetic yang-like DataFrame with correlated high-mass regime."""
    rng = np.random.default_rng(seed)
    logM_lo = rng.uniform(8.5, 10.0, n_low)
    logM_hi = rng.uniform(10.1, 11.5, n_high)
    delta_lo = rng.uniform(-1, 1, n_low)
    delta_hi = rng.normal(0, 1, n_high)
    sl_lo    = rng.normal(-0.15, 0.2, n_low)
    sl_hi    = -0.4 * delta_hi + rng.normal(-0.15, 0.35, n_high)

    return pd.DataFrame({
        "galaxy":       [f"G{i:03d}" for i in range(n_low + n_high)],
        MASS_COL:       np.concatenate([logM_lo, logM_hi]),
        YANG_ENV_COL:   np.concatenate([delta_lo, delta_hi]),
        SLOPE_COL:      np.concatenate([sl_lo, sl_hi]),
    })


def _make_csv(tmp_path: Path, df: pd.DataFrame, name: str = "yang_dataset.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_yang_env_col_value(self):
        assert YANG_ENV_COL == "delta_mass_std"

    def test_yang_env_col_is_string(self):
        assert isinstance(YANG_ENV_COL, str)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestYangMain:
    def test_creates_output_file(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path)])
        assert (tmp_path / "scm_tr_yang_dataset.csv").exists()

    def test_returns_dict(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        assert isinstance(result, dict)

    def test_returns_required_keys(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        for key in ["low", "high", "bootstrap", "fisher", "out_path"]:
            assert key in result

    def test_low_high_n_sum_to_total(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        assert result["low"]["n"] + result["high"]["n"] == len(df)

    def test_output_contains_regime_column(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path)])
        out = pd.read_csv(tmp_path / "scm_tr_yang_dataset.csv")
        assert "regime" in out.columns

    def test_regime_column_values_valid(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path)])
        out = pd.read_csv(tmp_path / "scm_tr_yang_dataset.csv")
        assert set(out["regime"].unique()).issubset({"low", "high"})

    def test_output_contains_input_columns(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path)])
        out = pd.read_csv(tmp_path / "scm_tr_yang_dataset.csv")
        for col in [MASS_COL, YANG_ENV_COL, SLOPE_COL]:
            assert col in out.columns

    def test_high_mass_regime_negative_correlation(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        assert result["high"]["rho"] < 0

    def test_high_mass_regime_significant(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        assert result["high"]["pval"] < 0.05

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "missing.csv"), "--out", str(tmp_path)])

    def test_out_path_in_return(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        assert "yang_dataset" in result["out_path"]

    def test_bootstrap_ci_ordering(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        bt = result["bootstrap"]
        assert bt["ci_lo"] <= bt["median"] <= bt["ci_hi"]

    def test_fisher_pval_in_unit_interval(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path)])
        assert 0 <= result["fisher"]["p_two_tail"] <= 1

    def test_output_row_count(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path)])
        out = pd.read_csv(tmp_path / "scm_tr_yang_dataset.csv")
        assert len(out) == len(df)

    def test_custom_threshold_splits_correctly(self, tmp_path):
        df = _make_yang_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--threshold", "10.5"])
        assert result["low"]["n"] + result["high"]["n"] == len(df)
