"""
tests/test_mw_hemisphere_test.py — Tests for scripts/mw_hemisphere_test.py.

Creates synthetic MW Cepheid-like data to verify:
- compute_hemisphere_delta returns correct dict keys and values
- bootstrap_hemisphere_delta returns correct CI structure
- main() end-to-end output in stat-value format
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.mw_hemisphere_test import (
    R_MIN_DEFAULT,
    N_BOOT_DEFAULT,
    compute_hemisphere_delta,
    bootstrap_hemisphere_delta,
    main,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mw_df(n: int = 600, seed: int = 77, delta_signal: float = 0.0) -> pd.DataFrame:
    """Synthetic MW Cepheid dataframe.

    Parameters
    ----------
    delta_signal : extra km/s added to northern hemisphere Vc to plant a signal
    """
    rng = np.random.default_rng(seed)
    R_kpc   = rng.uniform(2, 25, n)
    lon_deg = rng.uniform(0, 360, n)
    lat_deg = rng.normal(0, 15, n)
    Vc_kms  = 230 - 5 * (R_kpc - 8) / 8 + rng.normal(0, 5, n)
    Vc_kms[lat_deg > 0] += delta_signal
    return pd.DataFrame({
        "R_kpc":   R_kpc,
        "Vc_kms":  Vc_kms,
        "e_Vc":    rng.uniform(5, 15, n),
        "lon_deg": lon_deg,
        "lat_deg": lat_deg,
        "source":  "synthetic_gaia",
    })


def _make_csv(tmp_path: Path, df: pd.DataFrame, name: str = "mw_cepheids.csv") -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# compute_hemisphere_delta
# ---------------------------------------------------------------------------

class TestComputeHemisphereDelta:
    def test_returns_dict(self):
        df = _make_mw_df()
        result = compute_hemisphere_delta(df)
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = _make_mw_df()
        result = compute_hemisphere_delta(df)
        for key in ["n_north", "n_south", "mean_north", "mean_south", "delta"]:
            assert key in result

    def test_delta_equals_mean_north_minus_south(self):
        df = _make_mw_df()
        result = compute_hemisphere_delta(df)
        assert result["delta"] == pytest.approx(
            result["mean_north"] - result["mean_south"], rel=1e-9
        )

    def test_n_north_plus_south_equals_total(self):
        df = _make_mw_df(n=200)
        result = compute_hemisphere_delta(df)
        assert result["n_north"] + result["n_south"] == len(df)

    def test_planted_signal_recovered(self):
        """A +5 km/s shift in north should give delta≈5."""
        df = _make_mw_df(delta_signal=5.0, n=2000, seed=3)
        result = compute_hemisphere_delta(df)
        assert result["delta"] == pytest.approx(5.0, abs=0.5)

    def test_delta_positive_when_north_boosted(self):
        df = _make_mw_df(delta_signal=3.0, n=1000, seed=9)
        result = compute_hemisphere_delta(df)
        assert result["delta"] > 0

    def test_custom_columns(self):
        df = _make_mw_df()
        df = df.rename(columns={"lat_deg": "b_deg", "Vc_kms": "V_kms"})
        result = compute_hemisphere_delta(df, lat_col="b_deg", vc_col="V_kms")
        assert "delta" in result

    def test_n_north_and_south_positive(self):
        df = _make_mw_df(n=300)
        result = compute_hemisphere_delta(df)
        assert result["n_north"] > 0 and result["n_south"] > 0


# ---------------------------------------------------------------------------
# bootstrap_hemisphere_delta
# ---------------------------------------------------------------------------

class TestBootstrapHemisphereDelta:
    def _boot(self, n=400, n_boot=200, seed=42):
        df = _make_mw_df(n=n, seed=11)
        return bootstrap_hemisphere_delta(df, n_boot=n_boot, seed=seed)

    def test_returns_dict(self):
        assert isinstance(self._boot(), dict)

    def test_required_keys(self):
        result = self._boot()
        for key in ["boot_median", "ci_lo", "ci_hi", "n_boot"]:
            assert key in result

    def test_ci_lo_le_median_le_ci_hi(self):
        result = self._boot()
        assert result["ci_lo"] <= result["boot_median"] <= result["ci_hi"]

    def test_n_boot_stored(self):
        result = self._boot(n_boot=150)
        assert result["n_boot"] == 150

    def test_reproducible_with_seed(self):
        df = _make_mw_df(n=200, seed=5)
        r1 = bootstrap_hemisphere_delta(df, n_boot=100, seed=7)
        r2 = bootstrap_hemisphere_delta(df, n_boot=100, seed=7)
        assert r1["boot_median"] == pytest.approx(r2["boot_median"])

    def test_planted_signal_ci_excludes_zero(self):
        """With a large planted signal, 95% CI should not include zero."""
        df = _make_mw_df(delta_signal=10.0, n=1000, seed=2)
        result = bootstrap_hemisphere_delta(df, n_boot=500, seed=42)
        assert result["ci_lo"] > 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMWHemisphereMain:
    def test_creates_output_csv(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "100"])
        assert (tmp_path / "mw_hemisphere_test.csv").exists()

    def test_returns_dict(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "100"])
        assert isinstance(result, dict)

    def test_returns_required_keys(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "100"])
        for key in ["delta_result", "bootstrap", "r_min", "out_path"]:
            assert key in result

    def test_output_csv_has_stat_value_columns(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "100"])
        out = pd.read_csv(tmp_path / "mw_hemisphere_test.csv")
        assert "stat" in out.columns
        assert "value" in out.columns

    def test_output_csv_required_stats_present(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "100"])
        out = pd.read_csv(tmp_path / "mw_hemisphere_test.csv")
        stats = set(out["stat"].values)
        required = {
            "r_min_kpc", "n_north", "n_south",
            "mean_vc_north_kms", "mean_vc_south_kms",
            "delta_kms", "boot_median_kms", "ci_lo_kms", "ci_hi_kms",
        }
        assert required.issubset(stats)

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "gone.csv"), "--out", str(tmp_path)])

    def test_r_min_returned_correctly(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path),
                       "--n-boot", "50", "--r-min", "7.0"])
        assert result["r_min"] == pytest.approx(7.0)

    def test_delta_result_has_required_keys(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        for key in ["n_north", "n_south", "mean_north", "mean_south", "delta"]:
            assert key in result["delta_result"]

    def test_bootstrap_result_has_required_keys(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        result = main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50"])
        for key in ["boot_median", "ci_lo", "ci_hi", "n_boot"]:
            assert key in result["bootstrap"]

    def test_r_min_stored_in_output_csv(self, tmp_path):
        df = _make_mw_df()
        csv = _make_csv(tmp_path, df)
        main(["--csv", str(csv), "--out", str(tmp_path), "--n-boot", "50", "--r-min", "6.0"])
        out = pd.read_csv(tmp_path / "mw_hemisphere_test.csv")
        r_min_row = out[out["stat"] == "r_min_kpc"]
        assert float(r_min_row["value"].values[0]) == pytest.approx(6.0)
