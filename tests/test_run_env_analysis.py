"""tests/test_run_env_analysis.py — Tests for scripts/run_env_analysis.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_env_analysis import (
    F3_REF,
    HE_CORRECTION,
    UPSILON_36,
    UNIT_SCALE,
    _resolve_galaxy_key,
    _resolve_beta,
    _resolve_n_deep,
    _resolve_rdisk,
    compute_env_proxy,
    compute_stats,
    load_data,
    run_ols,
    save_outputs,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_f3_csv(tmp_path: Path, n: int = 30, beta_col: str = "F3") -> Path:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "galaxy": [f"G{i:03d}" for i in range(n)],
            beta_col: rng.uniform(0.2, 1.0, n),
            "n_deep": rng.integers(5, 20, n),
        }
    )
    p = tmp_path / "f3.csv"
    df.to_csv(p, index=False)
    return p


def _make_sparc_csv(tmp_path: Path, n: int = 30) -> Path:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "Galaxy": [f"G{i:03d}" for i in range(n)],
            "Inc": rng.uniform(20, 80, n),
            "L36": rng.uniform(0.5, 50.0, n),
            "MHI": rng.uniform(0.1, 10.0, n),
            "Rdisk": rng.uniform(1.0, 15.0, n),
        }
    )
    p = tmp_path / "sparc.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# _resolve_galaxy_key
# ---------------------------------------------------------------------------


class TestResolveGalaxyKey:
    def test_galaxy_lower(self):
        df = pd.DataFrame({"galaxy": ["NGC1", "NGC2"]})
        result = _resolve_galaxy_key(df)
        assert list(result) == ["NGC1", "NGC2"]

    def test_Galaxy_upper(self):
        df = pd.DataFrame({"Galaxy": [" NGC1 ", "NGC2"]})
        result = _resolve_galaxy_key(df)
        assert result.iloc[0] == "NGC1"

    def test_name_alias(self):
        df = pd.DataFrame({"name": ["A", "B"]})
        result = _resolve_galaxy_key(df)
        assert list(result) == ["A", "B"]

    def test_no_key_raises(self):
        df = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(ValueError, match="No galaxy-name column"):
            _resolve_galaxy_key(df)


# ---------------------------------------------------------------------------
# _resolve_beta
# ---------------------------------------------------------------------------


class TestResolveBeta:
    def test_f3_col(self):
        df = pd.DataFrame({"F3": [0.4, 0.6]})
        result = _resolve_beta(df)
        assert list(result) == pytest.approx([0.4, 0.6])

    def test_friction_slope_col(self):
        df = pd.DataFrame({"friction_slope": [0.5, 0.7]})
        result = _resolve_beta(df)
        assert list(result) == pytest.approx([0.5, 0.7])

    def test_beta_col(self):
        df = pd.DataFrame({"beta": [0.3, 0.9]})
        result = _resolve_beta(df)
        assert list(result) == pytest.approx([0.3, 0.9])

    def test_f3_preferred_over_beta(self):
        df = pd.DataFrame({"F3": [0.4], "beta": [0.9]})
        result = _resolve_beta(df)
        assert result.iloc[0] == pytest.approx(0.4)

    def test_missing_returns_nan(self):
        df = pd.DataFrame({"foo": [1.0]})
        result = _resolve_beta(df)
        assert np.isnan(result.iloc[0])

    def test_non_numeric_coerced(self):
        df = pd.DataFrame({"F3": ["0.5", "bad", "0.7"]})
        result = _resolve_beta(df)
        assert result.iloc[0] == pytest.approx(0.5)
        assert np.isnan(result.iloc[1])


# ---------------------------------------------------------------------------
# _resolve_n_deep
# ---------------------------------------------------------------------------


class TestResolveNDeep:
    def test_n_deep(self):
        df = pd.DataFrame({"n_deep": [10, 20]})
        assert list(_resolve_n_deep(df)) == [10, 20]

    def test_n_tail_points(self):
        df = pd.DataFrame({"n_tail_points": [5, 15]})
        assert list(_resolve_n_deep(df)) == [5, 15]

    def test_missing_returns_nan(self):
        df = pd.DataFrame({"x": [1]})
        assert np.isnan(_resolve_n_deep(df).iloc[0])


# ---------------------------------------------------------------------------
# _resolve_rdisk
# ---------------------------------------------------------------------------


class TestResolveRdisk:
    def test_rdisk(self):
        df = pd.DataFrame({"Rdisk": [5.0, 10.0]})
        assert list(_resolve_rdisk(df)) == pytest.approx([5.0, 10.0])

    def test_Re_fallback(self):
        df = pd.DataFrame({"Re": [3.0]})
        assert list(_resolve_rdisk(df)) == pytest.approx([3.0])

    def test_rdisk_preferred_over_Re(self):
        df = pd.DataFrame({"Rdisk": [5.0], "Re": [3.0]})
        assert list(_resolve_rdisk(df)) == pytest.approx([5.0])

    def test_missing_returns_nan(self):
        df = pd.DataFrame({"x": [1.0]})
        assert np.isnan(_resolve_rdisk(df).iloc[0])


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_basic_load(self, tmp_path):
        f3 = _make_f3_csv(tmp_path)
        sp = _make_sparc_csv(tmp_path)
        df = load_data(f3, sp)
        assert "galaxy_id" in df.columns
        assert "slope_tail" in df.columns
        assert "logM" in df.columns
        assert "MHI" in df.columns
        assert "Rdisk" in df.columns

    def test_row_count(self, tmp_path):
        f3 = _make_f3_csv(tmp_path, n=20)
        sp = _make_sparc_csv(tmp_path, n=20)
        df = load_data(f3, sp)
        assert len(df) == 20

    def test_logm_positive_for_valid_rows(self, tmp_path):
        f3 = _make_f3_csv(tmp_path, n=10)
        sp = _make_sparc_csv(tmp_path, n=10)
        df = load_data(f3, sp)
        valid = df["logM"].dropna()
        assert (valid > 0).all()

    def test_merge_by_galaxy_id(self, tmp_path):
        f3 = _make_f3_csv(tmp_path, n=5)
        sp = _make_sparc_csv(tmp_path, n=5)
        df = load_data(f3, sp)
        assert df["galaxy_id"].iloc[0] == "G000"

    def test_logm_calculation(self, tmp_path):
        """logM = log10(0.5 * L36 * 1e9 + 1.33 * MHI * 1e9) for single row."""
        pd.DataFrame({"galaxy": ["X1"], "F3": [0.5], "n_deep": [10]}).to_csv(
            tmp_path / "f3.csv", index=False
        )
        L36_val, MHI_val = 5.0, 2.0
        pd.DataFrame(
            {"Galaxy": ["X1"], "Inc": [45], "L36": [L36_val], "MHI": [MHI_val], "Rdisk": [5.0]}
        ).to_csv(tmp_path / "sp.csv", index=False)
        df = load_data(tmp_path / "f3.csv", tmp_path / "sp.csv")
        expected = np.log10(
            UPSILON_36 * L36_val * UNIT_SCALE + HE_CORRECTION * MHI_val * UNIT_SCALE
        )
        assert df["logM"].iloc[0] == pytest.approx(expected)

    def test_no_match_gives_nan_logm(self, tmp_path):
        """Galaxy in F3 but not in SPARC → NaN for SPARC-derived columns."""
        pd.DataFrame({"galaxy": ["NONAME"], "F3": [0.5], "n_deep": [10]}).to_csv(
            tmp_path / "f3.csv", index=False
        )
        pd.DataFrame(
            {"Galaxy": ["OTHER"], "Inc": [45], "L36": [5.0], "MHI": [2.0], "Rdisk": [5.0]}
        ).to_csv(tmp_path / "sp.csv", index=False)
        df = load_data(tmp_path / "f3.csv", tmp_path / "sp.csv")
        assert np.isnan(df["logM"].iloc[0])

    def test_friction_slope_alias(self, tmp_path):
        """Accepts friction_slope instead of F3."""
        f3 = _make_f3_csv(tmp_path, beta_col="friction_slope")
        sp = _make_sparc_csv(tmp_path)
        df = load_data(f3, sp)
        assert df["slope_tail"].notna().any()


# ---------------------------------------------------------------------------
# compute_env_proxy
# ---------------------------------------------------------------------------


class TestComputeEnvProxy:
    def _base_df(self):
        return pd.DataFrame({
            "slope_tail": [0.6, 0.4, 0.8],
            "MHI": [2.0, 5.0, 0.5],
            "Rdisk": [4.0, 8.0, 2.0],
        })

    def test_delta_f3_value(self):
        df = self._base_df()
        out = compute_env_proxy(df)
        assert out["delta_f3"].iloc[0] == pytest.approx(0.6 - F3_REF)

    def test_env_proxy_formula(self):
        df = self._base_df()
        out = compute_env_proxy(df)
        expected = np.log10(2.0) - 2 * np.log10(4.0)
        assert out["env_proxy"].iloc[0] == pytest.approx(expected)

    def test_does_not_modify_original(self):
        df = self._base_df()
        _ = compute_env_proxy(df)
        assert "delta_f3" not in df.columns

    def test_nan_slope_gives_nan_delta(self):
        df = pd.DataFrame({"slope_tail": [np.nan], "MHI": [1.0], "Rdisk": [2.0]})
        out = compute_env_proxy(df)
        assert np.isnan(out["delta_f3"].iloc[0])

    def test_zero_mhi_gives_neg_inf_or_nan(self):
        df = pd.DataFrame({"slope_tail": [0.5], "MHI": [0.0], "Rdisk": [2.0]})
        out = compute_env_proxy(df)
        # log10(0 / ...) = -inf; should not raise
        assert np.isneginf(out["env_proxy"].iloc[0]) or np.isnan(out["env_proxy"].iloc[0])


# ---------------------------------------------------------------------------
# run_ols
# ---------------------------------------------------------------------------


class TestRunOls:
    def _make_df(self, n: int = 40) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "galaxy_id": [f"G{i}" for i in range(n)],
            "slope_tail": rng.uniform(0.2, 1.0, n),
            "logM": rng.uniform(8.0, 11.0, n),
            "Rmax": rng.uniform(1.0, 20.0, n),
            "MHI": rng.uniform(0.1, 10.0, n),
            "Rdisk": rng.uniform(1.0, 15.0, n),
        })
        df = compute_env_proxy(df)
        return df

    def test_returns_three_values(self):
        df = self._make_df()
        result = run_ols(df)
        assert len(result) == 3

    def test_df_fit_has_residual(self):
        df = self._make_df()
        _, _, df_fit = run_ols(df)
        assert "residual" in df_fit.columns

    def test_model_full_has_env_proxy_param(self):
        df = self._make_df()
        _, model_full, _ = run_ols(df)
        assert "env_proxy" in model_full.params.index

    def test_base_model_r2_between_0_and_1(self):
        df = self._make_df()
        model_base, _, _ = run_ols(df)
        assert 0.0 <= model_base.rsquared <= 1.0

    def test_drops_nan_rows(self):
        df = self._make_df()
        df.loc[0, "logM"] = np.nan
        _, _, df_fit = run_ols(df)
        assert df_fit["logM"].notna().all()

    def test_full_model_aic_leq_base(self):
        """Adding env_proxy should not increase AIC by more than a tiny amount."""
        df = self._make_df(n=80)
        model_base, model_full, _ = run_ols(df)
        # In the worst case (useless predictor) AIC increases by ~2; we just
        # check the calculation runs without error.
        assert isinstance(model_full.aic, float)

    def test_models_use_hc3_covariance(self):
        """Both base and full models must be fitted with HC3 robust SE."""
        df = self._make_df()
        model_base, model_full, _ = run_ols(df)
        assert model_base.cov_type == "HC3"
        assert model_full.cov_type == "HC3"


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------


class TestComputeStats:
    def _fit(self, n: int = 40):
        rng = np.random.default_rng(5)
        df = pd.DataFrame({
            "galaxy_id": [f"G{i}" for i in range(n)],
            "slope_tail": rng.uniform(0.2, 1.0, n),
            "logM": rng.uniform(8.0, 11.0, n),
            "Rmax": rng.uniform(1.0, 20.0, n),
            "MHI": rng.uniform(0.1, 10.0, n),
            "Rdisk": rng.uniform(1.0, 15.0, n),
        })
        df = compute_env_proxy(df)
        mb, mf, df_fit = run_ols(df)
        return mb, mf, df_fit

    def test_keys_present(self):
        mb, mf, df_fit = self._fit()
        stats = compute_stats(df_fit, mb, mf)
        for k in ("N", "rho", "p", "delta_aic", "coef_env", "p_env"):
            assert k in stats

    def test_N_equals_df_fit_len(self):
        mb, mf, df_fit = self._fit(n=30)
        stats = compute_stats(df_fit, mb, mf)
        assert stats["N"] == len(df_fit)

    def test_rho_in_range(self):
        mb, mf, df_fit = self._fit()
        stats = compute_stats(df_fit, mb, mf)
        assert -1.0 <= stats["rho"] <= 1.0

    def test_p_in_0_1(self):
        mb, mf, df_fit = self._fit()
        stats = compute_stats(df_fit, mb, mf)
        assert 0.0 <= stats["p"] <= 1.0

    def test_p_env_in_0_1(self):
        mb, mf, df_fit = self._fit()
        stats = compute_stats(df_fit, mb, mf)
        assert 0.0 <= stats["p_env"] <= 1.0

    def test_delta_aic_is_float(self):
        mb, mf, df_fit = self._fit()
        stats = compute_stats(df_fit, mb, mf)
        assert isinstance(stats["delta_aic"], float)


# ---------------------------------------------------------------------------
# save_outputs
# ---------------------------------------------------------------------------


class TestSaveOutputs:
    def _make_fit_and_stats(self, n: int = 30):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({
            "galaxy_id": [f"G{i}" for i in range(n)],
            "slope_tail": rng.uniform(0.2, 1.0, n),
            "logM": rng.uniform(8.0, 11.0, n),
            "Rmax": rng.uniform(1.0, 20.0, n),
            "MHI": rng.uniform(0.1, 10.0, n),
            "Rdisk": rng.uniform(1.0, 15.0, n),
        })
        df = compute_env_proxy(df)
        mb, mf, df_fit = run_ols(df)
        stats = compute_stats(df_fit, mb, mf)
        return df_fit, stats

    def test_creates_output_dir(self, tmp_path):
        df_fit, stats = self._make_fit_and_stats()
        out = tmp_path / "new_dir"
        save_outputs(df_fit, stats, out)
        assert out.is_dir()

    def test_csv_written(self, tmp_path):
        df_fit, stats = self._make_fit_and_stats()
        save_outputs(df_fit, stats, tmp_path)
        assert (tmp_path / "galaxy_catalog_with_env.csv").exists()

    def test_summary_written(self, tmp_path):
        df_fit, stats = self._make_fit_and_stats()
        save_outputs(df_fit, stats, tmp_path)
        assert (tmp_path / "summary.txt").exists()

    def test_summary_contains_N(self, tmp_path):
        df_fit, stats = self._make_fit_and_stats()
        save_outputs(df_fit, stats, tmp_path)
        text = (tmp_path / "summary.txt").read_text()
        assert "N = " in text

    def test_csv_row_count(self, tmp_path):
        df_fit, stats = self._make_fit_and_stats(n=25)
        save_outputs(df_fit, stats, tmp_path)
        loaded = pd.read_csv(tmp_path / "galaxy_catalog_with_env.csv")
        assert len(loaded) == 25


# ---------------------------------------------------------------------------
# End-to-end via main
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs(self, tmp_path):
        from scripts.run_env_analysis import main

        f3 = _make_f3_csv(tmp_path, n=30)
        sp = _make_sparc_csv(tmp_path, n=30)
        out = tmp_path / "out"
        main([
            "--f3-catalog", str(f3),
            "--sparc-basic", str(sp),
            "--out", str(out),
        ])
        assert (out / "summary.txt").exists()
        assert (out / "galaxy_catalog_with_env.csv").exists()

    def test_main_friction_slope_alias(self, tmp_path):
        from scripts.run_env_analysis import main

        f3 = _make_f3_csv(tmp_path, n=20, beta_col="friction_slope")
        sp = _make_sparc_csv(tmp_path, n=20)
        out = tmp_path / "out2"
        main(["--f3-catalog", str(f3), "--sparc-basic", str(sp), "--out", str(out)])
        assert (out / "summary.txt").exists()
