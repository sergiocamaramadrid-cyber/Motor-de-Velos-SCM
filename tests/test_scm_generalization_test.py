"""
tests/test_scm_generalization_test.py — Tests for scripts/scm_generalization_test.py.

Uses synthetic in-memory data; no real SPARC download or external files required.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.scm_generalization_test import (
    load_dataset,
    run_loo_cv,
    run_permutation_baseline,
    summary_table,
    main,
    DATASET_DEFAULT,
    OUT_DIR_DEFAULT,
    N_PERM_DEFAULT,
    RANDOM_SEED_DEFAULT,
    FEATURE_COLS_DEFAULT,
    TARGET_COL_DEFAULT,
    REQUIRED_COLS,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_dataset(tmp_path: Path, n: int = 30, seed: int = 0) -> Path:
    """Write a minimal canonical-dataset CSV to tmp_path."""
    rng = np.random.default_rng(seed)
    logMbar = np.linspace(8.5, 11.5, n)
    env_proxy_formal = rng.normal(6.0, 0.5, n)
    slope_tail = -0.09 * env_proxy_formal + 0.05 * logMbar + rng.normal(0, 0.1, n)
    df = pd.DataFrame(
        {
            "galaxy": [f"G{i:04d}" for i in range(n)],
            "logMbar": logMbar,
            "slope_tail": slope_tail,
            "env_proxy": rng.normal(0, 1, n),
            "env_proxy_formal": env_proxy_formal,
        }
    )
    p = tmp_path / "canonical.csv"
    df.to_csv(p, index=False)
    return p


def _make_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    """Return an in-memory DataFrame."""
    rng = np.random.default_rng(seed)
    logMbar = np.linspace(8.5, 11.5, n)
    env_proxy_formal = rng.normal(6.0, 0.5, n)
    slope_tail = -0.09 * env_proxy_formal + 0.05 * logMbar + rng.normal(0, 0.1, n)
    return pd.DataFrame(
        {
            "galaxy": [f"G{i:04d}" for i in range(n)],
            "logMbar": logMbar,
            "slope_tail": slope_tail,
            "env_proxy": rng.normal(0, 1, n),
            "env_proxy_formal": env_proxy_formal,
        }
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_dataset_default_is_string(self):
        assert isinstance(DATASET_DEFAULT, str)

    def test_out_dir_default_is_string(self):
        assert isinstance(OUT_DIR_DEFAULT, str)

    def test_n_perm_default_positive(self):
        assert N_PERM_DEFAULT > 0

    def test_random_seed_default_int(self):
        assert isinstance(RANDOM_SEED_DEFAULT, int)

    def test_feature_cols_default_is_list(self):
        assert isinstance(FEATURE_COLS_DEFAULT, list)
        assert len(FEATURE_COLS_DEFAULT) >= 2

    def test_target_col_default_is_string(self):
        assert TARGET_COL_DEFAULT == "slope_tail"

    def test_required_cols_is_set(self):
        assert isinstance(REQUIRED_COLS, (set, frozenset))
        assert "slope_tail" in REQUIRED_COLS


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

class TestLoadDataset:
    def test_returns_dataframe(self, tmp_path):
        p = _make_dataset(tmp_path)
        df = load_dataset(p)
        assert isinstance(df, pd.DataFrame)

    def test_n_rows(self, tmp_path):
        p = _make_dataset(tmp_path, n=20)
        df = load_dataset(p)
        assert len(df) == 20

    def test_required_cols_present(self, tmp_path):
        p = _make_dataset(tmp_path)
        df = load_dataset(p)
        for col in REQUIRED_COLS:
            assert col in df.columns

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nonexistent.csv")

    def test_missing_column_raises(self, tmp_path):
        df = _make_df()
        df = df.drop(columns=["slope_tail"])
        p = tmp_path / "bad.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(p)

    def test_drops_nan_rows(self, tmp_path):
        df = _make_df(n=10)
        df.loc[2, "slope_tail"] = float("nan")
        p = tmp_path / "nan.csv"
        df.to_csv(p, index=False)
        result = load_dataset(p)
        assert len(result) == 9

    def test_index_reset_after_drop(self, tmp_path):
        df = _make_df(n=10)
        df.loc[0, "slope_tail"] = float("nan")
        p = tmp_path / "nan2.csv"
        df.to_csv(p, index=False)
        result = load_dataset(p)
        assert result.index.tolist() == list(range(len(result)))

    def test_galaxy_column_preserved(self, tmp_path):
        p = _make_dataset(tmp_path, n=5)
        df = load_dataset(p)
        assert "galaxy" in df.columns

    def test_accepts_string_path(self, tmp_path):
        p = _make_dataset(tmp_path)
        df = load_dataset(str(p))
        assert len(df) > 0

    def test_accepts_path_object(self, tmp_path):
        p = _make_dataset(tmp_path)
        df = load_dataset(Path(p))
        assert len(df) > 0


# ---------------------------------------------------------------------------
# run_loo_cv
# ---------------------------------------------------------------------------

class TestRunLooCv:
    def test_returns_dict(self):
        df = _make_df()
        result = run_loo_cv(df)
        assert isinstance(result, dict)

    def test_keys_present(self):
        df = _make_df()
        result = run_loo_cv(df)
        for key in ("n", "rmse_is", "rmse_loo", "rho_loo", "p_loo", "predictions"):
            assert key in result, f"Missing key: {key}"

    def test_n_matches_input(self):
        df = _make_df(n=25)
        result = run_loo_cv(df)
        assert result["n"] == 25

    def test_rmse_is_positive(self):
        df = _make_df()
        result = run_loo_cv(df)
        assert result["rmse_is"] > 0

    def test_rmse_loo_positive(self):
        df = _make_df()
        result = run_loo_cv(df)
        assert result["rmse_loo"] > 0

    def test_rmse_loo_geq_rmse_is(self):
        # LOO RMSE should be >= in-sample RMSE (no overfitting guarantee holds)
        df = _make_df()
        result = run_loo_cv(df)
        assert result["rmse_loo"] >= result["rmse_is"] - 1e-9

    def test_rho_in_range(self):
        df = _make_df()
        result = run_loo_cv(df)
        assert -1.0 <= result["rho_loo"] <= 1.0

    def test_p_in_range(self):
        df = _make_df()
        result = run_loo_cv(df)
        assert 0.0 <= result["p_loo"] <= 1.0

    def test_predictions_is_dataframe(self):
        df = _make_df()
        result = run_loo_cv(df)
        assert isinstance(result["predictions"], pd.DataFrame)

    def test_predictions_n_rows(self):
        df = _make_df(n=20)
        result = run_loo_cv(df)
        assert len(result["predictions"]) == 20

    def test_predictions_columns(self):
        df = _make_df()
        result = run_loo_cv(df)
        for col in ("galaxy", "y_true", "y_pred_is", "y_pred_loo", "residual_loo"):
            assert col in result["predictions"].columns

    def test_residual_loo_definition(self):
        df = _make_df()
        result = run_loo_cv(df)
        pred = result["predictions"]
        np.testing.assert_allclose(
            pred["residual_loo"].values,
            (pred["y_true"] - pred["y_pred_loo"]).values,
            atol=1e-5,
        )

    def test_default_feature_cols(self):
        df = _make_df()
        result = run_loo_cv(df)
        assert result["n"] == len(df)

    def test_custom_feature_cols(self):
        df = _make_df()
        result = run_loo_cv(df, feature_cols=["env_proxy_formal"])
        assert "rmse_loo" in result

    def test_missing_feature_col_raises(self):
        df = _make_df()
        with pytest.raises(ValueError, match="missing columns"):
            run_loo_cv(df, feature_cols=["nonexistent_col"])

    def test_signal_detected(self):
        # With a clear linear signal the correlation should be positive
        df = _make_df(n=50, seed=7)
        result = run_loo_cv(df)
        assert result["rho_loo"] > 0.3

    def test_rmse_is_float(self):
        df = _make_df()
        result = run_loo_cv(df)
        assert isinstance(result["rmse_is"], float)

    def test_small_n(self):
        df = _make_df(n=5)
        result = run_loo_cv(df)
        assert result["n"] == 5


# ---------------------------------------------------------------------------
# run_permutation_baseline
# ---------------------------------------------------------------------------

class TestRunPermutationBaseline:
    def test_returns_dict(self):
        df = _make_df()
        result = run_permutation_baseline(df, n_perm=20, seed=0)
        assert isinstance(result, dict)

    def test_keys_present(self):
        df = _make_df()
        result = run_permutation_baseline(df, n_perm=10, seed=0)
        for key in ("n_perm", "rmse_null_mean", "rmse_null_std",
                    "rmse_null_p95", "rmse_null_p05", "null_rmse_values"):
            assert key in result, f"Missing key: {key}"

    def test_n_perm_matches(self):
        df = _make_df()
        result = run_permutation_baseline(df, n_perm=15, seed=0)
        assert result["n_perm"] == 15
        assert len(result["null_rmse_values"]) == 15

    def test_null_rmse_positive(self):
        df = _make_df()
        result = run_permutation_baseline(df, n_perm=10, seed=0)
        assert result["rmse_null_mean"] > 0

    def test_p95_geq_mean(self):
        df = _make_df()
        result = run_permutation_baseline(df, n_perm=30, seed=0)
        assert result["rmse_null_p95"] >= result["rmse_null_mean"]

    def test_p05_leq_mean(self):
        df = _make_df()
        result = run_permutation_baseline(df, n_perm=30, seed=0)
        assert result["rmse_null_p05"] <= result["rmse_null_mean"]

    def test_reproducible_with_seed(self):
        df = _make_df()
        r1 = run_permutation_baseline(df, n_perm=10, seed=99)
        r2 = run_permutation_baseline(df, n_perm=10, seed=99)
        assert r1["rmse_null_mean"] == r2["rmse_null_mean"]

    def test_different_seeds_differ(self):
        df = _make_df(n=20)
        r1 = run_permutation_baseline(df, n_perm=20, seed=1)
        r2 = run_permutation_baseline(df, n_perm=20, seed=2)
        assert r1["rmse_null_mean"] != r2["rmse_null_mean"]

    def test_model_beats_null_on_signal_data(self):
        # With a clear signal the model's LOO RMSE should beat the null p95
        df = _make_df(n=50, seed=7)
        loo = run_loo_cv(df)
        perm = run_permutation_baseline(df, n_perm=100, seed=0)
        assert loo["rmse_loo"] < perm["rmse_null_p95"]

    def test_null_rmse_values_array(self):
        df = _make_df()
        result = run_permutation_baseline(df, n_perm=5, seed=0)
        assert hasattr(result["null_rmse_values"], "__len__")
        assert len(result["null_rmse_values"]) == 5


# ---------------------------------------------------------------------------
# summary_table
# ---------------------------------------------------------------------------

class TestSummaryTable:
    def _get_loo_perm(self, n=30):
        df = _make_df(n=n, seed=3)
        loo = run_loo_cv(df)
        perm = run_permutation_baseline(df, n_perm=20, seed=0)
        return loo, perm

    def test_returns_dataframe(self):
        loo, perm = self._get_loo_perm()
        tbl = summary_table(loo, perm)
        assert isinstance(tbl, pd.DataFrame)

    def test_has_metric_column(self):
        loo, perm = self._get_loo_perm()
        tbl = summary_table(loo, perm)
        assert "metric" in tbl.columns

    def test_has_value_column(self):
        loo, perm = self._get_loo_perm()
        tbl = summary_table(loo, perm)
        assert "value" in tbl.columns

    def test_has_interpretation_column(self):
        loo, perm = self._get_loo_perm()
        tbl = summary_table(loo, perm)
        assert "interpretation" in tbl.columns

    def test_contains_rmse_loo(self):
        loo, perm = self._get_loo_perm()
        tbl = summary_table(loo, perm)
        assert "rmse_loo" in tbl["metric"].values

    def test_contains_rho_loo(self):
        loo, perm = self._get_loo_perm()
        tbl = summary_table(loo, perm)
        assert "rho_loo" in tbl["metric"].values

    def test_contains_beats_null(self):
        loo, perm = self._get_loo_perm()
        tbl = summary_table(loo, perm)
        assert "beats_null" in tbl["metric"].values

    def test_beats_null_is_0_or_1(self):
        loo, perm = self._get_loo_perm()
        tbl = summary_table(loo, perm)
        val = tbl.loc[tbl["metric"] == "beats_null", "value"].values[0]
        assert val in (0, 1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, tmp_path):
        p = _make_dataset(tmp_path, n=15)
        result = main(["--dataset", str(p), "--out", str(tmp_path / "out"),
                       "--n-perm", "5"])
        assert isinstance(result, dict)

    def test_keys_present(self, tmp_path):
        p = _make_dataset(tmp_path, n=15)
        result = main(["--dataset", str(p), "--out", str(tmp_path / "out"),
                       "--n-perm", "5"])
        for key in ("dataset_path", "n", "loo", "permutation", "summary", "out_dir"):
            assert key in result

    def test_n_correct(self, tmp_path):
        p = _make_dataset(tmp_path, n=15)
        result = main(["--dataset", str(p), "--out", str(tmp_path / "out"),
                       "--n-perm", "5"])
        assert result["n"] == 15

    def test_writes_loo_predictions(self, tmp_path):
        p = _make_dataset(tmp_path, n=10)
        out = tmp_path / "out"
        main(["--dataset", str(p), "--out", str(out), "--n-perm", "5"])
        assert (out / "loo_predictions.csv").exists()

    def test_writes_permutation_baseline(self, tmp_path):
        p = _make_dataset(tmp_path, n=10)
        out = tmp_path / "out"
        main(["--dataset", str(p), "--out", str(out), "--n-perm", "5"])
        assert (out / "permutation_baseline.csv").exists()

    def test_writes_generalization_summary_csv(self, tmp_path):
        p = _make_dataset(tmp_path, n=10)
        out = tmp_path / "out"
        main(["--dataset", str(p), "--out", str(out), "--n-perm", "5"])
        assert (out / "generalization_summary.csv").exists()

    def test_writes_generalization_summary_json(self, tmp_path):
        p = _make_dataset(tmp_path, n=10)
        out = tmp_path / "out"
        main(["--dataset", str(p), "--out", str(out), "--n-perm", "5"])
        assert (out / "generalization_summary.json").exists()

    def test_writes_generalization_summary_txt(self, tmp_path):
        p = _make_dataset(tmp_path, n=10)
        out = tmp_path / "out"
        main(["--dataset", str(p), "--out", str(out), "--n-perm", "5"])
        assert (out / "generalization_summary.txt").exists()

    def test_json_valid(self, tmp_path):
        import json
        p = _make_dataset(tmp_path, n=10)
        out = tmp_path / "out"
        main(["--dataset", str(p), "--out", str(out), "--n-perm", "5"])
        with open(out / "generalization_summary.json") as fh:
            data = json.load(fh)
        assert "rmse_loo" in data

    def test_loo_predictions_n_rows(self, tmp_path):
        p = _make_dataset(tmp_path, n=12)
        out = tmp_path / "out"
        main(["--dataset", str(p), "--out", str(out), "--n-perm", "5"])
        pred = pd.read_csv(out / "loo_predictions.csv")
        assert len(pred) == 12

    def test_permutation_csv_n_rows(self, tmp_path):
        p = _make_dataset(tmp_path, n=10)
        out = tmp_path / "out"
        main(["--dataset", str(p), "--out", str(out), "--n-perm", "7"])
        perm = pd.read_csv(out / "permutation_baseline.csv")
        assert len(perm) == 7

    def test_missing_dataset_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--dataset", str(tmp_path / "none.csv"),
                  "--out", str(tmp_path / "out"), "--n-perm", "5"])

    def test_summary_df_in_result(self, tmp_path):
        p = _make_dataset(tmp_path, n=10)
        result = main(["--dataset", str(p), "--out", str(tmp_path / "out"),
                       "--n-perm", "5"])
        assert isinstance(result["summary"], pd.DataFrame)

    def test_canonical_dataset_works(self, tmp_path):
        """Canonical dataset shipped with the repo loads and runs."""
        canon = Path("data/scm_canonical_dataset.csv")
        if not canon.exists():
            pytest.skip("Canonical dataset not found")
        out = tmp_path / "canon_out"
        result = main(["--dataset", str(canon), "--out", str(out), "--n-perm", "20"])
        assert result["n"] == 79
