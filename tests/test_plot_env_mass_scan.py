"""tests/test_plot_env_mass_scan.py — Tests for scripts/plot_env_mass_scan.py."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.plot_env_mass_scan import (
    CATALOG_CSV_DEFAULT,
    FIGURE_CAPTION,
    N_MIN_DEFAULT,
    OUTPUT_PNG_DEFAULT,
    THRESHOLDS_DEFAULT,
    compute_scan,
    generate_figure,
    load_catalog,
    main,
)


# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

def _make_catalog(
    n: int = 50,
    logm_range: tuple[float, float] = (9.5, 11.5),
    seed: int = 0,
) -> pd.DataFrame:
    """Return a synthetic galaxy catalog DataFrame."""
    rng = np.random.default_rng(seed)
    logM = rng.uniform(*logm_range, n)
    env_proxy = rng.normal(0, 1, n)
    # weak anti-correlation with slope_tail to get a non-trivial rho
    slope_tail = -0.05 * env_proxy + rng.normal(0, 0.1, n)
    return pd.DataFrame({
        "galaxy": [f"G{i:03d}" for i in range(n)],
        "logM": logM,
        "env_proxy": env_proxy,
        "slope_tail": slope_tail,
    })


def _write_catalog(path: Path, **kwargs) -> Path:
    _make_catalog(**kwargs).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# compute_scan
# ---------------------------------------------------------------------------

class TestComputeScan:
    def test_returns_dataframe(self):
        df = _make_catalog()
        result = compute_scan(df)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_in_output(self):
        df = _make_catalog()
        result = compute_scan(df)
        assert {"threshold", "n", "rho", "p"}.issubset(result.columns)

    def test_row_count_matches_thresholds(self):
        df = _make_catalog()
        thresholds = [9.8, 10.0, 10.2]
        result = compute_scan(df, thresholds=thresholds)
        assert len(result) == len(thresholds)

    def test_threshold_column_values(self):
        df = _make_catalog()
        thresholds = [9.9, 10.1]
        result = compute_scan(df, thresholds=thresholds)
        assert list(result["threshold"]) == thresholds

    def test_n_counts_galaxies_above_threshold(self):
        df = _make_catalog(n=60)
        thresholds = [10.0]
        result = compute_scan(df, thresholds=thresholds)
        expected_n = int((df["logM"] > 10.0).sum())
        assert result.iloc[0]["n"] == expected_n

    def test_small_subsample_yields_nan(self):
        # Only 3 galaxies above very high threshold → NaN
        df = _make_catalog(n=20, logm_range=(9.5, 11.0))
        result = compute_scan(df, thresholds=[11.5], n_min=10)
        assert result.iloc[0]["n"] < 10
        assert math.isnan(result.iloc[0]["rho"])
        assert math.isnan(result.iloc[0]["p"])

    def test_large_subsample_yields_finite_rho(self):
        df = _make_catalog(n=60)
        result = compute_scan(df, thresholds=[9.5], n_min=5)
        assert math.isfinite(result.iloc[0]["rho"])
        assert math.isfinite(result.iloc[0]["p"])

    def test_rho_in_range(self):
        df = _make_catalog(n=60)
        result = compute_scan(df, thresholds=[9.5], n_min=5)
        rho = result.iloc[0]["rho"]
        assert -1.0 <= rho <= 1.0

    def test_p_in_range(self):
        df = _make_catalog(n=60)
        result = compute_scan(df, thresholds=[9.5], n_min=5)
        p = result.iloc[0]["p"]
        assert 0.0 <= p <= 1.0

    def test_missing_required_column_raises(self):
        df = _make_catalog().drop(columns=["env_proxy"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_scan(df)

    def test_missing_slope_tail_raises(self):
        df = _make_catalog().drop(columns=["slope_tail"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_scan(df)

    def test_missing_logm_raises(self):
        df = _make_catalog().drop(columns=["logM"])
        with pytest.raises(ValueError, match="missing required columns"):
            compute_scan(df)

    def test_uses_default_thresholds_when_none(self):
        df = _make_catalog(n=60)
        result = compute_scan(df)
        assert len(result) == len(THRESHOLDS_DEFAULT)

    def test_custom_n_min(self):
        df = _make_catalog(n=40, logm_range=(10.5, 11.5))
        # With n_min=1 even tiny subsample gets correlated
        result = compute_scan(df, thresholds=[11.4], n_min=1)
        # n should be small but still computed
        assert result.iloc[0]["n"] >= 0

    def test_dropna_ignores_nan_rows(self):
        df = _make_catalog(n=30)
        df.loc[df.index[:5], "env_proxy"] = np.nan
        result = compute_scan(df, thresholds=[9.5], n_min=5)
        assert result.iloc[0]["n"] == len(df) - 5  # still above threshold

    def test_n_column_is_integer_compatible(self):
        df = _make_catalog(n=40)
        result = compute_scan(df, thresholds=[9.5])
        assert int(result.iloc[0]["n"]) == result.iloc[0]["n"]

    def test_default_thresholds_are_list(self):
        assert isinstance(THRESHOLDS_DEFAULT, list)
        assert len(THRESHOLDS_DEFAULT) > 0


# ---------------------------------------------------------------------------
# load_catalog
# ---------------------------------------------------------------------------

class TestLoadCatalog:
    def test_returns_dataframe(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        df = load_catalog(p)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        df = load_catalog(p)
        for col in ["logM", "env_proxy", "slope_tail"]:
            assert col in df.columns

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Galaxy catalog not found"):
            load_catalog(tmp_path / "nonexistent.csv")

    def test_missing_column_raises(self, tmp_path):
        p = tmp_path / "cat.csv"
        df = _make_catalog()
        df.drop(columns=["slope_tail"]).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            load_catalog(p)

    def test_row_count_preserved(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv", n=35)
        df = load_catalog(p)
        assert len(df) == 35


# ---------------------------------------------------------------------------
# generate_figure
# ---------------------------------------------------------------------------

class TestGenerateFigure:
    def _scan(self, n=50, seed=0):
        df = _make_catalog(n=n, seed=seed)
        return compute_scan(df, thresholds=[9.8, 10.0, 10.2])

    def test_returns_figure(self, tmp_path):
        import matplotlib.pyplot as plt
        scan_df = self._scan()
        fig = generate_figure(scan_df, tmp_path / "fig.png")
        assert isinstance(fig, plt.Figure)

    def test_png_written(self, tmp_path):
        scan_df = self._scan()
        generate_figure(scan_df, tmp_path / "fig.png")
        assert (tmp_path / "fig.png").exists()

    def test_pdf_sibling_written(self, tmp_path):
        scan_df = self._scan()
        generate_figure(scan_df, tmp_path / "fig.png")
        assert (tmp_path / "fig.pdf").exists()

    def test_creates_output_directory(self, tmp_path):
        scan_df = self._scan()
        out = tmp_path / "sub" / "fig.png"
        generate_figure(scan_df, out)
        assert out.exists()

    def test_handles_all_nan_rho(self, tmp_path):
        """All NaN rho values (all thresholds above max logM) should not crash."""
        df = _make_catalog(n=20, logm_range=(9.0, 10.0))
        scan_df = compute_scan(df, thresholds=[11.0, 11.5], n_min=50)
        out = tmp_path / "fig.png"
        generate_figure(scan_df, out)
        assert out.exists()

    def test_handles_mixed_finite_nan_rho(self, tmp_path):
        df = _make_catalog(n=40)
        scan_df = compute_scan(df, thresholds=[9.5, 11.5], n_min=10)
        generate_figure(scan_df, tmp_path / "fig.png")
        assert (tmp_path / "fig.png").exists()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        result = main(["--catalog", str(p), "--out", str(out)])
        assert isinstance(result, dict)

    def test_required_keys(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        result = main(["--catalog", str(p), "--out", str(out)])
        assert {"scan_df", "thresholds", "n_min",
                "figure_path", "pdf_path"}.issubset(result)

    def test_scan_df_is_dataframe(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        result = main(["--catalog", str(p), "--out", str(out)])
        assert isinstance(result["scan_df"], pd.DataFrame)

    def test_scan_df_has_correct_columns(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        result = main(["--catalog", str(p), "--out", str(out)])
        assert {"threshold", "n", "rho", "p"}.issubset(result["scan_df"].columns)

    def test_figure_path_is_str(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        result = main(["--catalog", str(p), "--out", str(out)])
        assert isinstance(result["figure_path"], str)

    def test_pdf_path_ends_with_pdf(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        result = main(["--catalog", str(p), "--out", str(out)])
        assert result["pdf_path"].endswith(".pdf")

    def test_png_written(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        main(["--catalog", str(p), "--out", str(out)])
        assert out.exists()

    def test_pdf_written(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        main(["--catalog", str(p), "--out", str(out)])
        assert out.with_suffix(".pdf").exists()

    def test_custom_thresholds(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        result = main([
            "--catalog", str(p),
            "--out", str(out),
            "--thresholds", "9.9", "10.1",
        ])
        assert result["thresholds"] == [9.9, 10.1]
        assert len(result["scan_df"]) == 2

    def test_custom_n_min(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        result = main([
            "--catalog", str(p),
            "--out", str(out),
            "--n-min", "5",
        ])
        assert result["n_min"] == 5

    def test_scan_df_row_count_matches_thresholds(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "fig.png"
        result = main([
            "--catalog", str(p),
            "--out", str(out),
            "--thresholds", "9.8", "10.0", "10.2", "10.5",
        ])
        assert len(result["scan_df"]) == 4

    def test_missing_catalog_raises(self, tmp_path):
        out = tmp_path / "fig.png"
        with pytest.raises(FileNotFoundError):
            main([
                "--catalog", str(tmp_path / "missing.csv"),
                "--out", str(out),
            ])

    def test_creates_nested_output_dir(self, tmp_path):
        p = _write_catalog(tmp_path / "cat.csv")
        out = tmp_path / "nested" / "deep" / "fig.png"
        main(["--catalog", str(p), "--out", str(out)])
        assert out.exists()


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_thresholds_default_is_list(self):
        assert isinstance(THRESHOLDS_DEFAULT, list)

    def test_thresholds_default_non_empty(self):
        assert len(THRESHOLDS_DEFAULT) > 0

    def test_n_min_default_positive(self):
        assert N_MIN_DEFAULT > 0

    def test_catalog_csv_default_is_str(self):
        assert isinstance(CATALOG_CSV_DEFAULT, str)

    def test_output_png_default_is_str(self):
        assert isinstance(OUTPUT_PNG_DEFAULT, str)

    def test_figure_caption_is_str(self):
        assert isinstance(FIGURE_CAPTION, str)

    def test_figure_caption_mentions_spearman(self):
        assert "spearman" in FIGURE_CAPTION.lower()

    def test_figure_caption_mentions_env(self):
        assert "env" in FIGURE_CAPTION.lower()
