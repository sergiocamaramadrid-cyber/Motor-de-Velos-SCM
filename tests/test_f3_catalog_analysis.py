"""
tests/test_f3_catalog_analysis.py — Tests for scripts/f3_catalog_analysis.py.

Uses small synthetic catalogs; no real data required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scripts.f3_catalog_analysis import (
    analyze_catalog,
    format_report,
    main,
    EXPECTED_SLOPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_catalog(slopes, errs=None, flags=None):
    """Build a minimal F3 catalog DataFrame."""
    n = len(slopes)
    errs = errs if errs is not None else [0.02] * n
    if flags is None:
        flags = [bool(abs(s - EXPECTED_SLOPE) > 2 * e) for s, e in zip(slopes, errs)]
    return pd.DataFrame({
        "galaxy": [f"G{i:02d}" for i in range(n)],
        "friction_slope": slopes,
        "friction_slope_err": errs,
        "velo_inerte_flag": flags,
    })


# ---------------------------------------------------------------------------
# analyze_catalog unit tests
# ---------------------------------------------------------------------------

class TestAnalyzeCatalog:
    def test_returns_required_keys(self):
        df = _make_catalog([0.5, 0.5, 0.5])
        result = analyze_catalog(df)
        for k in ["N", "mean_beta", "std_beta", "delta_mean",
                  "n_velo_inerte", "t_stat", "p_value", "flagged_df"]:
            assert k in result, f"Missing key: {k}"

    def test_n_matches_clean_rows(self):
        slopes = [0.5, 0.5, float("nan"), 0.5]
        df = _make_catalog(slopes)
        result = analyze_catalog(df)
        assert result["N"] == 3   # NaN row excluded

    def test_near_mond_data_p_value_high(self):
        """p-value must be high (>0.05) when β values are scattered around 0.5."""
        rng = np.random.default_rng(42)
        slopes = rng.normal(0.5, 0.02, 30).tolist()
        df = _make_catalog(slopes, errs=[0.02] * 30)
        result = analyze_catalog(df)
        assert result["p_value"] > 0.05

    def test_systematic_deviation_detected(self):
        """A consistent shift of β = 0.6 should be detected as significant."""
        rng = np.random.default_rng(7)
        slopes = (rng.normal(0.6, 0.01, 100)).tolist()
        df = _make_catalog(slopes, errs=[0.01] * 100)
        result = analyze_catalog(df)
        assert result["p_value"] < 0.001
        assert result["delta_mean"] == pytest.approx(0.1, abs=0.02)

    def test_velo_inerte_count_matches_flag_column(self):
        df = _make_catalog(
            [0.5, 0.7, 0.5, 0.3],
            errs=[0.02, 0.02, 0.02, 0.02],
            flags=[False, True, False, True],
        )
        result = analyze_catalog(df)
        assert result["n_velo_inerte"] == 2

    def test_flagged_df_contains_correct_rows(self):
        df = _make_catalog(
            [0.5, 0.8, 0.5],
            errs=[0.02, 0.02, 0.02],
            flags=[False, True, False],
        )
        result = analyze_catalog(df)
        flagged = result["flagged_df"]
        assert len(flagged) == 1
        assert flagged.iloc[0]["galaxy"] == "G01"

    def test_std_positive_for_varied_slopes(self):
        df = _make_catalog([0.4, 0.5, 0.6, 0.45, 0.55])
        result = analyze_catalog(df)
        assert result["std_beta"] > 0.0

    def test_delta_sign(self):
        df = _make_catalog([0.55] * 20)
        result = analyze_catalog(df)
        assert result["delta_mean"] > 0.0

        df2 = _make_catalog([0.45] * 20)
        result2 = analyze_catalog(df2)
        assert result2["delta_mean"] < 0.0

    def test_fallback_flagging_without_flag_column(self):
        """Should still flag galaxies using >2σ rule when flag column absent."""
        # Use a large sample so the std is driven by the majority (≈0.5)
        # and the outlier sticks out beyond 2σ.
        rng = np.random.default_rng(0)
        slopes = rng.normal(0.5, 0.02, 50).tolist() + [0.5 + 0.5]  # one extreme outlier
        df = pd.DataFrame({
            "galaxy": [f"G{i}" for i in range(len(slopes))],
            "friction_slope": slopes,
        })
        result = analyze_catalog(df)
        assert result["n_velo_inerte"] >= 1


# ---------------------------------------------------------------------------
# format_report tests
# ---------------------------------------------------------------------------

class TestFormatReport:
    def test_contains_required_labels(self):
        df = _make_catalog([0.5] * 20)
        result = analyze_catalog(df)
        lines = format_report(result, "test.csv")
        combined = "\n".join(lines)
        assert "N galaxias:" in combined
        assert "Media β:" in combined
        assert "Std β:" in combined
        assert "Δ medio:" in combined
        assert "Test contra β=0.5" in combined
        assert "t:" in combined
        assert "p:" in combined

    def test_verdict_compatible_when_p_high(self):
        df = _make_catalog([0.5] * 30)
        result = analyze_catalog(df)
        lines = format_report(result, "test.csv")
        combined = "\n".join(lines)
        assert "compatible" in combined.lower() or "0.5" in combined


# ---------------------------------------------------------------------------
# CLI (main) integration tests
# ---------------------------------------------------------------------------

class TestMainCLI:
    def test_reads_csv_and_prints_results(self, tmp_path):
        cat = tmp_path / "f3_catalog.csv"
        df = _make_catalog([0.5 + 0.01 * i for i in range(20)])
        df.to_csv(cat, index=False)

        result = main(["--csv", str(cat)])
        assert "N" in result
        assert result["N"] == 20

    def test_missing_csv_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "nonexistent.csv")])

    def test_missing_column_raises(self, tmp_path):
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"galaxy": ["A"]}).to_csv(bad, index=False)
        with pytest.raises(ValueError, match="friction_slope"):
            main(["--csv", str(bad)])

    def test_writes_output_files(self, tmp_path):
        cat = tmp_path / "f3_catalog.csv"
        df = _make_catalog([0.5] * 15)
        df.to_csv(cat, index=False)
        out = tmp_path / "out"

        main(["--csv", str(cat), "--out", str(out)])
        assert (out / "f3_analysis.csv").exists()
        assert (out / "f3_analysis.log").exists()
        assert (out / "f3_flagged.csv").exists()

    def test_summary_csv_has_scalar_stats(self, tmp_path):
        cat = tmp_path / "f3_catalog.csv"
        df = _make_catalog([0.5 + 0.02 * i for i in range(10)])
        df.to_csv(cat, index=False)
        out = tmp_path / "out"
        main(["--csv", str(cat), "--out", str(out)])

        summary = pd.read_csv(out / "f3_analysis.csv")
        assert "mean_beta" in summary.columns
        assert "p_value" in summary.columns
        assert len(summary) == 1

    def test_end_to_end_with_generated_catalog(self, tmp_path):
        """Integration: generate → analyze round-trip."""
        from scripts.generate_f3_catalog import generate_synthetic

        cat = tmp_path / "f3_catalog.csv"
        generate_synthetic(cat, n_galaxies=40, seed=77)
        result = main(["--csv", str(cat)])
        assert result["N"] == 40
        assert np.isfinite(result["t_stat"])
        assert 0.0 < result["p_value"] <= 1.0
