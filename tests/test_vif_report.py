"""
tests/test_vif_report.py — Tests for scripts/vif_report.py.

Uses a small synthetic VIF table to verify labelling, formatting, and the
CLI entry-point (main).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.vif_report import (
    vif_verdict,
    format_vif_report,
    main,
    VIF_CSV_DEFAULT,
    _SEP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vif_csv(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    """Write a minimal vif_table.csv with configurable rows."""
    if rows is None:
        rows = [
            {"variable": "const",    "VIF": 1.05},
            {"variable": "logM",     "VIF": 2.80},
            {"variable": "log_gbar", "VIF": 4.20},
            {"variable": "log_j",    "VIF": 1.90},
            {"variable": "hinge",    "VIF": 3.10},
        ]
    p = tmp_path / "vif_table.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# vif_verdict
# ---------------------------------------------------------------------------

class TestVifVerdict:
    def test_independent_low(self):
        label, symbol = vif_verdict(1.0)
        assert symbol == "✔"
        assert "independent" in label

    def test_independent_boundary(self):
        label, symbol = vif_verdict(2.0)
        assert symbol == "✔"

    def test_moderate(self):
        label, symbol = vif_verdict(3.5)
        assert symbol == "✔"
        assert "moderate" in label

    def test_strong_boundary(self):
        label, symbol = vif_verdict(5.01)
        assert symbol == "⚠"
        assert "strong" in label

    def test_strong_midpoint(self):
        _, symbol = vif_verdict(7.5)
        assert symbol == "⚠"

    def test_structural_boundary(self):
        _, symbol = vif_verdict(10.01)
        assert symbol == "✖"

    def test_structural_high(self):
        label, symbol = vif_verdict(50.0)
        assert symbol == "✖"
        assert "structural" in label


# ---------------------------------------------------------------------------
# format_vif_report
# ---------------------------------------------------------------------------

class TestFormatVifReport:
    def test_returns_list_of_strings(self, tmp_path):
        csv_path = _make_vif_csv(tmp_path)
        vif_df = pd.read_csv(csv_path)
        lines = format_vif_report(vif_df, str(csv_path))
        assert isinstance(lines, list)
        assert all(isinstance(ln, str) for ln in lines)

    def test_contains_separator(self, tmp_path):
        csv_path = _make_vif_csv(tmp_path)
        vif_df = pd.read_csv(csv_path)
        lines = format_vif_report(vif_df, str(csv_path))
        assert any(_SEP in ln for ln in lines)

    def test_contains_all_variables(self, tmp_path):
        csv_path = _make_vif_csv(tmp_path)
        vif_df = pd.read_csv(csv_path)
        lines = format_vif_report(vif_df, str(csv_path))
        joined = "\n".join(lines)
        for var in ["const", "logM", "log_gbar", "log_j", "hinge"]:
            assert var in joined, f"Variable '{var}' missing from report"

    def test_hinge_verdict_independent(self, tmp_path):
        rows = [
            {"variable": "const",    "VIF": 1.0},
            {"variable": "logM",     "VIF": 1.5},
            {"variable": "log_gbar", "VIF": 1.5},
            {"variable": "log_j",    "VIF": 1.5},
            {"variable": "hinge",    "VIF": 1.8},
        ]
        csv_path = _make_vif_csv(tmp_path, rows)
        vif_df = pd.read_csv(csv_path)
        lines = format_vif_report(vif_df, str(csv_path))
        joined = "\n".join(lines)
        assert "independent" in joined
        assert "✔" in joined

    def test_hinge_verdict_watch(self, tmp_path):
        rows = [
            {"variable": "const",    "VIF": 1.0},
            {"variable": "logM",     "VIF": 1.5},
            {"variable": "log_gbar", "VIF": 1.5},
            {"variable": "log_j",    "VIF": 1.5},
            {"variable": "hinge",    "VIF": 7.5},
        ]
        csv_path = _make_vif_csv(tmp_path, rows)
        vif_df = pd.read_csv(csv_path)
        lines = format_vif_report(vif_df, str(csv_path))
        joined = "\n".join(lines)
        assert "⚠" in joined

    def test_hinge_verdict_redundant(self, tmp_path):
        rows = [
            {"variable": "const",    "VIF": 1.0},
            {"variable": "logM",     "VIF": 1.5},
            {"variable": "log_gbar", "VIF": 1.5},
            {"variable": "log_j",    "VIF": 1.5},
            {"variable": "hinge",    "VIF": 25.0},
        ]
        csv_path = _make_vif_csv(tmp_path, rows)
        vif_df = pd.read_csv(csv_path)
        lines = format_vif_report(vif_df, str(csv_path))
        joined = "\n".join(lines)
        assert "✖" in joined
        assert "redundant" in joined

    def test_missing_hinge_column(self, tmp_path):
        rows = [
            {"variable": "logM",     "VIF": 2.0},
            {"variable": "log_gbar", "VIF": 2.5},
        ]
        csv_path = _make_vif_csv(tmp_path, rows)
        vif_df = pd.read_csv(csv_path)
        lines = format_vif_report(vif_df, str(csv_path))
        joined = "\n".join(lines)
        assert "hinge column not found" in joined

    def test_csv_path_in_header(self, tmp_path):
        csv_path = _make_vif_csv(tmp_path)
        vif_df = pd.read_csv(csv_path)
        lines = format_vif_report(vif_df, "my/custom/path.csv")
        joined = "\n".join(lines)
        assert "my/custom/path.csv" in joined


# ---------------------------------------------------------------------------
# main (CLI entry-point)
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dataframe(self, tmp_path):
        csv_path = _make_vif_csv(tmp_path)
        result = main(["--csv", str(csv_path)])
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"variable", "VIF"}

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            main(["--csv", str(tmp_path / "nonexistent.csv")])

    def test_raises_on_missing_columns(self, tmp_path):
        bad_csv = tmp_path / "bad.csv"
        pd.DataFrame({"foo": [1, 2]}).to_csv(bad_csv, index=False)
        with pytest.raises(ValueError, match="missing required columns"):
            main(["--csv", str(bad_csv)])

    def test_writes_log_with_out(self, tmp_path):
        csv_path = _make_vif_csv(tmp_path)
        out_dir = tmp_path / "out"
        main(["--csv", str(csv_path), "--out", str(out_dir)])
        log_file = out_dir / "vif_report.log"
        assert log_file.exists()
        text = log_file.read_text(encoding="utf-8")
        assert "hinge" in text
        assert "VIF" in text

    def test_no_out_does_not_create_log(self, tmp_path):
        csv_path = _make_vif_csv(tmp_path)
        main(["--csv", str(csv_path)])
        assert not (tmp_path / "vif_report.log").exists()

    def test_input_alias_accepted(self, tmp_path):
        """--input is a synonym for --csv and must work identically."""
        csv_path = _make_vif_csv(tmp_path)
        result = main(["--input", str(csv_path)])
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"variable", "VIF"}
