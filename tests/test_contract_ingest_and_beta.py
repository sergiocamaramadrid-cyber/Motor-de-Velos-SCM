"""
tests/test_contract_ingest_and_beta.py — End-to-end test for BIG-SPARC
ingest → F3 catalog pipeline.

Covers:
  - Synthetic BIG-SPARC fixture creation (galaxies + rc_points tables)
  - Ingest via module API (``ingest()``)
  - Ingest via CLI module invocation (``python -m scripts.ingest_big_sparc_contract``)
  - Ingest via direct script execution (``python scripts/ingest_big_sparc_contract.py``)
  - F3 catalog generation via API (``generate_catalog()``)
  - F3 catalog generation via CLI module invocation
  - F3 catalog generation via direct script execution
  - Contract validation utilities (``read_table``, ``compute_vbar_kms``,
    ``validate_contract``)
  - ``--min-deep`` flag behaviour
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.contract_utils import (
    CONTRACT_COLUMNS,
    compute_vbar_kms,
    read_table,
    validate_contract,
)
from scripts.ingest_big_sparc_contract import ingest
from scripts.generate_f3_catalog_from_contract import generate_catalog

# ---------------------------------------------------------------------------
# Repository root (for script-direct invocation)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Synthetic fixture helpers
# ---------------------------------------------------------------------------

N_GALAXIES = 10
N_POINTS_PER_GALAXY = 15


def _make_galaxies(n: int = N_GALAXIES) -> pd.DataFrame:
    """Minimal galaxies table."""
    return pd.DataFrame({"galaxy": [f"TG{i:03d}" for i in range(n)]})


def _make_rc_points(
    n_gal: int = N_GALAXIES,
    n_pts: int = N_POINTS_PER_GALAXY,
    include_vbar: bool = False,
) -> pd.DataFrame:
    """Synthetic rotation-curve points table.

    If *include_vbar* is True, ``vbar_kms`` is already present (passthrough
    path).  Otherwise v_gas and v_disk are provided so the ingestor can derive
    it via quadrature.
    """
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_gal):
        galaxy = f"TG{i:03d}"
        r = np.linspace(0.5, 10.0, n_pts)
        vobs = 80.0 + i * 10 + rng.normal(0, 3, n_pts)
        verr = np.full(n_pts, 5.0)
        v_gas = 0.3 * vobs
        v_disk = 0.75 * vobs
        v_bul = np.zeros(n_pts)
        vbar = np.sqrt(v_gas**2 + v_disk**2 + v_bul**2)
        for j in range(n_pts):
            row: dict = {
                "galaxy": galaxy,
                "r_kpc": round(float(r[j]), 3),
                "vobs_kms": round(float(vobs[j]), 3),
                "vobs_err_kms": round(float(verr[j]), 3),
            }
            if include_vbar:
                row["vbar_kms"] = round(float(vbar[j]), 3)
            else:
                row["v_gas"] = round(float(v_gas[j]), 3)
                row["v_disk"] = round(float(v_disk[j]), 3)
                row["v_bul"] = round(float(v_bul[j]), 3)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    """Write synthetic BIG-SPARC CSV fixtures and return the directory path."""
    root = tmp_path_factory.mktemp("big_sparc")
    _make_galaxies().to_csv(root / "galaxies.csv", index=False)
    _make_rc_points().to_csv(root / "rc_points.csv", index=False)
    return root


@pytest.fixture(scope="module")
def fixture_dir_precomputed(tmp_path_factory) -> Path:
    """Fixture with vbar_kms already present (no component columns needed)."""
    root = tmp_path_factory.mktemp("big_sparc_pre")
    _make_galaxies().to_csv(root / "galaxies.csv", index=False)
    _make_rc_points(include_vbar=True).to_csv(root / "rc_points.csv", index=False)
    return root


# ---------------------------------------------------------------------------
# Tests: contract_utils
# ---------------------------------------------------------------------------

class TestReadTable:
    def test_reads_csv(self, tmp_path):
        df = _make_galaxies()
        p = tmp_path / "g.csv"
        df.to_csv(p, index=False)
        result = read_table(p)
        pd.testing.assert_frame_equal(result, df)

    def test_reads_parquet(self, tmp_path):
        df = _make_galaxies()
        p = tmp_path / "g.parquet"
        df.to_parquet(p, index=False)
        result = read_table(p)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), df)

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_table(tmp_path / "nonexistent.csv")


class TestComputeVbarKms:
    def test_basic_quadrature(self):
        result = compute_vbar_kms(v_gas=3.0, v_disk=4.0)
        assert abs(float(result) - 5.0) < 1e-9

    def test_with_bulge(self):
        result = compute_vbar_kms(v_gas=0.0, v_disk=3.0, v_bul=4.0)
        assert abs(float(result) - 5.0) < 1e-9

    def test_sign_follows_disk(self):
        result = compute_vbar_kms(v_gas=3.0, v_disk=-4.0)
        assert float(result) < 0

    def test_array_input(self):
        v_gas = np.array([3.0, 0.0])
        v_disk = np.array([4.0, 5.0])
        result = compute_vbar_kms(v_gas, v_disk)
        assert result.shape == (2,)
        assert abs(result[0] - 5.0) < 1e-9
        assert abs(result[1] - 5.0) < 1e-9

    def test_returns_finite(self):
        rng = np.random.default_rng(0)
        v_gas = rng.uniform(-50, 50, 100)
        v_disk = rng.uniform(-100, 100, 100)
        result = compute_vbar_kms(v_gas, v_disk)
        assert np.all(np.isfinite(result))


class TestValidateContract:
    def test_valid_df_passes(self):
        df = pd.DataFrame({c: [1.0] for c in CONTRACT_COLUMNS})
        df["galaxy"] = ["G0"]
        validate_contract(df)  # should not raise

    def test_missing_column_raises(self):
        df = pd.DataFrame({c: [1.0] for c in CONTRACT_COLUMNS if c != "vbar_kms"})
        df["galaxy"] = ["G0"]
        with pytest.raises(ValueError, match="vbar_kms"):
            validate_contract(df)


# ---------------------------------------------------------------------------
# Tests: ingest (API)
# ---------------------------------------------------------------------------

class TestIngestAPI:
    def test_returns_dataframe(self, fixture_dir, tmp_path):
        result = ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            tmp_path / "out",
        )
        assert isinstance(result, pd.DataFrame)

    def test_contract_columns_present(self, fixture_dir, tmp_path):
        result = ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            tmp_path / "out",
        )
        assert list(result.columns) == CONTRACT_COLUMNS

    def test_row_count(self, fixture_dir, tmp_path):
        result = ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            tmp_path / "out",
        )
        assert len(result) == N_GALAXIES * N_POINTS_PER_GALAXY

    def test_sorted_by_galaxy_and_radius(self, fixture_dir, tmp_path):
        result = ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            tmp_path / "out",
        )
        expected = result.sort_values(["galaxy", "r_kpc"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_parquet_written(self, fixture_dir, tmp_path):
        out = tmp_path / "out"
        ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            out,
        )
        assert (out / "big_sparc_contract.parquet").exists()

    def test_parquet_readable(self, fixture_dir, tmp_path):
        out = tmp_path / "out"
        ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            out,
        )
        reloaded = pd.read_parquet(out / "big_sparc_contract.parquet")
        assert list(reloaded.columns) == CONTRACT_COLUMNS

    def test_precomputed_vbar_passthrough(self, fixture_dir_precomputed, tmp_path):
        """If vbar_kms already present, component columns are not required."""
        result = ingest(
            fixture_dir_precomputed / "galaxies.csv",
            fixture_dir_precomputed / "rc_points.csv",
            tmp_path / "out",
        )
        assert "vbar_kms" in result.columns

    def test_missing_galaxies_file_raises(self, tmp_path):
        rc = tmp_path / "rc.csv"
        _make_rc_points().to_csv(rc, index=False)
        with pytest.raises(FileNotFoundError):
            ingest(tmp_path / "nonexistent.csv", rc, tmp_path / "out")

    def test_missing_rc_file_raises(self, fixture_dir, tmp_path):
        with pytest.raises(FileNotFoundError):
            ingest(
                fixture_dir / "galaxies.csv",
                tmp_path / "nonexistent.csv",
                tmp_path / "out",
            )

    def test_missing_required_rc_column_raises(self, tmp_path):
        gal = tmp_path / "g.csv"
        _make_galaxies().to_csv(gal, index=False)
        # rc_points without vobs_kms
        rc_bad = _make_rc_points().drop(columns=["vobs_kms"])
        rc_bad_path = tmp_path / "rc_bad.csv"
        rc_bad.to_csv(rc_bad_path, index=False)
        with pytest.raises(ValueError):
            ingest(gal, rc_bad_path, tmp_path / "out")

    def test_no_matching_galaxies_raises(self, tmp_path):
        gal = pd.DataFrame({"galaxy": ["UNKNOWN_X"]})
        gal_path = tmp_path / "g.csv"
        gal.to_csv(gal_path, index=False)
        rc_path = tmp_path / "rc.csv"
        _make_rc_points().to_csv(rc_path, index=False)
        with pytest.raises(ValueError, match="empty"):
            ingest(gal_path, rc_path, tmp_path / "out")

    def test_no_nan_in_output(self, fixture_dir, tmp_path):
        result = ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            tmp_path / "out",
        )
        assert result.notna().all().all(), "Output contains NaN values"

    def test_vbar_positive_for_positive_disk(self, fixture_dir, tmp_path):
        """vbar_kms must be positive when v_disk > 0 (our synthetic data)."""
        result = ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            tmp_path / "out",
        )
        assert (result["vbar_kms"] > 0).all()

    def test_deterministic_output(self, fixture_dir, tmp_path):
        """Running ingest twice on the same input must give identical output."""
        out1 = tmp_path / "out1"
        out2 = tmp_path / "out2"
        r1 = ingest(fixture_dir / "galaxies.csv", fixture_dir / "rc_points.csv", out1)
        r2 = ingest(fixture_dir / "galaxies.csv", fixture_dir / "rc_points.csv", out2)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Tests: ingest via CLI (module and script)
# ---------------------------------------------------------------------------

class TestIngestCLI:
    def test_module_invocation(self, fixture_dir, tmp_path):
        """python -m scripts.ingest_big_sparc_contract must succeed."""
        out = tmp_path / "out_mod"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.ingest_big_sparc_contract",
                "--galaxies",
                str(fixture_dir / "galaxies.csv"),
                "--rc-points",
                str(fixture_dir / "rc_points.csv"),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert (out / "big_sparc_contract.parquet").exists()

    def test_script_invocation(self, fixture_dir, tmp_path):
        """python scripts/ingest_big_sparc_contract.py must succeed."""
        out = tmp_path / "out_script"
        script = str(_REPO_ROOT / "scripts" / "ingest_big_sparc_contract.py")
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--galaxies",
                str(fixture_dir / "galaxies.csv"),
                "--rc-points",
                str(fixture_dir / "rc_points.csv"),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert (out / "big_sparc_contract.parquet").exists()

    def test_module_stdout_contains_count(self, fixture_dir, tmp_path):
        out = tmp_path / "out_stdout"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.ingest_big_sparc_contract",
                "--galaxies",
                str(fixture_dir / "galaxies.csv"),
                "--rc-points",
                str(fixture_dir / "rc_points.csv"),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert str(N_GALAXIES) in result.stdout or str(N_GALAXIES * N_POINTS_PER_GALAXY) in result.stdout


# ---------------------------------------------------------------------------
# Tests: F3 catalog (API)
# ---------------------------------------------------------------------------

class TestF3CatalogAPI:
    @pytest.fixture()
    def parquet_path(self, fixture_dir, tmp_path_factory):
        out = tmp_path_factory.mktemp("ingest_f3")
        ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            out,
        )
        return out / "big_sparc_contract.parquet"

    def test_returns_dataframe(self, parquet_path, tmp_path):
        catalog = generate_catalog(parquet_path, tmp_path / "f3")
        assert isinstance(catalog, pd.DataFrame)

    def test_row_count_equals_galaxy_count(self, parquet_path, tmp_path):
        catalog = generate_catalog(parquet_path, tmp_path / "f3")
        assert len(catalog) == N_GALAXIES

    def test_required_columns(self, parquet_path, tmp_path):
        catalog = generate_catalog(parquet_path, tmp_path / "f3")
        for col in ["galaxy", "n_points", "vflat_kms", "deep_slope", "f3_flag"]:
            assert col in catalog.columns, f"Missing column: {col}"

    def test_f3_catalog_csv_written(self, parquet_path, tmp_path):
        out = tmp_path / "f3"
        generate_catalog(parquet_path, out)
        assert (out / "f3_catalog.csv").exists()

    def test_f3_flag_is_boolean(self, parquet_path, tmp_path):
        catalog = generate_catalog(parquet_path, tmp_path / "f3")
        assert catalog["f3_flag"].dtype == bool

    def test_n_points_correct(self, parquet_path, tmp_path):
        catalog = generate_catalog(parquet_path, tmp_path / "f3")
        assert (catalog["n_points"] == N_POINTS_PER_GALAXY).all()

    def test_vflat_positive(self, parquet_path, tmp_path):
        catalog = generate_catalog(parquet_path, tmp_path / "f3")
        assert (catalog["vflat_kms"] > 0).all()

    def test_min_deep_zero_fills_slope(self, parquet_path, tmp_path):
        """With min_deep=0 and a high vbar_deep threshold, every galaxy gets a slope."""
        catalog = generate_catalog(
            parquet_path, tmp_path / "f3_md0", min_deep=0, vbar_deep=500.0
        )
        assert catalog["deep_slope"].notna().all()

    def test_min_deep_high_yields_nan_slopes(self, parquet_path, tmp_path):
        """With min_deep > N_POINTS_PER_GALAXY no slope should be computable."""
        catalog = generate_catalog(
            parquet_path, tmp_path / "f3_mdhi", min_deep=N_POINTS_PER_GALAXY + 1
        )
        assert catalog["deep_slope"].isna().all()

    def test_min_deep_flag_excludes_nan_slope_galaxies(self, parquet_path, tmp_path):
        """Galaxies with NaN deep_slope must NOT be flagged as F3."""
        catalog = generate_catalog(
            parquet_path, tmp_path / "f3_excl", min_deep=N_POINTS_PER_GALAXY + 1
        )
        assert not catalog["f3_flag"].any()

    def test_deterministic(self, parquet_path, tmp_path):
        c1 = generate_catalog(parquet_path, tmp_path / "f3_det1")
        c2 = generate_catalog(parquet_path, tmp_path / "f3_det2")
        pd.testing.assert_frame_equal(c1, c2)


# ---------------------------------------------------------------------------
# Tests: F3 catalog via CLI (module and script)
# ---------------------------------------------------------------------------

class TestF3CatalogCLI:
    @pytest.fixture()
    def parquet_path(self, fixture_dir, tmp_path_factory):
        out = tmp_path_factory.mktemp("ingest_f3cli")
        ingest(
            fixture_dir / "galaxies.csv",
            fixture_dir / "rc_points.csv",
            out,
        )
        return out / "big_sparc_contract.parquet"

    def test_module_invocation(self, parquet_path, tmp_path):
        out = tmp_path / "f3_mod"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.generate_f3_catalog_from_contract",
                "--input",
                str(parquet_path),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert (out / "f3_catalog.csv").exists()

    def test_script_invocation(self, parquet_path, tmp_path):
        out = tmp_path / "f3_script"
        script = str(_REPO_ROOT / "scripts" / "generate_f3_catalog_from_contract.py")
        result = subprocess.run(
            [
                sys.executable,
                script,
                "--input",
                str(parquet_path),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        assert (out / "f3_catalog.csv").exists()

    def test_min_deep_flag_cli(self, parquet_path, tmp_path):
        """--min-deep must be accepted and reflected in output."""
        out = tmp_path / "f3_md5"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.generate_f3_catalog_from_contract",
                "--input",
                str(parquet_path),
                "--out",
                str(out),
                "--min-deep",
                "5",
            ],
            capture_output=True,
            text=True,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, f"STDERR:\n{result.stderr}"
        catalog = pd.read_csv(out / "f3_catalog.csv")
        assert len(catalog) == N_GALAXIES
