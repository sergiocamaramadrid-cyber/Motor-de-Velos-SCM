"""
tests/test_sparc_contract_ingest.py — Integration tests for ingest_sparc_contract.py.

Uses a small synthetic SPARC-like dataset (5 galaxies, 20 points each) so the
test runs without any real SPARC download.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from contract_utils import REQUIRED_COLS  # noqa: E402
from ingest_sparc_contract import ingest  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic SPARC fixture
# ---------------------------------------------------------------------------

N_GALAXIES = 5
N_PTS = 20
RNG = np.random.default_rng(7)


@pytest.fixture(scope="module")
def sparc_dir(tmp_path_factory):
    """Create a minimal synthetic SPARC directory."""
    root = tmp_path_factory.mktemp("sparc_data")
    galaxy_names = [f"SYN{i:03d}" for i in range(N_GALAXIES)]
    v_flats = np.linspace(100.0, 280.0, N_GALAXIES)

    # Galaxy summary table
    galaxy_table = pd.DataFrame({
        "Galaxy": galaxy_names,
        "D": np.linspace(5.0, 60.0, N_GALAXIES),
        "Inc": np.linspace(30.0, 80.0, N_GALAXIES),
        "L36": 1e9 * np.arange(1, N_GALAXIES + 1, dtype=float),
        "Vflat": v_flats,
        "e_Vflat": np.full(N_GALAXIES, 5.0),
    })
    galaxy_table.to_csv(root / "SPARC_Lelli2016c.csv", index=False)

    # Rotation-curve files
    for name, vf in zip(galaxy_names, v_flats):
        r = np.linspace(0.5, 12.0, N_PTS)
        v_obs = np.full(N_PTS, vf) + RNG.normal(0, 3, N_PTS)
        v_gas = np.linspace(0.10 * vf, 0.35 * vf, N_PTS)
        v_disk = np.linspace(0.55 * vf, 0.80 * vf, N_PTS)
        v_bul = np.zeros(N_PTS)
        SBdisk = np.zeros(N_PTS)
        SBbul = np.zeros(N_PTS)
        rc = pd.DataFrame({
            "r": r, "v_obs": v_obs, "v_obs_err": np.full(N_PTS, 5.0),
            "v_gas": v_gas, "v_disk": v_disk, "v_bul": v_bul,
            "SBdisk": SBdisk, "SBbul": SBbul,
        })
        rc.to_csv(root / f"{name}_rotmod.dat", sep=" ", index=False, header=False)

    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestIngestSparcContractModule:
    def test_returns_dataframe(self, sparc_dir, tmp_path):
        df = ingest(sparc_dir=sparc_dir, out_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_contract_columns_present(self, sparc_dir, tmp_path):
        df = ingest(sparc_dir=sparc_dir, out_dir=tmp_path)
        for col in REQUIRED_COLS:
            assert col in df.columns, f"Missing contract column: {col}"

    def test_row_count(self, sparc_dir, tmp_path):
        df = ingest(sparc_dir=sparc_dir, out_dir=tmp_path)
        assert len(df) == N_GALAXIES * N_PTS

    def test_n_galaxies(self, sparc_dir, tmp_path):
        df = ingest(sparc_dir=sparc_dir, out_dir=tmp_path)
        assert df["galaxy"].nunique() == N_GALAXIES

    def test_sorted_by_galaxy_then_radius(self, sparc_dir, tmp_path):
        df = ingest(sparc_dir=sparc_dir, out_dir=tmp_path)
        for gal, sub in df.groupby("galaxy"):
            assert sub["r_kpc"].is_monotonic_increasing, (
                f"Galaxy {gal}: r_kpc is not sorted"
            )

    def test_writes_parquet(self, sparc_dir, tmp_path):
        ingest(sparc_dir=sparc_dir, out_dir=tmp_path)
        assert (tmp_path / "sparc_contract.parquet").exists()

    def test_dry_run_no_file(self, sparc_dir, tmp_path):
        ingest(sparc_dir=sparc_dir, out_dir=tmp_path, dry_run=True)
        assert not (tmp_path / "sparc_contract.parquet").exists()

    def test_vbar_kms_positive(self, sparc_dir, tmp_path):
        df = ingest(sparc_dir=sparc_dir, out_dir=tmp_path)
        assert (df["vbar_kms"] >= 0).all(), "vbar_kms contains negative values"


class TestIngestSparcContractScript:
    def test_script_runs_and_writes_parquet(self, sparc_dir, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "ingest_sparc_contract.py"),
                "--sparc-dir", str(sparc_dir),
                "--out-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "sparc_contract.parquet").exists()

    def test_script_prints_n_galaxies_and_n_points(self, sparc_dir, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "ingest_sparc_contract.py"),
                "--sparc-dir", str(sparc_dir),
                "--out-dir", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        combined = result.stdout + result.stderr
        assert "N_galaxies" in combined
        assert "N_points" in combined

    def test_script_dry_run(self, sparc_dir, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "ingest_sparc_contract.py"),
                "--sparc-dir", str(sparc_dir),
                "--out-dir", str(tmp_path),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "sparc_contract.parquet").exists()


class TestEndToEndSparcToCatalog:
    """Full pipeline: SPARC dir → contract Parquet → friction-slope catalog Parquet."""

    def test_sparc_to_catalog_parquet(self, sparc_dir, tmp_path):
        # Import catalog generator (already on sys.path via SCRIPTS_DIR)
        from generate_f3_catalog_from_contract import generate_catalog

        # Step 1: ingest
        contract_dir = tmp_path / "contract"
        ingest(sparc_dir=sparc_dir, out_dir=contract_dir)
        assert (contract_dir / "sparc_contract.parquet").exists()

        # Step 2: generate catalog with --data-dir
        out_parquet = tmp_path / "f3_catalog.parquet"
        cat = generate_catalog(
            data_dir=contract_dir,
            out=str(out_parquet),
            min_deep=2,
        )
        assert out_parquet.exists()
        assert len(cat) == N_GALAXIES

        # Verify required columns per problem statement
        required = {"friction_slope", "friction_slope_err", "n_deep", "velo_inerte_flag"}
        assert required.issubset(set(cat.columns)), (
            f"Missing columns: {required - set(cat.columns)}"
        )

    def test_catalog_data_dir_cli(self, sparc_dir, tmp_path):
        """Test --data-dir CLI arg with Parquet output path."""
        contract_dir = tmp_path / "contract2"
        ingest(sparc_dir=sparc_dir, out_dir=contract_dir)

        out_parquet = tmp_path / "f3_out.parquet"
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_f3_catalog_from_contract.py"),
                "--data-dir", str(contract_dir),
                "--out", str(out_parquet),
                "--min-deep", "2",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out_parquet.exists()

        df = pd.read_parquet(out_parquet)
        assert len(df) == N_GALAXIES
        for col in ("friction_slope", "friction_slope_err", "n_deep", "velo_inerte_flag"):
            assert col in df.columns, f"Missing column: {col}"
