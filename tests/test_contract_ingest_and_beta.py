"""
tests/test_contract_ingest_and_beta.py — End-to-end integration test.

Covers:
  1. contract_utils.validate_contract / compute_vbar_kms
  2. ingest_big_sparc_contract.ingest  (module path)
  3. ingest_big_sparc_contract via direct script invocation (subprocess)
  4. generate_f3_catalog_from_contract.generate_catalog  (module path)
  5. generate_f3_catalog_from_contract via direct script invocation (subprocess)

A small synthetic BIG-SPARC-like dataset (3 galaxies, 30 points each) is used
so the test runs entirely in memory / tmp without any external data download.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Locate scripts dir so we can import from it without installing the package
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from contract_utils import (  # noqa: E402
    REQUIRED_COLS,
    compute_vbar_kms,
    validate_contract,
)
from ingest_big_sparc_contract import ingest  # noqa: E402
from generate_f3_catalog_from_contract import generate_catalog  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic dataset fixture
# ---------------------------------------------------------------------------

N_GALAXIES = 3
N_PTS = 30
RNG = np.random.default_rng(42)


def _make_synthetic_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (galaxies, rc_points) DataFrames mimicking BIG-SPARC layout."""
    galaxy_names = [f"BGAL{i:03d}" for i in range(N_GALAXIES)]
    v_flats = [120.0, 180.0, 250.0]

    galaxies = pd.DataFrame({
        "galaxy": galaxy_names,
        "distance_mpc": [10.0, 25.0, 50.0],
    })

    rows = []
    for name, vf in zip(galaxy_names, v_flats):
        r = np.linspace(0.3, 20.0, N_PTS)
        vobs = vf + RNG.normal(0, 3, N_PTS)
        vobs_err = np.full(N_PTS, 5.0)
        # Component velocities that vary with radius so vbar is not constant,
        # enabling a meaningful β fit for each galaxy.
        vgas = np.linspace(0.10 * vf, 0.40 * vf, N_PTS)
        vdisk = np.linspace(0.50 * vf, 0.80 * vf, N_PTS)
        vbul = np.zeros(N_PTS)
        for i in range(N_PTS):
            rows.append({
                "galaxy": name,
                "r_kpc": r[i],
                "vobs_kms": vobs[i],
                "vobs_err_kms": vobs_err[i],
                "vgas_kms": vgas[i],
                "vdisk_kms": vdisk[i],
                "vbul_kms": vbul[i],
            })

    rc_points = pd.DataFrame(rows)
    return galaxies, rc_points


@pytest.fixture(scope="module")
def synthetic_data_dir(tmp_path_factory):
    """Write synthetic BIG-SPARC tables to a tmp directory."""
    d = tmp_path_factory.mktemp("big_sparc_data")
    galaxies, rc_points = _make_synthetic_tables()
    galaxies.to_csv(d / "galaxies.csv", index=False)
    rc_points.to_csv(d / "rc_points.csv", index=False)
    return d


# ---------------------------------------------------------------------------
# 1. contract_utils unit tests
# ---------------------------------------------------------------------------


class TestContractUtils:
    def test_validate_contract_ok(self):
        df = pd.DataFrame({col: [1.0] for col in REQUIRED_COLS})
        validate_contract(df)  # should not raise

    def test_validate_contract_missing(self):
        df = pd.DataFrame({"galaxy": ["G1"], "r_kpc": [1.0]})
        with pytest.raises(ValueError, match="missing columns"):
            validate_contract(df)

    def test_compute_vbar_kms_from_components(self):
        df = pd.DataFrame({
            "vgas_kms": [30.0],
            "vdisk_kms": [40.0],
            "vbul_kms": [0.0],
        })
        result = compute_vbar_kms(df)
        assert "vbar_kms" in result.columns
        expected = np.sqrt(30.0 ** 2 + 40.0 ** 2)
        assert abs(result["vbar_kms"].iloc[0] - expected) < 1e-9

    def test_compute_vbar_kms_passthrough(self):
        """If vbar_kms already exists, it must be left unchanged."""
        df = pd.DataFrame({"vbar_kms": [99.0], "vgas_kms": [10.0]})
        result = compute_vbar_kms(df)
        assert result["vbar_kms"].iloc[0] == 99.0


# ---------------------------------------------------------------------------
# 2. ingest_big_sparc_contract — module path
# ---------------------------------------------------------------------------


class TestIngestModule:
    def test_ingest_returns_dataframe(self, synthetic_data_dir, tmp_path):
        df = ingest(data_dir=synthetic_data_dir, out_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)

    def test_ingest_contract_columns(self, synthetic_data_dir, tmp_path):
        df = ingest(data_dir=synthetic_data_dir, out_dir=tmp_path)
        for col in REQUIRED_COLS:
            assert col in df.columns, f"Missing contract column: {col}"

    def test_ingest_row_count(self, synthetic_data_dir, tmp_path):
        df = ingest(data_dir=synthetic_data_dir, out_dir=tmp_path)
        assert len(df) == N_GALAXIES * N_PTS

    def test_ingest_sorted(self, synthetic_data_dir, tmp_path):
        df = ingest(data_dir=synthetic_data_dir, out_dir=tmp_path)
        assert list(df["galaxy"]) == sorted(df["galaxy"].tolist())

    def test_ingest_writes_parquet(self, synthetic_data_dir, tmp_path):
        ingest(data_dir=synthetic_data_dir, out_dir=tmp_path)
        assert (tmp_path / "big_sparc_contract.parquet").exists()

    def test_ingest_dry_run_no_file(self, synthetic_data_dir, tmp_path):
        ingest(data_dir=synthetic_data_dir, out_dir=tmp_path, dry_run=True)
        assert not (tmp_path / "big_sparc_contract.parquet").exists()


# ---------------------------------------------------------------------------
# 3. ingest_big_sparc_contract — direct script invocation
# ---------------------------------------------------------------------------


class TestIngestScript:
    def test_script_runs_and_writes_parquet(self, synthetic_data_dir, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "ingest_big_sparc_contract.py"),
                "--data-dir", str(synthetic_data_dir),
                "--out", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "big_sparc_contract.parquet").exists()

    def test_script_dry_run(self, synthetic_data_dir, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "ingest_big_sparc_contract.py"),
                "--data-dir", str(synthetic_data_dir),
                "--out", str(tmp_path),
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "big_sparc_contract.parquet").exists()


# ---------------------------------------------------------------------------
# 4. generate_f3_catalog_from_contract — module path
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def contract_parquet(synthetic_data_dir, tmp_path_factory):
    """Ingest synthetic data once and return the contract Parquet path."""
    out = tmp_path_factory.mktemp("contract_out")
    ingest(data_dir=synthetic_data_dir, out_dir=out)
    return out / "big_sparc_contract.parquet"


class TestGenerateCatalogModule:
    def test_catalog_returns_dataframe(self, contract_parquet, tmp_path):
        cat = generate_catalog(
            contract_path=contract_parquet,
            out_dir=tmp_path,
            min_deep=2,
        )
        assert isinstance(cat, pd.DataFrame)

    def test_catalog_has_expected_columns(self, contract_parquet, tmp_path):
        cat = generate_catalog(
            contract_path=contract_parquet,
            out_dir=tmp_path,
            min_deep=2,
        )
        expected = [
            "galaxy", "n_total", "n_deep",
            "friction_slope", "friction_slope_err",
            "r_value", "p_value", "delta_from_mond",
            "velo_inerte_flag", "verdict",
        ]
        assert cat.columns.tolist() == expected

    def test_catalog_one_row_per_galaxy(self, contract_parquet, tmp_path):
        cat = generate_catalog(
            contract_path=contract_parquet,
            out_dir=tmp_path,
            min_deep=2,
        )
        assert len(cat) == N_GALAXIES

    def test_catalog_writes_parquet(self, contract_parquet, tmp_path):
        generate_catalog(
            contract_path=contract_parquet,
            out_dir=tmp_path,
            min_deep=2,
        )
        assert (tmp_path / "f3_beta_catalog.parquet").exists()

    def test_catalog_min_deep_flag(self, contract_parquet, tmp_path):
        """--min-deep controls the 'unreliable' verdict threshold.

        When min_deep exceeds the actual deep-point count, the verdict must
        contain the 'Only N deep points' reliability warning.  When min_deep=1
        (always satisfied), that specific warning must be absent.
        """
        # With deep_threshold=2.0, all 30 points per galaxy are "deep"
        # (vbar < 2×210 km/s for all synthetic galaxies).
        cat_strict = generate_catalog(
            contract_path=contract_parquet,
            out_dir=tmp_path,
            min_deep=1000,       # above dataset size → reliability warning on every row
            deep_threshold=2.0,
        )
        # Every verdict must carry the "Only N deep points" reliability warning.
        unreliable_msg = cat_strict["verdict"].str.contains(
            "deep points — result may not be reliable", na=False
        )
        assert unreliable_msg.all(), (
            "Expected 'Only N deep points' warning in every row with min_deep=1000"
        )

        cat_relaxed = generate_catalog(
            contract_path=contract_parquet,
            out_dir=tmp_path,
            min_deep=1,           # low threshold — satisfied whenever ≥1 deep point exists
            deep_threshold=2.0,
        )
        # None of the verdicts should carry the reliability warning.
        unreliable_msg_relaxed = cat_relaxed["verdict"].str.contains(
            "deep points — result may not be reliable", na=False
        )
        assert not unreliable_msg_relaxed.any(), (
            "Expected no 'Only N deep points' warning with min_deep=1"
        )


# ---------------------------------------------------------------------------
# 5. generate_f3_catalog_from_contract — direct script invocation
# ---------------------------------------------------------------------------


class TestGenerateCatalogScript:
    def test_script_runs_and_writes_parquet(self, contract_parquet, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS_DIR / "generate_f3_catalog_from_contract.py"),
                "--contract", str(contract_parquet),
                "--out", str(tmp_path),
                "--min-deep", "2",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "f3_beta_catalog.parquet").exists()

    def test_script_module_invocation(self, contract_parquet, tmp_path):
        result = subprocess.run(
            [
                sys.executable,
                "-m", "generate_f3_catalog_from_contract",
                "--contract", str(contract_parquet),
                "--out", str(tmp_path),
                "--min-deep", "2",
            ],
            capture_output=True,
            text=True,
            cwd=str(SCRIPTS_DIR),
        )
        assert result.returncode == 0, result.stderr
        assert (tmp_path / "f3_beta_catalog.parquet").exists()
