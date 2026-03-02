import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_script(cmd, cwd=None):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    assert r.returncode == 0, f"Command failed: {cmd}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    return r.stdout


def _make_synthetic_data(raw: Path) -> None:
    """Write synthetic galaxies.csv and rc_points.csv into raw/."""
    df_gal = pd.DataFrame({"galaxy_id": ["G1", "G2"]})
    df_gal.to_csv(raw / "galaxies.csv", index=False)

    r = np.array([1, 2, 3, 4], dtype=float)
    v = np.array([50, 50, 50, 50], dtype=float)
    df_rc = pd.DataFrame({
        "galaxy_id": ["G1"] * 4 + ["G2"] * 4,
        "r_kpc": np.concatenate([r, r]),
        "vrot_kms": np.concatenate([v, v]),
        "vbar_kms": np.concatenate([v, v]),
    })
    df_rc.to_csv(raw / "rc_points.csv", index=False)


def test_ingest_and_generate_beta(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    raw.mkdir()
    _make_synthetic_data(raw)

    # ingest - module style (python -m scripts.ingest_big_sparc_contract)
    run_script([sys.executable, "-m", "scripts.ingest_big_sparc_contract",
                "--input-dir", str(raw),
                "--out-dir", str(out)],
               cwd=str(REPO_ROOT))

    assert (out / "galaxies.parquet").exists()
    assert (out / "rc_points.parquet").exists()

    # generate catalog - module style
    cat = tmp_path / "f3.parquet"
    run_script([sys.executable, "-m", "scripts.generate_f3_catalog_from_contract",
                "--data-dir", str(out),
                "--out", str(cat),
                "--min-deep", "2"],
               cwd=str(REPO_ROOT))

    df = pd.read_parquet(cat)
    assert "friction_slope" in df.columns
    assert "friction_slope_err" in df.columns
    assert "velo_inerte_flag" in df.columns


def test_ingest_and_generate_beta_script_style(tmp_path: Path):
    """Same pipeline using direct script invocation (python scripts/*.py)."""
    raw = tmp_path / "raw"
    out = tmp_path / "processed"
    raw.mkdir()
    _make_synthetic_data(raw)

    # ingest - script style (python scripts/ingest_big_sparc_contract.py)
    run_script([sys.executable, "scripts/ingest_big_sparc_contract.py",
                "--input-dir", str(raw),
                "--out-dir", str(out)],
               cwd=str(REPO_ROOT))

    assert (out / "galaxies.parquet").exists()
    assert (out / "rc_points.parquet").exists()

    # generate catalog - script style
    cat = tmp_path / "f3.parquet"
    run_script([sys.executable, "scripts/generate_f3_catalog_from_contract.py",
                "--data-dir", str(out),
                "--out", str(cat),
                "--min-deep", "2"],
               cwd=str(REPO_ROOT))

    df = pd.read_parquet(cat)
    assert "friction_slope" in df.columns
    assert "friction_slope_err" in df.columns
    assert "velo_inerte_flag" in df.columns
