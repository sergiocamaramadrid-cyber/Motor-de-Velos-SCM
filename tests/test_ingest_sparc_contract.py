import pandas as pd
from pathlib import Path
import subprocess
import sys


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, (
        f"Command failed: {cmd}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    )
    return r.stdout


def test_ingest_sparc_contract(tmp_path: Path):
    sparc_dir = tmp_path / "SPARC"
    out_dir = tmp_path / "out"
    sparc_dir.mkdir()

    # Minimal SPARC-like rotmod with header + whitespace columns
    # Need Rad, Vobs, eVobs, Vgas, Vdisk, Vbul
    content = """# comment
Rad Vobs eVobs Vgas Vdisk Vbul
0.5  40.0 2.0  20.0 30.0 0.0
1.0  50.0 2.0  25.0 35.0 0.0
2.0  55.0 3.0  30.0 40.0 0.0
"""
    (sparc_dir / "TEST_rotmod.dat").write_text(content)

    run([sys.executable, "scripts/ingest_sparc_contract.py",
         "--sparc-dir", str(sparc_dir),
         "--out-dir", str(out_dir)])

    assert (out_dir / "galaxies.parquet").exists()
    assert (out_dir / "rc_points.parquet").exists()

    df_gal = pd.read_parquet(out_dir / "galaxies.parquet")
    df_rc = pd.read_parquet(out_dir / "rc_points.parquet")

    assert list(df_gal.columns) == ["galaxy_id"]
    assert df_gal.iloc[0]["galaxy_id"] == "TEST"

    # contract columns exist
    assert "galaxy_id" in df_rc.columns
    assert "r_kpc" in df_rc.columns
    assert "vrot_kms" in df_rc.columns

    # should have components for g_bar computation
    assert ("vgas_kms" in df_rc.columns) or ("vbar_kms" in df_rc.columns)
    assert ("vstar_kms" in df_rc.columns) or ("vbar_kms" in df_rc.columns)
