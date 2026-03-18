from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_sparc_radial_csv import build_from_master
from scripts.build_sparc_radial_csv import build_from_rotmod


def test_build_from_master_generates_clean_radial_table(tmp_path: Path) -> None:
    master_csv = tmp_path / "sparc_175_master_sample.csv"
    pd.DataFrame(
        {
            "galaxy": ["G1", "G2"],
            "logSigmaHI_out": [1.0, 1.1],
            "logMbar": [9.7, 9.9],
            "logRd": [0.4, 0.5],
            "f3_scm": [0.48, 0.52],
            "delta_f3": [-0.02, 0.01],
        }
    ).to_csv(master_csv, index=False)

    out = build_from_master(master_csv, n_rings=8)

    assert len(out) == 16
    assert out["galaxy"].nunique() == 2
    assert set(["galaxy", "r", "gbar", "gobs", "SB"]).issubset(out.columns)
    assert np.isfinite(out[["r", "gbar", "gobs", "SB"]].to_numpy()).all()
    assert (out[["r", "gbar", "gobs", "SB"]] > 0).all().all()


def test_build_from_rotmod_maps_columns(tmp_path: Path) -> None:
    sparc_dir = tmp_path / "SPARC" / "rotmod"
    sparc_dir.mkdir(parents=True, exist_ok=True)
    rotmod = sparc_dir / "GAL001_rotmod.dat"
    pd.DataFrame(
        {
            0: [1.0, 2.0, 3.0],
            1: [100.0, 110.0, 120.0],
            2: [0.0, 0.0, 0.0],
            3: [20.0, 20.0, 20.0],
            4: [80.0, 90.0, 100.0],
            5: [10.0, 10.0, 10.0],
        }
    ).to_csv(rotmod, sep=" ", index=False, header=False)

    out = build_from_rotmod(tmp_path / "SPARC")

    assert len(out) == 3
    assert (out["galaxy"] == "GAL001").all()
    expected_sb = np.array([80.0**2 / 1.0, 90.0**2 / 2.0, 100.0**2 / 3.0])
    assert np.allclose(out["SB"].to_numpy(), expected_sb)
