from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_sparc_175_master import build_sparc_175_master


def _write_rotmod(path, radii, vobs):
    zeros = np.zeros_like(radii)
    data = np.column_stack([radii, vobs, zeros, zeros, zeros])
    np.savetxt(path, data)


def test_build_sparc_175_master_writes_required_contract_and_formulas(tmp_path):
    sparc_dir = tmp_path / "data" / "SPARC"
    metadata_dir = sparc_dir / "metadata"
    rotmod_dir = sparc_dir / "rotmod"
    metadata_dir.mkdir(parents=True)
    rotmod_dir.mkdir(parents=True)

    # Required support files (validated for presence).
    for name in ("MassModels_Lelli2016c.mrt", "RAR.mrt", "RARbins.mrt", "CDR_Lelli2016b.mrt"):
        pd.DataFrame([{"Galaxy": "G1"}]).to_csv(metadata_dir / name, index=False)

    pd.DataFrame(
        [
            {"Galaxy": "G1", "MHI": 1.0, "RHI": 10.0, "L_3.6": 2.0, "Rdisk": 2.5, "Inc": 60.0},
            {"Galaxy": "G2", "MHI": 2.0, "RHI": 20.0, "L_3.6": 8.0, "Rdisk": 4.0, "Inc": 75.0},
        ]
    ).to_csv(metadata_dir / "SPARC_Lelli2016c.mrt", index=False)

    r = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    _write_rotmod(rotmod_dir / "G1_rotmod.dat", r, 100.0 * (r ** -0.2))
    _write_rotmod(rotmod_dir / "G2_rotmod.dat", r, 90.0 * (r ** -0.1))

    out_path = tmp_path / "data" / "sparc_175_master.csv"
    out = build_sparc_175_master(sparc_dir=sparc_dir, out_path=out_path)

    assert out_path.exists()
    assert list(out.columns) == [
        "galaxy",
        "deep_slope",
        "delta_f3",
        "n_tail_points",
        "tail_r_min",
        "tail_r_max",
        "logSigmaHI_out",
        "logMstar",
        "logRd",
        "inclination",
    ]
    assert len(out) == 2

    g1 = out.loc[out["galaxy"] == "G1"].iloc[0]
    assert g1["deep_slope"] == pytest.approx(-0.2, abs=1e-10)
    assert g1["delta_f3"] == pytest.approx(0.3, abs=1e-10)
    assert int(g1["n_tail_points"]) == 5
    assert g1["tail_r_min"] == pytest.approx(2.0)
    assert g1["tail_r_max"] == pytest.approx(6.0)
    expected_sigma = (1.0 * 1e9) / (np.pi * (10.0**2) * 1e6)
    assert g1["logSigmaHI_out"] == pytest.approx(np.log10(expected_sigma))
    assert g1["logMstar"] == pytest.approx(np.log10(0.5 * 2.0) + 9.0)
    assert g1["logRd"] == pytest.approx(np.log10(2.5))
    assert g1["inclination"] == pytest.approx(60.0)


def test_build_sparc_175_master_requires_support_tables(tmp_path):
    sparc_dir = tmp_path / "data" / "SPARC"
    metadata_dir = sparc_dir / "metadata"
    rotmod_dir = sparc_dir / "rotmod"
    metadata_dir.mkdir(parents=True)
    rotmod_dir.mkdir(parents=True)

    pd.DataFrame([{"Galaxy": "G1", "MHI": 1.0, "RHI": 10.0, "L_3.6": 2.0, "Rdisk": 2.5, "Inc": 60.0}]).to_csv(
        metadata_dir / "SPARC_Lelli2016c.mrt", index=False
    )
    _write_rotmod(rotmod_dir / "G1_rotmod.dat", np.array([1, 2, 3], dtype=float), np.array([100, 90, 80], dtype=float))

    with pytest.raises(FileNotFoundError, match="Missing SPARC support table\\(s\\)"):
        build_sparc_175_master(
            sparc_dir=sparc_dir,
            out_path=tmp_path / "data" / "sparc_175_master.csv",
        )
