from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_full_radial_dataset import build_full_radial_dataset
from scripts.build_full_radial_dataset import merge_radial
from scripts.build_full_radial_dataset import read_dens_zip
from scripts.build_full_radial_dataset import read_rotmod_from_zip


def _write_zip(path: Path, members: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)


def test_read_rotmod_from_zip_computes_dynamic_columns(tmp_path: Path) -> None:
    rot_zip = tmp_path / "rot.zip"
    _write_zip(
        rot_zip,
        {
            "GAL001_rotmod.dat": "\n".join(
                [
                    "1.0 100.0 2.0 20.0 80.0 10.0",
                    "2.0 120.0 2.0 30.0 90.0 20.0",
                ]
            )
        },
    )

    out = read_rotmod_from_zip(str(rot_zip))
    assert len(out) == 2
    assert (out["galaxy"] == "GAL001").all()
    assert {"Vbar", "gobs", "gbar"}.issubset(out.columns)
    assert np.isfinite(out[["r", "Vbar", "gobs", "gbar"]].to_numpy()).all()


def test_read_dens_zip_parses_numeric_tables(tmp_path: Path) -> None:
    dens_zip = tmp_path / "dens.zip"
    _write_zip(
        dens_zip,
        {
            "GAL001.dens": "\n".join(
                [
                    "# comment",
                    "1.0 100.0",
                    "2.0 80.0",
                ]
            )
        },
    )

    out = read_dens_zip(str(dens_zip))
    assert len(out) == 2
    assert set(out.columns) == {"galaxy", "r", "SB"}
    assert np.isfinite(out[["r", "SB"]].to_numpy()).all()
    assert (out["SB"] > 0).all()


def test_merge_radial_uses_interp_and_vdisk_fallback() -> None:
    rot_df = pd.DataFrame(
        {
            "galaxy": ["G1", "G1", "G2"],
            "r": [1.0, 2.0, 1.0],
            "Vdisk": [10.0, 20.0, 30.0],
            "gobs": [1e-11, 2e-11, 3e-11],
            "gbar": [8e-12, 1.8e-11, 2.5e-11],
        }
    )
    dens_df = pd.DataFrame({"galaxy": ["G1", "G1"], "r": [1.0, 2.0], "SB": [100.0, 50.0]})

    out = merge_radial(rot_df, dens_df)
    g1 = out[out["galaxy"] == "G1"].sort_values("r")
    g2 = out[out["galaxy"] == "G2"].sort_values("r")

    assert np.allclose(g1["SB"].to_numpy(), [100.0, 50.0])
    assert np.allclose(g2["SB"].to_numpy(), [30.0**2])


def test_build_full_radial_dataset_end_to_end(tmp_path: Path) -> None:
    rot_zip = tmp_path / "rot.zip"
    ltg_zip = tmp_path / "ltg.zip"
    etg_zip = tmp_path / "etg.zip"
    metadata_csv = tmp_path / "meta.csv"
    out_csv = tmp_path / "sparc_full_radial.csv"

    _write_zip(
        rot_zip,
        {
            "GAL001_rotmod.dat": "\n".join(
                [
                    "1.0 100.0 2.0 20.0 80.0 10.0",
                    "2.0 120.0 2.0 30.0 90.0 20.0",
                ]
            ),
            "GAL002_rotmod.dat": "1.0 90.0 3.0 15.0 70.0 5.0",
        },
    )
    _write_zip(ltg_zip, {"GAL001.dens": "r SB\n1.0 100.0\n2.0 70.0"})
    _write_zip(etg_zip, {"GAL002.dens": "r SB\n1.0 60.0"})

    pd.DataFrame({"galaxy": ["GAL001", "GAL002"], "type": ["LTG", "ETG"]}).to_csv(metadata_csv, index=False)

    out = build_full_radial_dataset(
        rotmod_zip=str(rot_zip),
        ltg_dens_zip=str(ltg_zip),
        etg_dens_zip=str(etg_zip),
        metadata_mrt=str(metadata_csv),
        output=str(out_csv),
    )

    assert out_csv.exists()
    assert len(out) == 3
    assert out["galaxy"].nunique() == 2
    assert {"SB", "gobs", "gbar", "F3"}.issubset(out.columns)
    assert np.isfinite(out[["r", "SB", "gobs", "gbar", "F3"]].to_numpy()).all()
