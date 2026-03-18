import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts.build_sparc_radial_csv import KPC_TO_M
from scripts.build_sparc_radial_csv import MIN_SB
from scripts.build_sparc_radial_csv import read_rotmod_zip


def _write_zip(path: Path, files: dict[str, str]) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)


def test_read_rotmod_zip_builds_expected_columns_and_values(tmp_path: Path) -> None:
    zip_path = tmp_path / "rotmods.zip"
    _write_zip(
        zip_path,
        {
            "GAL001_rotmod.dat": "\n".join(
                [
                    "# r Vobs eVobs Vgas Vdisk Vbul",
                    "1.0 100.0 5.0 20.0 80.0 10.0",
                    "2.0 110.0 5.5 20.0 90.0 10.0",
                ]
            )
        },
    )

    out = read_rotmod_zip(str(zip_path))

    assert list(out.columns) == [
        "galaxy",
        "r",
        "Vobs",
        "eVobs",
        "Vgas",
        "Vdisk",
        "Vbul",
        "Vbar",
        "gobs",
        "gbar",
        "SB",
        "F3",
    ]
    assert len(out) == 2
    assert out["galaxy"].tolist() == ["GAL001", "GAL001"]
    expected_sb = np.maximum(np.array([80.0**2, 90.0**2]), MIN_SB)
    assert np.allclose(out["SB"].to_numpy(), expected_sb)

    r_m = np.array([1.0, 2.0]) * KPC_TO_M
    expected_gobs = (np.array([100.0, 110.0]) * 1000.0) ** 2 / r_m
    assert np.allclose(out["gobs"].to_numpy(), expected_gobs)


def test_read_rotmod_zip_filters_non_physical_and_invalid_rows(tmp_path: Path) -> None:
    zip_path = tmp_path / "rotmods.zip"
    _write_zip(
        zip_path,
        {
            "G_bad_rotmod.dat": "\n".join(
                [
                    "0.0 100.0 5.0 20.0 80.0 10.0",  # filtered: r <= 0
                    "1.0 nan 5.0 20.0 80.0 10.0",  # filtered: non-finite gobs
                    "2.0 120.0 5.0 20.0 80.0 10.0",  # kept
                ]
            )
        },
    )

    out = read_rotmod_zip(str(zip_path))
    assert len(out) == 1
    assert out.iloc[0]["r"] == 2.0


def test_read_rotmod_zip_raises_when_zip_has_no_rotmod_files(tmp_path: Path) -> None:
    zip_path = tmp_path / "empty.zip"
    _write_zip(zip_path, {"README.txt": "nothing useful"})

    with pytest.raises(ValueError, match=r"No \*_rotmod\.dat files found inside the ZIP\."):
        read_rotmod_zip(str(zip_path))
