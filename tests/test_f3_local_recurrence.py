from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts import test_f3_local_recurrence


def _write_rotmod(path: Path, n: int = 16) -> None:
    r = np.linspace(0.5, 8.0, n)
    vobs = 60.0 * np.power(r, 0.55 + 0.05 * np.sin(r))
    err = np.full(n, 2.0)
    vgas = 20.0 * np.power(r, 0.20)
    vdisk = 35.0 * np.power(r, 0.15)
    vbul = 10.0 * np.power(r, 0.10)
    arr = np.column_stack([r, vobs, err, vgas, vdisk, vbul])
    np.savetxt(path, arr, fmt="%.8f")


def test_local_slope_loglog_recovers_power_law() -> None:
    r = np.linspace(1.0, 10.0, 21)
    v = r**0.7
    slopes = test_f3_local_recurrence.local_slope_loglog(r=r, v=v, window=5)

    mid = slopes[np.isfinite(slopes)]
    assert len(mid) > 0
    assert np.allclose(mid, 0.7, atol=1e-3)


def test_main_writes_expected_outputs(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "rotmod"
    out_dir = tmp_path / "results"
    data_dir.mkdir(parents=True, exist_ok=True)

    _write_rotmod(data_dir / "G1_rotmod.dat", n=16)
    _write_rotmod(data_dir / "G2_rotmod.dat", n=18)

    monkeypatch.setattr(
        "sys.argv",
        [
            "test_f3_local_recurrence.py",
            "--data_dir",
            str(data_dir),
            "--out_dir",
            str(out_dir),
            "--window",
            "5",
            "--min_points",
            "10",
            "--bootstrap",
            "20",
        ],
    )
    test_f3_local_recurrence.main()

    summary = json.loads((out_dir / "executive_summary.json").read_text(encoding="utf-8"))

    assert summary["n_files_found"] == 2
    assert summary["n_galaxies"] == 2

    expected_files = [
        "per_galaxy_f3_recurrence.csv",
        "top20_f3_recurrence_improve.csv",
        "top20_f3_recurrence_worsen.csv",
        "skipped_galaxies.csv",
        "executive_summary.json",
    ]
    for name in expected_files:
        assert (out_dir / name).exists(), f"Missing output file: {name}"


def test_main_with_sparc_rotmod_subdir(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "SPARC"
    rotmod_dir = data_dir / "rotmod"
    out_dir = tmp_path / "results_root_mode"
    rotmod_dir.mkdir(parents=True, exist_ok=True)

    _write_rotmod(rotmod_dir / "G1_rotmod.dat", n=16)
    _write_rotmod(rotmod_dir / "G2_rotmod.dat", n=18)

    monkeypatch.setattr(
        "sys.argv",
        [
            "test_f3_local_recurrence.py",
            "--data_dir",
            str(data_dir),
            "--out_dir",
            str(out_dir),
            "--window",
            "5",
            "--min_points",
            "10",
            "--bootstrap",
            "20",
        ],
    )
    test_f3_local_recurrence.main()

    summary = json.loads((out_dir / "executive_summary.json").read_text(encoding="utf-8"))
    assert summary["n_files_found"] == 2
    assert summary["n_galaxies"] == 2
