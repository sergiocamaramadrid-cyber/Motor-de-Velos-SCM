from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pilot_f3_test import compute_f3_from_file, main


def _write_rotmod(path: Path, r: np.ndarray, vobs: np.ndarray) -> None:
    # SPARC-like columns: r_kpc Vobs errV Vgas Vdisk Vbul Vbar
    errV = np.full_like(vobs, 1.0, dtype=float)
    zeros = np.zeros_like(vobs, dtype=float)
    data = np.column_stack([r, vobs, errV, zeros, zeros, zeros, zeros])
    np.savetxt(path, data, fmt="%.8e")


def test_compute_f3_powerlaw_exact(tmp_path: Path):
    """
    If V ~ r^a, then logV = const + a logR, so F3 should recover a.
    """
    a = 0.12
    r = np.linspace(0.2, 5.0, 60)
    v = 30.0 * (r ** a)  # perfect power law

    f = tmp_path / "DDO75_rotmod.dat"
    _write_rotmod(f, r, v)

    res = compute_f3_from_file(f, outer_frac=0.7)

    assert res.status in ("ok", "warn")
    assert math.isfinite(res.F3)
    assert abs(res.F3 - a) < 1e-3
    assert res.n_used >= 3


def test_cli_writes_csv(tmp_path: Path):
    sparc = tmp_path / "Rotmod"
    sparc.mkdir(parents=True, exist_ok=True)

    # Build two fake galaxies with known slopes
    for name, a in [("DDO69", 0.05), ("DDO70", 0.15)]:
        r = np.linspace(0.3, 8.0, 80)
        v = 40.0 * (r ** a)
        _write_rotmod(sparc / f"{name}_rotmod.dat", r, v)

    out_csv = tmp_path / "results" / "F3_values.csv"

    rc = main(
        [
            "--sparc-dir",
            str(sparc),
            "--out",
            str(out_csv),
            "--galaxies",
            "DDO69",
            "DDO70",
        ]
    )

    assert rc == 0
    assert out_csv.exists()

    df = pd.read_csv(out_csv)
    assert set(df["galaxy"].tolist()) == {"DDO69", "DDO70"}, (
        f"Expected galaxies DDO69, DDO70 in output; got {df['galaxy'].tolist()}"
    )

    # Check slopes recovered
    f3_map = dict(zip(df["galaxy"], df["F3"]))
    assert abs(f3_map["DDO69"] - 0.05) < 1e-3
    assert abs(f3_map["DDO70"] - 0.15) < 1e-3
