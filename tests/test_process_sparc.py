from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.process_sparc import consolidate_sparc


def test_consolidate_sparc_builds_v1_catalog_from_rotmod_dir(tmp_path):
    sparc_dir = tmp_path / "SPARC"
    rotmod_dir = sparc_dir / "rotmod"
    rotmod_dir.mkdir(parents=True)
    out_csv = tmp_path / "results" / "rotation_curves-v1.0.csv"

    pd.DataFrame(
        [
            [1.0, 100.0, 2.0, 30.0, 40.0, 50.0],
            [2.0, 110.0, 2.5, 35.0, 45.0, 55.0],
        ]
    ).to_csv(rotmod_dir / "NGC2403_rotmod.dat", sep=" ", index=False, header=False)

    out = consolidate_sparc(input_dir=sparc_dir, output_file=out_csv)

    assert out_csv.exists()
    assert list(out.columns) == ["galaxy", "radius", "v_obs", "v_gas", "v_disk", "v_bulge", "v_bar"]
    assert out["galaxy"].nunique() == 1
    expected_vbar = np.sqrt(30.0**2 + 40.0**2 + 50.0**2)
    assert out.loc[0, "v_bar"] == pytest.approx(expected_vbar)


def test_consolidate_sparc_fails_if_no_rotmods(tmp_path):
    (tmp_path / "SPARC" / "rotmod").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"No \*_rotmod\.dat files found"):
        consolidate_sparc(input_dir=tmp_path / "SPARC", output_file=tmp_path / "out.csv")


def test_consolidate_sparc_fails_if_input_directory_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="SPARC input directory not found"):
        consolidate_sparc(input_dir=tmp_path / "SPARC", output_file=tmp_path / "out.csv")


def test_consolidate_sparc_skips_malformed_rotmod_and_keeps_valid_ones(tmp_path):
    sparc_dir = tmp_path / "SPARC"
    rotmod_dir = sparc_dir / "rotmod"
    rotmod_dir.mkdir(parents=True)
    out_csv = tmp_path / "results" / "rotation_curves-v1.0.csv"

    pd.DataFrame([[1.0, 100.0, 2.0, 30.0, 40.0, 50.0]]).to_csv(
        rotmod_dir / "NGC2403_rotmod.dat", sep=" ", index=False, header=False
    )
    (rotmod_dir / "BADGAL_rotmod.dat").write_text("not-a-numeric-rotmod\n", encoding="utf-8")

    out = consolidate_sparc(input_dir=sparc_dir, output_file=out_csv)

    assert out_csv.exists()
    assert sorted(out["galaxy"].unique().tolist()) == ["NGC2403"]
