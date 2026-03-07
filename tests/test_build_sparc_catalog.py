from __future__ import annotations

import pandas as pd
import pytest

from scripts import build_sparc_catalog


def test_load_tables_reads_expected_files(tmp_path, monkeypatch):
    metadata_dir = tmp_path / "data" / "SPARC" / "metadata"
    metadata_dir.mkdir(parents=True)

    pd.DataFrame([{"Galaxy": "NGC2403", "Distance": 3.2}]).to_csv(
        metadata_dir / "SPARC_Lelli2016c.mrt", index=False
    )
    pd.DataFrame([{"Galaxy": "NGC2403", "Vflat": 130.0}]).to_csv(
        metadata_dir / "CDR_Lelli2016b.mrt", index=False
    )
    pd.DataFrame([{"Galaxy": "NGC2403", "Mbar": 1.1e10}]).to_csv(
        metadata_dir / "BTFR_Lelli2019.mrt", index=False
    )
    pd.DataFrame([{"Galaxy": "NGC2403", "Vbar": 140.0}]).to_csv(
        metadata_dir / "MassModels_Lelli2016c.mrt", index=False
    )

    monkeypatch.setattr(build_sparc_catalog, "DATA_DIR", metadata_dir)

    meta, cdr, btfr = build_sparc_catalog.load_tables()

    assert list(meta.columns) == ["Galaxy", "Distance"]
    assert list(cdr.columns) == ["Galaxy", "Vflat"]
    assert list(btfr.columns) == ["Galaxy", "Mbar"]


def test_build_catalog_merges_tables_and_writes_csv(tmp_path, monkeypatch):
    metadata_dir = tmp_path / "data" / "SPARC" / "metadata"
    out_dir = tmp_path / "data" / "SPARC"
    metadata_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"Galaxy": "NGC2403", "Distance": 3.2, "Inclination": 61.0},
            {"Galaxy": "NGC3198", "Distance": 13.8, "Inclination": 71.0},
        ]
    ).to_csv(metadata_dir / "SPARC_Lelli2016c.mrt", index=False)
    pd.DataFrame(
        [
            {"Galaxy": " NGC2403 ", "Vflat": 130.0, "Rmax": 18.0},
        ]
    ).to_csv(metadata_dir / "CDR_Lelli2016b.mrt", index=False)
    pd.DataFrame(
        [
            {"Galaxy": "NGC2403", "Mbar": 1.1e10},
            {"Galaxy": "NGC3198", "Mbar": 2.2e10},
        ]
    ).to_csv(metadata_dir / "BTFR_Lelli2019.mrt", index=False)
    pd.DataFrame(
        [
            {"Galaxy": "NGC2403", "Vbar": 140.0},
            {"Galaxy": "NGC3198", "Vbar": 150.0},
        ]
    ).to_csv(metadata_dir / "MassModels_Lelli2016c.mrt", index=False)

    monkeypatch.setattr(build_sparc_catalog, "DATA_DIR", metadata_dir)
    monkeypatch.setattr(build_sparc_catalog, "OUT_DIR", out_dir)

    out = build_sparc_catalog.build_catalog()
    saved = pd.read_csv(out_dir / "sparc_master_catalog.csv")

    assert len(out) == 2
    assert len(saved) == 2
    assert saved.loc[saved["Galaxy"] == "NGC2403", "Vflat"].iloc[0] == 130.0
    assert pd.isna(saved.loc[saved["Galaxy"] == "NGC3198", "Vflat"]).item()


def test_load_tables_raises_when_required_file_is_missing(tmp_path, monkeypatch):
    metadata_dir = tmp_path / "data" / "SPARC" / "metadata"
    metadata_dir.mkdir(parents=True)

    pd.DataFrame([{"Galaxy": "NGC2403", "Distance": 3.2}]).to_csv(
        metadata_dir / "SPARC_Lelli2016c.mrt", index=False
    )
    pd.DataFrame([{"Galaxy": "NGC2403", "Vflat": 130.0}]).to_csv(
        metadata_dir / "CDR_Lelli2016b.mrt", index=False
    )

    monkeypatch.setattr(build_sparc_catalog, "DATA_DIR", metadata_dir)

    with pytest.raises(FileNotFoundError, match="Missing SPARC metadata table\\(s\\):"):
        build_sparc_catalog.load_tables()
