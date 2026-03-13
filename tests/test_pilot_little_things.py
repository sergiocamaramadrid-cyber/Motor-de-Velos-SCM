from pathlib import Path

import json
import pandas as pd

from scripts.pilot_little_things import (
    REQUIRED_COLS,
    load_catalog,
    load_galaxy_list,
    run_pilot,
)


def _write_catalog(path: Path) -> None:
    df = pd.DataFrame(
        [
            {"galaxy_id": "G1", "logM": 7.0, "logVobs": 1.4, "log_gbar": -11.8, "log_j": 1.2},
            {"galaxy_id": "G2", "logM": 7.3, "logVobs": 1.5, "log_gbar": -11.5, "log_j": 1.4},
            {"galaxy_id": "G3", "logM": 7.6, "logVobs": 1.6, "log_gbar": -11.2, "log_j": 1.6},
        ]
    )
    df.to_csv(path, index=False)


def test_load_catalog_validates_required_columns(tmp_path: Path) -> None:
    bad_catalog = tmp_path / "bad.csv"
    pd.DataFrame([{"galaxy_id": "G1"}]).to_csv(bad_catalog, index=False)

    try:
        load_catalog(bad_catalog)
        raise AssertionError("Expected ValueError for missing required columns")
    except ValueError as exc:
        for col in REQUIRED_COLS:
            if col != "galaxy_id":
                assert col in str(exc)


def test_load_galaxy_list(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog)
    assert load_galaxy_list(catalog) == ["G1", "G2", "G3"]


def test_run_pilot_writes_outputs_and_metadata(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.csv"
    _write_catalog(catalog)
    outdir = tmp_path / "out"

    pred_path, summary_path, metadata_path = run_pilot(
        catalog_path=catalog,
        outdir=outdir,
        n=10,
        seed=42,
    )

    assert pred_path.exists()
    assert summary_path.exists()
    assert metadata_path.exists()

    pred = pd.read_csv(pred_path)
    assert len(pred) == 3
    assert set(["pred_logV_btfr", "pred_logV_interp", "best_model"]).issubset(pred.columns)

    summary = pd.read_csv(summary_path)
    assert set(summary["model"]) == {"btfr", "interp"}
    assert summary["n_galaxies"].nunique() == 1
    assert int(summary["n_galaxies"].iloc[0]) == 3

    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert meta["n_requested"] == 10
    assert meta["n_selected"] == 3
    assert len(meta["sample_ids"]) == 3
