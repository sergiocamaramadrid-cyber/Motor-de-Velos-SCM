from pathlib import Path


def test_rotmod_exists_after_ingestion() -> None:
    data_dir = Path("data/SPARC/rotmod")
    assert data_dir.exists()
