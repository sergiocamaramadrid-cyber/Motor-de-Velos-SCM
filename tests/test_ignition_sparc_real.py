from pathlib import Path


def test_ignition_rotmod_directory_exists() -> None:
    data_dir = Path("data/SPARC/rotmod")
    assert data_dir.exists()
