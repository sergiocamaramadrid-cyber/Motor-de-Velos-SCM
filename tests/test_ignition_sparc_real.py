from pathlib import Path


def test_rotmod_dir_path_is_expected() -> None:
    data_dir = Path("data/SPARC/rotmod")
    assert str(data_dir) == "data/SPARC/rotmod"
