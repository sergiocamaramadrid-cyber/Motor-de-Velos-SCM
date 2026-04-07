from urllib.error import URLError

from scripts import download_sparc_data as mod


def test_download_file_failure_returns_false(monkeypatch, tmp_path):
    dest = tmp_path / "file.dat"

    def fake_urlretrieve(url, filename):
        raise URLError("network down")

    monkeypatch.setattr(mod.urllib.request, "urlretrieve", fake_urlretrieve)

    ok = mod._download_file("https://example.com/file.dat", dest, retries=1)

    assert ok is False
    assert not dest.exists()


def test_parse_galaxy_table_columns(monkeypatch, tmp_path):
    mrt_path = tmp_path / "SPARC_Lelli2016c.mrt"
    # Use the fallback line-scan path: a # Galaxy header followed by data rows
    mrt_path.write_text(
        "# Galaxy\n"
        "NGC2403\n"
        "NGC3198\n"
    )

    # Force the pd.read_csv attempt to fail so the fallback path is exercised
    def fake_read_csv(path, *args, **kwargs):
        raise OSError("forced failure")

    monkeypatch.setattr(mod.pd, "read_csv", fake_read_csv)

    df = mod._parse_galaxy_table(mrt_path)

    assert list(df.columns) == ["Galaxy"]
    assert len(df) == 2
    assert df["Galaxy"].tolist() == ["NGC2403", "NGC3198"]


def test_parse_galaxy_names_from_table(monkeypatch, tmp_path):
    mrt_path = tmp_path / "SPARC_Lelli2016c.mrt"
    # Use the fallback line-scan path: a # Galaxy header followed by data rows
    mrt_path.write_text(
        "# Galaxy\n"
        "UGC06973\n"
        "NGC2403\n"
        "DDO154\n"
    )

    # Force the pd.read_csv attempt to fail so the fallback path is exercised
    def fake_read_csv(path, *args, **kwargs):
        raise OSError("forced failure")

    monkeypatch.setattr(mod.pd, "read_csv", fake_read_csv)

    df = mod._parse_galaxy_table(mrt_path)
    names = df["Galaxy"].tolist()

    assert names == ["UGC06973", "NGC2403", "DDO154"]
