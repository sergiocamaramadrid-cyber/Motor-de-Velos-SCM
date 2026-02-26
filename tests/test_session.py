"""
tests/test_session.py — Unit tests for src/session.py.

Covers Session creation, serialisation, round-trip JSON loading,
checksum computation, and integration with run_pipeline.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.session import Session, create_session, _sha256, _git_commit


# ---------------------------------------------------------------------------
# _sha256
# ---------------------------------------------------------------------------

class TestSha256:
    def test_known_content(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_bytes(b"hello")
        import hashlib
        expected = hashlib.sha256(b"hello").hexdigest()
        assert _sha256(p) == expected

    def test_different_content_different_digest(self, tmp_path):
        p1 = tmp_path / "a.txt"
        p2 = tmp_path / "b.txt"
        p1.write_bytes(b"aaa")
        p2.write_bytes(b"bbb")
        assert _sha256(p1) != _sha256(p2)

    def test_returns_hex_string(self, tmp_path):
        p = tmp_path / "f.bin"
        p.write_bytes(b"\x00\xff")
        digest = _sha256(p)
        assert isinstance(digest, str)
        assert len(digest) == 64
        int(digest, 16)  # must be valid hex


# ---------------------------------------------------------------------------
# _git_commit
# ---------------------------------------------------------------------------

class TestGitCommit:
    def test_returns_str_or_none(self):
        result = _git_commit()
        assert result is None or isinstance(result, str)

    def test_if_str_looks_like_sha(self):
        result = _git_commit()
        if result is not None:
            assert len(result) == 40
            int(result, 16)  # valid hex


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

class TestSession:
    def _make_session(self):
        return Session(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit="abc123" * 6 + "ab",  # 40-char fake SHA
            parameters={"a0": 1.2e-10},
            input_checksums={"SPARC_Lelli2016c.csv": "deadbeef" * 8},
        )

    def test_to_dict_has_required_keys(self):
        s = self._make_session()
        d = s.to_dict()
        for key in ("session_id", "timestamp", "git_commit",
                    "parameters", "input_checksums"):
            assert key in d

    def test_save_creates_file(self, tmp_path):
        s = self._make_session()
        path = tmp_path / "session.json"
        s.save(path)
        assert path.exists()

    def test_save_valid_json(self, tmp_path):
        s = self._make_session()
        path = tmp_path / "session.json"
        s.save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_round_trip(self, tmp_path):
        s = self._make_session()
        path = tmp_path / "session.json"
        s.save(path)
        s2 = Session.load(path)
        assert s2.session_id == s.session_id
        assert s2.timestamp == s.timestamp
        assert s2.git_commit == s.git_commit
        assert s2.parameters == s.parameters
        assert s2.input_checksums == s.input_checksums

    def test_none_git_commit_round_trip(self, tmp_path):
        s = Session(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit=None,
            parameters={},
            input_checksums={},
        )
        path = tmp_path / "session.json"
        s.save(path)
        s2 = Session.load(path)
        assert s2.git_commit is None

    def test_parameters_preserved(self, tmp_path):
        s = Session(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            git_commit=None,
            parameters={"a0": 1.5e-10, "flag": True},
            input_checksums={},
        )
        path = tmp_path / "session.json"
        s.save(path)
        s2 = Session.load(path)
        assert s2.parameters["a0"] == pytest.approx(1.5e-10)
        assert s2.parameters["flag"] is True


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir(tmp_path):
    """Minimal data directory with a SPARC galaxy-table CSV."""
    galaxy_csv = tmp_path / "SPARC_Lelli2016c.csv"
    pd.DataFrame({
        "Galaxy": ["NGC0001"],
        "D": [10.0],
        "Inc": [45.0],
        "L36": [1e9],
        "Vflat": [150.0],
        "e_Vflat": [5.0],
    }).to_csv(galaxy_csv, index=False)
    return tmp_path


class TestCreateSession:
    def test_returns_session_instance(self, data_dir):
        s = create_session(data_dir, {"a0": 1.2e-10})
        assert isinstance(s, Session)

    def test_session_id_is_uuid(self, data_dir):
        s = create_session(data_dir, {"a0": 1.2e-10})
        parsed = uuid.UUID(s.session_id)
        assert parsed.version == 4

    def test_timestamp_is_iso8601(self, data_dir):
        s = create_session(data_dir, {"a0": 1.2e-10})
        # Should not raise
        datetime.fromisoformat(s.timestamp)

    def test_timestamp_is_utc(self, data_dir):
        s = create_session(data_dir, {"a0": 1.2e-10})
        ts = datetime.fromisoformat(s.timestamp)
        assert ts.tzinfo is not None

    def test_parameters_recorded(self, data_dir):
        s = create_session(data_dir, {"a0": 2.5e-10})
        assert s.parameters["a0"] == pytest.approx(2.5e-10)

    def test_checksum_recorded_for_galaxy_table(self, data_dir):
        s = create_session(data_dir, {"a0": 1.2e-10})
        assert "SPARC_Lelli2016c.csv" in s.input_checksums
        assert len(s.input_checksums["SPARC_Lelli2016c.csv"]) == 64

    def test_checksum_is_stable(self, data_dir):
        s1 = create_session(data_dir, {"a0": 1.2e-10})
        s2 = create_session(data_dir, {"a0": 1.2e-10})
        assert s1.input_checksums == s2.input_checksums

    def test_no_galaxy_table_gives_empty_checksums(self, tmp_path):
        s = create_session(tmp_path, {"a0": 1.2e-10})
        assert s.input_checksums == {}

    def test_each_run_has_unique_session_id(self, data_dir):
        s1 = create_session(data_dir, {"a0": 1.2e-10})
        s2 = create_session(data_dir, {"a0": 1.2e-10})
        assert s1.session_id != s2.session_id

    def test_git_commit_str_or_none(self, data_dir):
        s = create_session(data_dir, {"a0": 1.2e-10})
        assert s.git_commit is None or isinstance(s.git_commit, str)


# ---------------------------------------------------------------------------
# Integration: run_pipeline writes session.json
# ---------------------------------------------------------------------------

class TestRunPipelineSession:
    @pytest.fixture()
    def sparc_dir(self, tmp_path):
        """Tiny 3-galaxy synthetic SPARC dataset."""
        galaxy_csv = tmp_path / "SPARC_Lelli2016c.csv"
        names = ["A001", "A002", "A003"]
        pd.DataFrame({
            "Galaxy": names,
            "D": [10.0, 20.0, 30.0],
            "Inc": [45.0, 60.0, 75.0],
            "L36": [1e9, 2e9, 3e9],
            "Vflat": [150.0, 200.0, 250.0],
            "e_Vflat": [5.0, 5.0, 5.0],
        }).to_csv(galaxy_csv, index=False)

        rng = np.random.default_rng(0)
        for name, vf in zip(names, [150.0, 200.0, 250.0]):
            r = np.linspace(0.5, 15.0, 20)
            rc = pd.DataFrame({
                "r": r,
                "v_obs": np.full(20, vf) + rng.normal(0, 3, 20),
                "v_obs_err": np.full(20, 5.0),
                "v_gas": 0.3 * vf * np.ones(20),
                "v_disk": 0.75 * vf * np.ones(20),
                "v_bul": np.zeros(20),
                "SBdisk": np.zeros(20),
                "SBbul": np.zeros(20),
            })
            rc.to_csv(tmp_path / f"{name}_rotmod.dat", sep=" ", index=False, header=False)
        return tmp_path

    def test_session_json_created(self, sparc_dir, tmp_path):
        from src.scm_analysis import run_pipeline
        out = tmp_path / "results"
        run_pipeline(sparc_dir, out, verbose=False)
        assert (out / "session.json").exists()

    def test_session_json_is_valid(self, sparc_dir, tmp_path):
        from src.scm_analysis import run_pipeline
        out = tmp_path / "results"
        run_pipeline(sparc_dir, out, verbose=False)
        data = json.loads((out / "session.json").read_text(encoding="utf-8"))
        for key in ("session_id", "timestamp", "git_commit",
                    "parameters", "input_checksums"):
            assert key in data

    def test_session_a0_matches(self, sparc_dir, tmp_path):
        from src.scm_analysis import run_pipeline
        out = tmp_path / "results"
        run_pipeline(sparc_dir, out, a0=0.7e-10, verbose=False)
        data = json.loads((out / "session.json").read_text(encoding="utf-8"))
        assert data["parameters"]["a0"] == pytest.approx(0.7e-10)

    def test_session_loadable(self, sparc_dir, tmp_path):
        from src.scm_analysis import run_pipeline
        out = tmp_path / "results"
        run_pipeline(sparc_dir, out, verbose=False)
        s = Session.load(out / "session.json")
        assert isinstance(s, Session)
        assert uuid.UUID(s.session_id).version == 4
