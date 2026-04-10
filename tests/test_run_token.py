"""
tests/test_run_token.py — Unit tests for scripts/run_token.py.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure the scripts directory is importable.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_token  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expected_token_id(git_hash: str, timestamp: str) -> str:
    raw = f"{git_hash}:{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# _git_hash
# ---------------------------------------------------------------------------

class TestGitHash:
    def test_returns_string(self):
        result = run_token._git_hash()
        assert isinstance(result, str)

    def test_fallback_on_error(self):
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            result = run_token._git_hash()
        assert result == "unknown"

    def test_fallback_on_subprocess_error(self):
        import subprocess
        with mock.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ):
            result = run_token._git_hash()
        assert result == "unknown"

    def test_strips_newline(self):
        fake = mock.MagicMock()
        fake.stdout = "abc123\n"
        with mock.patch("subprocess.run", return_value=fake):
            result = run_token._git_hash()
        assert result == "abc123"


# ---------------------------------------------------------------------------
# _sha256_file
# ---------------------------------------------------------------------------

class TestSha256File:
    def test_correct_digest(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert run_token._sha256_file(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert run_token._sha256_file(f) == expected

    def test_large_file(self, tmp_path):
        data = b"x" * 200_000
        f = tmp_path / "large.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert run_token._sha256_file(f) == expected


# ---------------------------------------------------------------------------
# _make_token_id
# ---------------------------------------------------------------------------

class TestMakeTokenId:
    def test_length_16(self):
        tid = run_token._make_token_id("abc", "2024-01-01T00:00:00Z")
        assert len(tid) == 16

    def test_deterministic(self):
        tid1 = run_token._make_token_id("abc", "2024-01-01T00:00:00Z")
        tid2 = run_token._make_token_id("abc", "2024-01-01T00:00:00Z")
        assert tid1 == tid2

    def test_differs_on_different_hash(self):
        tid1 = run_token._make_token_id("aaa", "2024-01-01T00:00:00Z")
        tid2 = run_token._make_token_id("bbb", "2024-01-01T00:00:00Z")
        assert tid1 != tid2

    def test_differs_on_different_timestamp(self):
        tid1 = run_token._make_token_id("abc", "2024-01-01T00:00:00Z")
        tid2 = run_token._make_token_id("abc", "2024-01-02T00:00:00Z")
        assert tid1 != tid2

    def test_hex_chars_only(self):
        tid = run_token._make_token_id("x" * 40, "2025-06-01T12:00:00Z")
        assert all(c in "0123456789abcdef" for c in tid)

    def test_matches_expected(self):
        git_hash = "deadbeef"
        timestamp = "2024-03-15T10:30:00Z"
        expected = _expected_token_id(git_hash, timestamp)
        assert run_token._make_token_id(git_hash, timestamp) == expected


# ---------------------------------------------------------------------------
# create_token
# ---------------------------------------------------------------------------

class TestCreateToken:
    def test_required_keys(self):
        token = run_token.create_token()
        assert "token_id" in token
        assert "git_hash" in token
        assert "timestamp" in token

    def test_token_id_length(self):
        token = run_token.create_token()
        assert len(token["token_id"]) == 16

    def test_no_checksums_when_no_inputs(self):
        token = run_token.create_token()
        assert "checksums" not in token

    def test_checksums_present_when_inputs_given(self, tmp_path):
        f = tmp_path / "a.csv"
        f.write_bytes(b"col\n1\n2\n")
        token = run_token.create_token(inputs=[str(f)])
        assert "checksums" in token

    def test_checksum_value_correct(self, tmp_path):
        data = b"col,val\n1,2\n"
        f = tmp_path / "data.csv"
        f.write_bytes(data)
        token = run_token.create_token(inputs=[str(f)])
        expected = hashlib.sha256(data).hexdigest()
        assert token["checksums"][str(f)] == expected

    def test_missing_file_marked_not_found(self, tmp_path):
        missing = str(tmp_path / "no_such_file.csv")
        token = run_token.create_token(inputs=[missing])
        assert token["checksums"][missing] == "not_found"

    def test_multiple_inputs(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_bytes(b"a")
        f2.write_bytes(b"b")
        token = run_token.create_token(inputs=[str(f1), str(f2)])
        assert str(f1) in token["checksums"]
        assert str(f2) in token["checksums"]

    def test_timestamp_format(self):
        token = run_token.create_token()
        ts = token["timestamp"]
        # ISO-8601 UTC: YYYY-MM-DDTHH:MM:SSZ
        from datetime import datetime, timezone
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        assert dt is not None

    def test_token_id_deterministic_for_same_inputs(self):
        fixed_hash = "a1b2c3d4"
        fixed_ts = "2024-06-01T08:00:00Z"
        expected_id = _expected_token_id(fixed_hash, fixed_ts)
        with mock.patch.object(run_token, "_git_hash", return_value=fixed_hash), \
             mock.patch.object(run_token, "_make_token_id", wraps=run_token._make_token_id) as mk:
            # Verify that the token_id is the SHA-256 of hash:timestamp
            token = run_token.create_token()
            actual_id = _expected_token_id(token["git_hash"], token["timestamp"])
            assert token["token_id"] == actual_id

    def test_git_hash_unknown_fallback(self):
        with mock.patch.object(run_token, "_git_hash", return_value="unknown"):
            token = run_token.create_token()
        assert token["git_hash"] == "unknown"
        assert len(token["token_id"]) == 16


# ---------------------------------------------------------------------------
# save_token / load_token
# ---------------------------------------------------------------------------

class TestSaveLoadToken:
    def test_save_creates_file(self, tmp_path):
        token = run_token.create_token()
        out = tmp_path / "token.json"
        run_token.save_token(token, out)
        assert out.exists()

    def test_save_returns_path(self, tmp_path):
        token = run_token.create_token()
        out = tmp_path / "sub" / "token.json"
        result = run_token.save_token(token, out)
        assert isinstance(result, Path)
        assert result.exists()

    def test_save_creates_parent_dirs(self, tmp_path):
        token = run_token.create_token()
        out = tmp_path / "a" / "b" / "c" / "token.json"
        run_token.save_token(token, out)
        assert out.exists()

    def test_load_returns_dict(self, tmp_path):
        token = run_token.create_token()
        out = tmp_path / "token.json"
        run_token.save_token(token, out)
        loaded = run_token.load_token(out)
        assert isinstance(loaded, dict)

    def test_roundtrip(self, tmp_path):
        token = run_token.create_token()
        out = tmp_path / "token.json"
        run_token.save_token(token, out)
        loaded = run_token.load_token(out)
        assert loaded == token

    def test_saved_json_is_valid(self, tmp_path):
        token = run_token.create_token()
        out = tmp_path / "token.json"
        run_token.save_token(token, out)
        text = out.read_text(encoding="utf-8")
        parsed = json.loads(text)
        assert parsed["token_id"] == token["token_id"]

    def test_roundtrip_with_checksums(self, tmp_path):
        f = tmp_path / "input.csv"
        f.write_bytes(b"x,y\n1,2\n")
        token = run_token.create_token(inputs=[str(f)])
        out = tmp_path / "token.json"
        run_token.save_token(token, out)
        loaded = run_token.load_token(out)
        assert loaded["checksums"] == token["checksums"]


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------

class TestMain:
    def test_returns_dict(self, tmp_path):
        out = str(tmp_path / "tok.json")
        result = run_token.main(["--out", out])
        assert isinstance(result, dict)

    def test_file_created(self, tmp_path):
        out = tmp_path / "tok.json"
        run_token.main(["--out", str(out)])
        assert out.exists()

    def test_default_out_filename(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_token.main([])
        assert (tmp_path / "run_token.json").exists()

    def test_with_inputs(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"a,b\n1,2\n")
        out = str(tmp_path / "tok.json")
        result = run_token.main(["--out", out, "--inputs", str(f)])
        assert "checksums" in result
        assert str(f) in result["checksums"]

    def test_token_id_in_output(self, tmp_path):
        out = str(tmp_path / "tok.json")
        result = run_token.main(["--out", out])
        assert "token_id" in result
        assert len(result["token_id"]) == 16

    def test_output_matches_saved(self, tmp_path):
        out = tmp_path / "tok.json"
        result = run_token.main(["--out", str(out)])
        saved = json.loads(out.read_text(encoding="utf-8"))
        assert saved["token_id"] == result["token_id"]
