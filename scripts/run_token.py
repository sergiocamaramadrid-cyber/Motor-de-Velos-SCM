"""
scripts/run_token.py — Run-provenance token generator for the SCM pipeline.

A *run token* is a compact, human-readable JSON record that uniquely identifies
one invocation of any SCM analysis pipeline.  It captures the information
required to reproduce or audit a run:

  - ``token_id``   — deterministic SHA-256 hex digest (first 16 chars) built
                     from the git commit hash and the UTC timestamp.
  - ``git_hash``   — full SHA-1 of the HEAD commit (or ``"unknown"`` outside
                     a git repository).
  - ``timestamp``  — ISO-8601 UTC timestamp of when the token was created.
  - ``checksums``  — mapping of file paths to their SHA-256 hex digests (only
                     present when input paths are supplied).

Public API
----------
create_token(inputs=None) -> dict
    Build and return a token dict.  *inputs* is an optional sequence of file
    paths whose SHA-256 checksums are recorded in the token.

save_token(token, path) -> Path
    Write *token* as pretty-printed JSON to *path* and return the resolved Path.

load_token(path) -> dict
    Read and return a token dict from a JSON file.

main(argv=None) -> dict
    CLI entry-point.  Writes a token to ``--out`` (default: run_token.json).

CLI usage
---------
::

    python scripts/run_token.py
    python scripts/run_token.py --out results/run_token.json
    python scripts/run_token.py --inputs data/sparc_basic.csv data/env_proxy.csv
    python scripts/run_token.py --out results/token.json --inputs data/sparc_basic.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git_hash() -> str:
    """Return the full SHA-1 of the current HEAD commit, or ``"unknown"``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_token_id(git_hash: str, timestamp: str) -> str:
    """Build a deterministic 16-character token ID from *git_hash* + *timestamp*."""
    raw = f"{git_hash}:{timestamp}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_token(inputs: Sequence[str | Path] | None = None) -> dict:
    """Create and return a run-provenance token.

    Parameters
    ----------
    inputs:
        Optional sequence of file paths.  A SHA-256 checksum is computed for
        each existing file and stored under ``token["checksums"]``.

    Returns
    -------
    dict with keys ``token_id``, ``git_hash``, ``timestamp``, and (if
    *inputs* was provided) ``checksums``.
    """
    git_hash = _git_hash()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    token_id = _make_token_id(git_hash, timestamp)

    token: dict = {
        "token_id": token_id,
        "git_hash": git_hash,
        "timestamp": timestamp,
    }

    if inputs:
        checksums: dict[str, str] = {}
        for p in inputs:
            path = Path(p)
            if path.is_file():
                checksums[str(path)] = _sha256_file(path)
            else:
                checksums[str(path)] = "not_found"
        token["checksums"] = checksums

    return token


def save_token(token: dict, path: str | Path) -> Path:
    """Write *token* as JSON to *path*.

    Parameters
    ----------
    token:
        Token dict as returned by :func:`create_token`.
    path:
        Destination file path.  Parent directories are created if necessary.

    Returns
    -------
    Resolved :class:`pathlib.Path` of the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(token, indent=2) + "\n", encoding="utf-8")
    return out.resolve()


def load_token(path: str | Path) -> dict:
    """Read and return a token dict from a JSON file at *path*."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    """CLI entry-point.  Returns the token dict."""
    parser = argparse.ArgumentParser(
        description="Generate a run-provenance token for an SCM pipeline run.",
    )
    parser.add_argument(
        "--out",
        default="run_token.json",
        help="Output path for the JSON token file (default: run_token.json).",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        metavar="FILE",
        help="Input files whose SHA-256 checksums are recorded in the token.",
    )
    args = parser.parse_args(argv)

    token = create_token(inputs=args.inputs)
    out_path = save_token(token, args.out)
    print(f"Token written to: {out_path}", file=sys.stderr)
    print(json.dumps(token, indent=2))
    return token


if __name__ == "__main__":
    main()
