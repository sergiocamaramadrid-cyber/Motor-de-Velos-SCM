"""
session.py — Session tracking for the Motor de Velos SCM pipeline.

Each pipeline run creates a :class:`Session` that records:

- A unique session ID (UUID4)
- ISO-8601 timestamp (UTC)
- Git commit hash (if available)
- Input-file checksums (SHA-256)
- Run parameters (e.g. ``a0``)

Sessions are serialised to ``session.json`` inside the pipeline output
directory so that every result set is fully traceable.

Example
-------
::

    from src.session import create_session

    session = create_session(data_dir=Path("data/SPARC"), parameters={"a0": 1.2e-10})
    session.save(out_dir / "session.json")
"""

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _git_commit() -> Optional[str]:
    """Return the current HEAD commit hash, or *None* if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class Session:
    """Metadata record for a single pipeline run.

    Attributes
    ----------
    session_id : str
        Unique UUID4 identifier for this run.
    timestamp : str
        ISO-8601 UTC timestamp of when the session was created.
    git_commit : str or None
        HEAD commit hash at the time of the run, or *None* if unavailable.
    parameters : dict
        Run parameters (e.g. ``{'a0': 1.2e-10}``).
    input_checksums : dict
        Mapping of file name → SHA-256 hex digest for key input files.
    """

    session_id: str
    timestamp: str
    git_commit: Optional[str]
    parameters: Dict[str, Any]
    input_checksums: Dict[str, str]

    def to_dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON serialisation."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write the session metadata to *path* as indented JSON.

        Parameters
        ----------
        path : Path
            Destination file (created or overwritten).
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Session":
        """Load a :class:`Session` from a JSON file.

        Parameters
        ----------
        path : Path
            Path to a ``session.json`` file previously written by :meth:`save`.

        Returns
        -------
        Session
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def create_session(data_dir: Path, parameters: Dict[str, Any]) -> Session:
    """Create a new :class:`Session` for a pipeline run.

    Computes a SHA-256 checksum for the first SPARC galaxy-table file found
    in *data_dir* and captures the current git HEAD (if available).

    Parameters
    ----------
    data_dir : Path
        Root data directory.  The function searches for the galaxy-table file
        (``SPARC_Lelli2016c.csv`` / ``.mrt``) to checksum.
    parameters : dict
        Run parameters to record (e.g. ``{'a0': 1.2e-10}``).

    Returns
    -------
    Session
        Populated :class:`Session` ready to be saved.
    """
    data_dir = Path(data_dir)

    # Checksum the first galaxy-table file found (mirrors load_galaxy_table logic)
    checksums: Dict[str, str] = {}
    candidates = [
        data_dir / "SPARC_Lelli2016c.csv",
        data_dir / "SPARC_Lelli2016c.mrt",
        data_dir / "raw" / "SPARC_Lelli2016c.csv",
        data_dir / "processed" / "SPARC_Lelli2016c.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            checksums[candidate.name] = _sha256(candidate)
            break

    return Session(
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        git_commit=_git_commit(),
        parameters=dict(parameters),
        input_checksums=checksums,
    )
