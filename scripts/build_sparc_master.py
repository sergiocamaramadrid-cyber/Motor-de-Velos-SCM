#!/usr/bin/env python3
"""Backward-compatible alias for SPARC 175 master catalog generation."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from scripts.build_sparc_175_master import main
except ImportError:
    if __name__ == "__main__":
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from scripts.build_sparc_175_master import main
    else:
        raise


if __name__ == "__main__":
    try:
        argv = ["--sparc-dir" if arg == "--data-root" else arg for arg in sys.argv[1:]]
        main(argv)
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
