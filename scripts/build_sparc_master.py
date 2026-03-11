#!/usr/bin/env python3
"""
Backward-compatible alias for SPARC master catalog generation.

Delegates to ``scripts/build_sparc_full_catalog.py`` so commands like
``python scripts/build_sparc_master.py`` continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from scripts.build_sparc_full_catalog import main
except ImportError:
    if __name__ == "__main__":
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from scripts.build_sparc_full_catalog import main
    else:
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guard
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
