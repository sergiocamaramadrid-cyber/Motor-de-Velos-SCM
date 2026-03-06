#!/usr/bin/env python3
"""
Backward-compatible alias for SPARC v1 consolidation.

Delegates to ``scripts/process_sparc.py`` so existing commands like
``python scripts/consolidar_sparc_v1.py`` continue to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from scripts.process_sparc import main
except ImportError:
    if __name__ == "__main__":
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from scripts.process_sparc import main
    else:
        raise


if __name__ == "__main__":
    raise SystemExit(main())
