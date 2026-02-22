"""SCM model definitions and core computation helpers."""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SCMConfig:
    """Configuration for a single SCM analysis run."""

    data_dir: pathlib.Path
    out_dir: pathlib.Path
    variables: List[str] = field(default_factory=list)
    alpha: float = 0.05


@dataclass
class SCMResult:
    """Container for the output of one SCM model fit."""

    model_name: str
    score: float
    terms: Dict[str, float] = field(default_factory=dict)
    notes: Optional[str] = None

    def top_terms(self, n: int = 10) -> List[str]:
        """Return the *n* highest-scoring term names."""
        return sorted(self.terms, key=lambda k: self.terms[k], reverse=True)[:n]


def compute_universal_score(result: SCMResult) -> float:
    """Compute a normalized universal comparability score from *result*.

    Returns 0.0 when *result* contains no terms.
    """
    if not result.terms:
        return 0.0
    return sum(result.terms.values()) / len(result.terms)
