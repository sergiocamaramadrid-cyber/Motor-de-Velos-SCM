# Motor de Velos SCM

## Overview

This repository contains the SCM (Motor de Velos) framework and its experimental extensions for structural analysis of relationships in data.

---

## SCM-RAA v2.6-experimental

An experimental structural classification layer designed to evaluate relationships between variables.

This module extends the SCM framework but is developed and validated independently.

Located in:

- `notebooks/experimental/`
- `data/processed/`
- `docs/scm_raa/`

---

## What SCM-RAA does

SCM-RAA classifies relationships into three levels:

- **Foreground** → robust structural signal
- **Midground** → weak or diffuse structure
- **Background** → noise / no detectable structure

---

## Key features

- CRTT (piecewise vs linear model comparison using AIC)
- Regime Signature (quantitative vector)
- Bootstrap stability analysis
- Decision layer with explicit false-positive control

---

## Validation

Validated using:

- Synthetic noise (no false positives)
- Financial data (strong structural signal)
- Astrophysical datasets (weak/modulated signals)
- Control datasets (correct classification as noise)

---

## SCM v2.7 — Structural Signal Classification

The framework now classifies not only the presence of signal, but its structure:

| Class | Description |
|---|---|
| `global_critical` | Criticality is uniform across the full state space |
| `regime_dependent` | Criticality is fragmented across mass/energy regimes |
| `transition` | System is near a structural boundary |
| `non_linear` | Signal exists but lacks a linear critical structure |
| `none` | No detectable structure |

**Key insight:** Signal is not uniformly distributed in state space. Some systems concentrate criticality, while others fragment it across regimes.

Validated across: SPARC (regime_dependent), YANG (regime_dependent), MOJAVE (global_critical), SP500 (transition), ECONOMY (non_linear).

> v2.7 is an evolution of v2.6, not a replacement. The v2.6 decision layer (`foreground_confirmed`, `midground_candidate`, `background_confirmed`) remains fully valid.

---

## Important note

This module is **experimental** and intended for:

- validation
- audit
- controlled expansion

It is not a final predictive model.

---

## Repository structure

- `notebooks/experimental/` → reproducible pipelines
- `data/processed/` → validated outputs
- `docs/scm_raa/` → technical documentation
- `scripts/` → analysis scripts
- `tests/` → unit and integration tests

---

## Installation

### Requirements

- Python 3.10 or later
- Dependencies: see `requirements.txt`

### Setup

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Citation

See `CITATION.cff` and the Zenodo archive (DOI: see release page).

---

## License

Refer to the LICENSE file.

---

## Contact

Author: Sergio Cámara Madrid  
Repository: https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM