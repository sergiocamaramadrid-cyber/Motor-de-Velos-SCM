# Motor de Velos SCM

## Overview

This repository contains the SCM (Motor de Velos) framework and its experimental extensions for structural analysis of relationships in data.

---

## SCM-RAA v2.6-experimental

An experimental structural classification layer designed to evaluate relationships between variables.

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