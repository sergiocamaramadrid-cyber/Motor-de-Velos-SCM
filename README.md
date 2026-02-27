<p align="center">
  <img src="assets/logo.png" alt="SCM Framework – Motor de Velos" width="800">
</p>

<h1 align="center">SCM Framework – Motor de Velos</h1>

<p align="center">
  <b>Statistical Condensation Model (SCM)</b><br>
  Dynamic inference framework for galactic rotation and pressure-regime classification
</p>

<p align="center">
  <img src="https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/version-v0.6.2-blue.svg" alt="version">
  <img src="https://img.shields.io/badge/status-validated-success.svg" alt="status">
  <img src="https://img.shields.io/badge/python-3.10+-brightgreen.svg" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="license">
</p>

---

## Overview

The SCM Framework is a reproducible scientific pipeline for modeling galactic rotation curves and detecting pressure regimes through statistical and dynamical inference.

Motor-de-Velos-SCM provides a reproducible, auditable pipeline to evaluate galaxy rotation curves under the SCM (Motor de Velos; Fluid Condensation) model. The repository implements end-to-end workflows from raw data preprocessing to model comparison and diagnostic reporting.

The system provides:

- ξ pressure calibration
- Regime classification
- Rotation curve fitting
- Pressure injector detection
- Full audit trail and reproducibility

Core capabilities:
- Deterministic data processing pipelines with explicit preprocessing steps.
- Fixed, pre‑specified out‑of‑sample (OOS) validation using radial splits (no post‑hoc tuning).
- Model comparison using the corrected Akaike Information Criterion (AICc).
- Diagnostic tests for deep‑regime slope behaviour and other targeted hypotheses.
- Versioned, machine‑readable outputs and logging to support audit and replication.

Validated on:

- SPARC catalog
- Local Group galaxies
- M81 Group galaxies

---

## Key Result

Global calibration: **ξ = 1.37 ± 0.02**

Observed range: **1.28 ≤ ξ ≤ 1.48**

---

## What this repo provides

- Reproducible SCM pipeline (`python -m src.scm_analysis`)
- Audit artifacts (VIF / condition number / quality report)
- Optional pressure-injector detection (`--detect-pressure-injectors`)
- Validation reports under `audits/validated/` and consolidated summaries under `reports/`

---

## Quickstart

```bash
python -m pip install -r requirements.txt
pytest -q
python -m src.scm_analysis --help
```

---

## Installation

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
pip install -r requirements.txt
```

### Full setup (recommended)

```bash
# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Developer / tests

```bash
pytest
```

---

## Example Usage

```bash
python -m src.scm_analysis \
  --target-galaxy M82 \
  --custom-data ./data/m82_rotcurve.txt \
  --detect-pressure-injectors \
  --audit-mode high-pressure
```

### SPARC Validation

```bash
python scripts/process_sparc.py \
  --input data/SPARC/sparc_raw.csv \
  --out results/SPARC/rotation_curves-v1.0.csv
```

### Deep-Regime Slope Diagnostic

```bash
python scripts/deep_slope_test.py \
  --csv results/universal_term_comparison_full.csv \
  --g0 1.2e-10 \
  --deep-threshold 0.3 \
  --out results/diagnostics/deep_slope_test
```

---

## Repository Structure

```
Motor-de-Velos-SCM/
│
├── assets/      → visual identity
├── src/         → core engine
├── scripts/     → CLI analysis scripts
├── tests/       → validation
├── audits/      → scientific audit trail
├── reports/     → analysis reports
├── results/     → generated outputs (not versioned)
├── data/        → data fixtures and ingestion instructions
└── docs/        → documentation and data contracts
```

---

## Statistical Protocol

The evaluation framework follows fixed rules:

- Radial split OOS (no post-hoc tuning)
- AICc-based model comparison
- Deterministic merge contracts
- Explicit deep-regime slope test
- Versioned output naming

Details: `docs/SPARC_EXPECTED_BEHAVIOUR.md`

---

## Reproducibility

Each run records:

- Git commit hash
- Input file checksums
- Command-line arguments
- Parameter values (e.g., g0, thresholds)

Outputs follow the naming convention:
```
results/<module>/<artifact>-v<semver>.csv
```

---

## Scientific Validation Status

| Field | Value |
|---|---|
| Status | VALIDATED |
| Framework version | v0.6.2 |
| Reproducible | YES |

---

## Data Policy

Raw datasets (e.g., SPARC, LITTLE THINGS) are **not versioned**.  
Generated results are **not versioned**.  
Download and preprocessing scripts are provided for reproducibility.  
See `docs/SPARC_EXPECTED_BEHAVIOUR.md` for formal data contract.

---

## Historical Context

Author: Sergio Cámara Madrid  
Consolidation date: 2026-02-12

This repository preserves the conceptual origins of the SCM — Motor de Velos (Fluid Condensation Model). The historical note is maintained for provenance and attribution; all scientific claims and evaluations are supported by reproducible analyses, documented statistical protocols, and versioned code.

For the full historical and conceptual background, see: `docs/HISTORICAL_NOTE_MOTOR_DE_VELOS.md`

---

## Limitations

The framework evaluates rotation-curve behavior; it does not claim cosmological completeness.  
Statistical validation is dataset-dependent.  
Interpretation remains separate from computational reproducibility.

---

## Citation

If you use this work, please cite the repository using the metadata in [`CITATION.cff`](CITATION.cff) (GitHub displays a **"Cite this repository"** button automatically).

---

## License

MIT License — see the LICENSE file for details.

---

## Contact

Author: Sergio Cámara Madrid  
Repository: https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM