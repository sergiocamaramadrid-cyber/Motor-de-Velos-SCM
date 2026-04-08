# Motor-de-Velos-SCM

## Historical Context / Contexto histórico

Author: Sergio Cámara Madrid  
Consolidation date: 2026-02-12

This repository preserves the conceptual origins of the SCM — Motor de Velos (Fluid Condensation Model). The historical note is maintained for provenance and attribution; all scientific claims and evaluations are supported by reproducible analyses, documented statistical protocols, and versioned code.

For the full historical and conceptual background, see:
`docs/HISTORICAL_NOTE_MOTOR_DE_VELOS.md`

The remainder of this README focuses on the reproducible computational framework and instructions to run the evaluation pipelines.

---

## Overview

Motor-de-Velos-SCM provides a reproducible, auditable pipeline to evaluate galaxy rotation curves under the SCM (Motor de Velos; Fluid Condensation) model. The repository implements end-to-end workflows from raw data preprocessing to model comparison and diagnostic reporting.

Core capabilities
- Deterministic data processing pipelines with explicit preprocessing steps.
- Fixed, pre‑specified out‑of‑sample (OOS) validation using radial splits (no post‑hoc tuning).
- Model comparison using the corrected Akaike Information Criterion (AICc).
- Diagnostic tests for deep‑regime slope behaviour and other targeted hypotheses.
- Versioned, machine‑readable outputs and logging to support audit and replication.

Design goals
- Reproducible: reproducible runs should record input checksums and git commit hashes when generating results.
- Deterministic: deterministic preprocessing and evaluation steps.
- Audit-friendly: clear inputs/outputs and diagnostics.
- Version-controlled: code and analysis scripts tracked in the repository.

---

## Repository structure

The repository is organized as follows:

- src/: Core model implementations and analysis modules (Python package layout).
- scripts/: CLI-style scripts for preprocessing, validation and diagnostics (e.g. scripts/process_sparc.py, scripts/deep_slope_test.py).
- data/: Data ingestion instructions and small fixtures; large raw datasets are not included (see docs/ for data contracts).
- results/: Generated outputs (not versioned). Follow naming convention: results/<module>/<artifact>-v<semver>.csv
- docs/: Formal documentation, data contracts and validation protocols (machine- and reviewer-oriented).
- notebooks/: Exploratory and validation notebooks (non-deterministic; for inspection and figure generation).
- paper/: Manuscript figures, supplementary materials and submission assets.
- tests/ (if present): Unit and integration tests for code and pipelines.
- Top-level metadata: CITATION.md, LICENSE, requirements.txt, environment.yml.

---

## Installation

### Requirements

- Python 3.10 or later.
- System tools: git.
- Dependencies: see `requirements.txt`.
- Optional: Conda environment via `environment.yml` for reproducible environments.

### Setup (recommended)

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Optional (Conda)

```bash
conda env create -f environment.yml
conda activate motor-de-velos
pip install -r requirements.txt    # if additional deps are needed
```

### Developer / tests (if present)

- Run unit tests: `pytest`  
- Linting/format: `pre-commit run --all-files` (if pre-commit is configured)

Notes:
- If the repository provides an installable package (setup.py / pyproject.toml), prefer `pip install -e .` for development.
- Reproducible runs should record input checksums and git commit hashes when generating results; ensure you install dependencies in a clean environment to reproduce analyses.

---

## Data Policy

Raw datasets (e.g., full SPARC catalogue, LITTLE THINGS) are **not versioned**.  
Generated results are **not versioned**.  
Small curated fixtures (e.g., `data/sparc_subset.csv`, N = 79) **are versioned**.  
Download and preprocessing scripts are provided for reproducibility.  
See `docs/SPARC_EXPECTED_BEHAVIOUR.md` for formal data contract.

---

## Running the Framework

### SPARC Validation (Example)

```bash
python scripts/process_sparc.py \
  --input data/SPARC/sparc_raw.csv \
  --out results/SPARC/rotation_curves-v1.0.csv
```

### SPARC Subset — Split-by-Mass Environmental Analysis

A curated subset of 79 SPARC galaxies (`data/sparc_subset.csv`) is committed to
the repository for fully reproducible environmental analyses.  Duplicates have
been removed; columns: `galaxy`, `logM`, `delta_mass_std`, `slope_tail`.

Generate the two-panel δF₃ vs δ_mass,std figure split at the sample median of logM:

```bash
python scripts/plot_sparc_split_mass.py
# or with explicit paths:
python scripts/plot_sparc_split_mass.py \
  --csv data/sparc_subset.csv \
  --out results/SPARC_split_mass_environment.png
```

Key results (N = 79, median logM ≈ 10.64):

| Subsample | N  | ρ (Spearman) | p-value |
|---|---|---|---|
| Low mass (logM < 10.64)  | 39 | −0.15 | 0.36 (n.s.) |
| High mass (logM ≥ 10.64) | 40 | **−0.49** | **0.001** |

The high-mass regime shows a significant negative environmental dependence; the
low-mass regime does not.  Both results are protected by regression tests in
`tests/test_plot_sparc_split_mass.py` (56 tests).

### SPARC Subset — Mass Threshold Scan

For a data-driven search of the optimal logM cut, use the threshold scan script.
It sweeps logM cuts from 10.0 to 11.3 (step 0.05), computes Spearman ρ for
galaxies with `logM ≥ m_cut`, and identifies the cut that maximises the
composite signal score `|ρ| × √N × (−log₁₀ p)`.

```bash
python scripts/plot_sparc_mass_scan.py
# or with explicit paths and custom range:
python scripts/plot_sparc_mass_scan.py \
  --csv data/sparc_subset.csv \
  --out results/sparc_mass_scan.png \
  --m-start 10.0 --m-stop 11.3 --m-step 0.05 --n-min 15
```

Key result on the committed 79-galaxy catalog:

| logM cut | N  | ρ (Spearman) | p-value | Score |
|---|---|---|---|---|
| **10.05** (best) | 56 | **−0.48** | **1.8 × 10⁻⁴** | 13.46 |

All 21 evaluated cuts show ρ < 0 and p < 0.05.  Results are protected by
regression tests in `tests/test_plot_sparc_mass_scan.py` (51 tests).

---

### Deep-Regime Slope Diagnostic

```bash
python scripts/deep_slope_test.py \
  --csv results/universal_term_comparison_full.csv \
  --g0 1.2e-10 \
  --deep-threshold 0.3 \
  --out results/diagnostics/deep_slope_test
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

Reproducible runs should record input checksums and git commit hashes when generating results.

Each run should record:

- Git commit hash  
- Input file checksums  
- Command-line arguments  
- Parameter values (e.g., g0, thresholds)

Outputs should be written under:
```
results/<module>/<artifact>-v<semver>.csv
```

---

## Limitations

The framework evaluates rotation-curve behavior; it does not claim cosmological completeness.  
Statistical validation is dataset-dependent.  
Interpretation remains separate from computational reproducibility.

---

## Citation

See:

- `CITATION.md`  
- Zenodo archive (DOI when available)

---

## License

Refer to the LICENSE file.

---

## Contact

Author: Sergio Cámara Madrid  
Repository: https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM

EOF