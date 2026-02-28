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

Raw datasets (e.g., SPARC, LITTLE THINGS) are **not versioned**.  
Generated results are **not versioned**.  
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

### Deep-Regime Slope Diagnostic

```bash
python scripts/deep_slope_test.py \
  --csv results/universal_term_comparison_full.csv \
  --g0 1.2e-10 \
  --deep-threshold 0.3 \
  --out results/diagnostics/deep_slope_test
```

### Hinge-Friction vs SFR Test

Tests whether hinge-derived "friction" proxies (F1/F2/F3) predict SFR at fixed
baryonic mass, with HC3-robust OLS, a permutation test, and a mass-matched
Wilcoxon test.

**Required input files** (see `data/hinge_sfr/README.md` for the full column
contract):

`profiles.csv` — one row per radial point:

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `galaxy` | str | — | Galaxy identifier |
| `r_kpc` | float | **kpc** | Galactocentric radius |
| `vbar_kms` | float | **km/s** | Baryonic rotation velocity (or `gbar_m_s2`) |
| `rmax_kpc` | float | kpc | Outermost radius (optional) |

`galaxy_table.csv` — one row per galaxy:

| Column | Type | Units | Description |
|--------|------|-------|-------------|
| `galaxy` | str | — | Galaxy identifier |
| `log_mbar` | float | log10(M_sun) | log baryonic mass |
| `log_sfr` | float | log10(M_sun/yr) | log SFR (or `sfr` in M_sun/yr) |
| `morph_bin` | str | — | Morphology bin (optional) |

Sample data with 30 representative galaxies is provided in `data/hinge_sfr/`.

```bash
python scripts/hinge_sfr_test.py \
  --profiles data/hinge_sfr/profiles.csv \
  --galaxy-table data/hinge_sfr/galaxy_table.csv \
  --out results/hinge_sfr \
  --log-g0 -9.921 \
  --d 1.0
```

Outputs are written to `results/hinge_sfr/`: `hinge_features.csv`,
`regression_F*.txt`, `permutation_F*.txt`, `matched_pairs_F*.txt`.

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