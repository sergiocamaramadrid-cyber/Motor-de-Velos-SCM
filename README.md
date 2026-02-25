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
- scripts/: CLI-style scripts for validation and diagnostics (e.g. scripts/compare_nu_models.py, scripts/deep_slope_test.py).
- data/: Data ingestion instructions and small fixtures; large raw datasets are not included (see docs/ for data contracts).
- results/: Generated outputs (not versioned). Follow naming convention: results/<module>/<artifact>-v<semver>.csv
- docs/: Formal documentation, data contracts and validation protocols (machine- and reviewer-oriented).
- notebooks/: Exploratory and validation notebooks (non-deterministic; for inspection and figure generation).
- paper/: Manuscript figures, supplementary materials and submission assets.
- tests/ (if present): Unit and integration tests for code and pipelines.
- Top-level metadata: CITATION.md, LICENSE, requirements.txt.

---

## Installation

Choose **one** of the two options below. Do **not** run both.

### Option A — venv + pip (recommended)

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — conda

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM

conda create -n scm python=3.10
conda activate scm

pip install -r requirements.txt
```

### Running the tests

```bash
pytest
```

---

## Data Policy

Raw datasets (e.g., SPARC, LITTLE THINGS) are **not versioned**.  
Generated results are **not versioned**.  
Download and preprocessing scripts are provided for reproducibility.  
See `docs/SPARC_EXPECTED_BEHAVIOUR.md` for formal data contract.

---

## Running the Framework

### Full SPARC pipeline

```bash
python -m src.scm_analysis \
  --data-dir data/SPARC \
  --out results/ \
  --a0 1.2e-10
```

Produces `results/per_galaxy_summary.csv`, `results/audit/sparc_global.csv`,
and `results/diagnostics/` (VIF, condition number, partial correlations).

### ν-model comparison

```bash
python scripts/compare_nu_models.py \
  --data-dir data/SPARC \
  --out results/diagnostics/compare_nu_models \
  --a0 1.2e-10 \
  --deep-threshold 0.3
```

### Deep-regime slope diagnostic

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

Full methodological scope (validated claims, out-of-scope claims, falsifiable
predictions): `docs/METHODOLOGY.md`

Technical data contract: `docs/SPARC_EXPECTED_BEHAVIOUR.md`

---

## Reproducibility

To reproduce a specific run (e.g. one cited in a manuscript or report):

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
git checkout <commit-sha>

# Option A
source .venv/bin/activate
# Option B
# conda activate scm

python -m src.scm_analysis \
  --data-dir data/SPARC \
  --out results/ \
  --a0 1.2e-10
```

Each run should record:

- Git commit hash (`git rev-parse HEAD`)
- Input file checksums (`sha256sum data/SPARC/*.csv`)
- Exact command-line arguments and parameter values

Outputs follow the naming convention:
```
results/<module>/<artifact>-v<semver>.csv
```

---

## Scientific Scope and Limitations

**What this framework is:** a reproducible, auditable pipeline that evaluates a
phenomenological rotation-curve model (SCM — Motor de Velos) against the SPARC
catalogue of 175 disc galaxies.

**What this framework is not:**

- It is **not** a cosmological model. No CMB, BAO, large-scale structure, or
  galaxy-cluster predictions are made.
- It does **not** claim to replace or falsify ΛCDM, dark-matter halos, or MOND.
- No result in this repository constitutes a definitive physical explanation
  until external validation and peer review have been completed.

**Current validation status (v0.6.0):**

- ✅ In-sample goodness-of-fit (χ²) and AICc model comparison
- ✅ Fixed radial-split out-of-sample protocol (no post-hoc tuning)
- ✅ Structural permutation test and sensitivity analysis
- ✅ Deep-regime slope diagnostic (β = 0.5 test)
- ✅ Multicollinearity diagnostics (VIF, condition number, partial correlation)
- ✅ 175-galaxy reproducible audit table

**What is needed before stronger claims:**

1. External dataset validation (LITTLE THINGS, galaxy clusters)
2. Formal head-to-head comparison against NFW halos and full MOND implementation
3. Independent replication and peer-reviewed publication

For the full methodological scope declaration, including falsifiable predictions
and the explicit list of out-of-scope claims, see:
`docs/METHODOLOGY.md`

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