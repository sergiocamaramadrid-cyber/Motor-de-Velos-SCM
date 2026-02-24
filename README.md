# Motor-de-Velos-SCM

## Historical Context / Contexto histórico

This project, authored by Sergio Cámara Madrid and consolidated on 12 February 2026,
is based on the *Fluid Condensation Model (Motor de Velos)*.

For the full historical and conceptual background, see:

➡️ `docs/HISTORICAL_NOTE_MOTOR_DE_VELOS.md`

The sections below describe the reproducible computational framework.

---

## Overview

Motor-de-Velos-SCM implements a reproducible statistical framework for evaluating
galaxy rotation curves under the SCM – Motor de Velos (Fluid Condensation) model.

The repository provides:

- Explicit data processing pipelines.
- Fixed out-of-sample (OOS) validation protocol.
- AICc-based model comparison.
- Diagnostic tests for deep-regime slope behavior.
- Transparent and reproducible outputs.

The framework is designed to be:

- Deterministic
- Reproducible
- Audit-friendly
- Version-controlled

---

## Repository Structure

```
src/            Core models and analysis code
scripts/        Execution scripts (validation, diagnostics)
data/           Data instructions (raw data not versioned)
results/        Generated outputs (not versioned)
docs/           Formal documentation and contracts
notebooks/      Exploratory and validation notebooks
paper/          Manuscript resources
```

---

## Installation

### Requirements

Python 3.10+  
Dependencies listed in `requirements.txt`  
Optional: Conda environment via `environment.yml`

### Setup

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

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