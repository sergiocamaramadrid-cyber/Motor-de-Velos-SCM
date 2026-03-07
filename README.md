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

## Quick data check

Before downloading SPARC or LITTLE THINGS again, check whether the data already
exists locally:

```bash
find . -iname "*sparc*"
find . -name "*rotmod.dat"
find . -name "*.csv"
find . -iname "*ddo*"
```

If ~175 `*_rotmod.dat` files are present, SPARC is already available and can be
reused. Otherwise, rebuild the catalog with:

```bash
python scripts/build_sparc_full_catalog.py
```

For private or temporary GitHub links (for example `/tasks/...`) that return
404, provide a screenshot, the visible text, or the correct public link.
See `data/README.md` for detailed guidance.

---

## Running the Framework

### SPARC Validation (Example)

```bash
python scripts/process_sparc.py \
  --input data/SPARC \
  --out results/SPARC/rotation_curves-v1.0.csv
```

### SCM framework on SPARC catalog

```bash
python -m src.scm_analysis \
  --data-dir data/SPARC \
  --out results/SPARC/scm_run
```

If you also maintain the merged SPARC metadata catalog:

```bash
python scripts/build_sparc_catalog.py
python -m src.scm_analysis \
  --data-dir data/SPARC \
  --out results/SPARC/scm_run
```

Main outputs under `results/SPARC/scm_run/`:

- `per_galaxy_summary.csv`
- `universal_term_comparison_full.csv`
- `deep_slope_test.csv`
- `sensitivity_a0.csv`
- `executive_summary.txt`
- `top10_universal.tex`
- `scm_summary.json`

Also supported (compatibility alias):

```bash
python scripts/consolidar_sparc_v1.py \
  --input data/SPARC \
  --out results/SPARC/rotation_curves-v1.0.csv
```

### SPARC coordinate enrichment (for anisotropy test)

```bash
python scripts/enrich_sparc_with_coordinates.py \
  --metadata data/SPARC/SPARC_Lelli2016c.csv \
  --input results/SPARC/rotation_curves-v1.0.csv \
  --output results/SPARC/rotation_curves-v1.1-coords.csv
```

### Deep-Regime Slope Diagnostic

```bash
python scripts/deep_slope_test.py \
  --csv results/universal_term_comparison_full.csv \
  --g0 1.2e-10 \
  --deep-threshold 0.3 \
  --out results/diagnostics/deep_slope_test
```

### BIG-SPARC veil test pipeline

```bash
# Option A: if your source table is contract-style
# (galaxy, r_kpc, vobs_kms, vbar_kms), normalize it first:
python scripts/prepare_big_sparc_catalog.py \
  --input data/big_sparc/contract/big_sparc_contract.parquet \
  --out data/big_sparc_catalog.csv

# Option B: build directly from SPARC *_rotmod.dat files
# (supports both full run and subset via --galaxies):
python scripts/prepare_big_sparc_catalog.py \
  --sparc-dir data/SPARC \
  --galaxies NGC2403,NGC3198,NGC6503,DDO154,UGC0128 \
  --out data/big_sparc_catalog.csv

# Option C: download/build full SPARC catalog in one command
# (writes data/SPARC/sparc_full.csv with galaxy,r_kpc,g_obs,g_bar,logMbar,logSigmaHI_out):
python scripts/build_sparc_full_catalog.py

# Run the β pipeline
python scripts/run_big_sparc_veil_test.py \
  --catalog data/SPARC/sparc_full.csv \
  --out results
```

### Recommended practical order for SPARC analysis

Use this sequence to "walk" the full SPARC sample with reproducible checks, from raw inputs to final statistical interpretation.

1. **Confirm SPARC inputs are complete enough**

   Expected structure:

   ```text
   data/SPARC/
   ├── SPARC_table2.mrt (or SPARC_Lelli2016c.mrt/.csv)
   ├── MassModels_Lelli2016c.mrt
   └── rotmod/
       └── *_rotmod.dat
   ```

   A near-complete run typically has ~175 `*_rotmod.dat` files.

   Quick file-count check:

   ```bash
   find data/SPARC/rotmod -name '*_rotmod.dat' | wc -l
   ```

2. **Build the homogeneous SPARC catalog**

   ```bash
   python scripts/build_sparc_full_catalog.py --data-root data/SPARC --out results/SPARC
   ```

   First file to inspect:

   - `results/SPARC/sparc_full_catalog.csv`

   Quick checks:

   - galaxy count (target: ~170–175),
   - anomalous NaN patterns,
   - galaxies with very few radial points.

3. **Compute the framework observable catalog (`F3`)**

   When working from a contract-compliant table, a practical command is:

   ```bash
   python scripts/generate_f3_catalog_from_contract.py \
     --input data/big_sparc/contract/big_sparc_contract.parquet \
     --out results/SPARC
   ```

   Then inspect:

   - `results/SPARC/f3_catalog.csv`

   The legacy quantity `deep_slope` is preserved for backward compatibility, while the centered observable `delta_f3 = deep_slope - 0.5` is adopted as the primary diagnostic quantity.

   Check that `F3_SCM` shows a broad/continuous distribution and does not collapse to a single value.

   If you prefer a single guarded command (generate + validate, with optional β analysis):

   ```bash
   ./run_f3_pipeline.sh --input data/big_sparc/contract/big_sparc_contract.parquet --out results/SPARC
   # add --full-analysis to run scripts/run_big_sparc_veil_test.py after validation
   ```

   Mini command to print `min/max/std` from the generated catalog:

   ```bash
   python -c "import pandas as pd; p='results/SPARC/f3_catalog.csv'; df=pd.read_csv(p); c='F3_SCM' if 'F3_SCM' in df.columns else 'deep_slope'; s=df[c].dropna(); print({'columna': c, 'min': float(s.min()), 'max': float(s.max()), 'std': float(s.std(ddof=1))})"
   ```

4. **Compute the deep-regime slope catalog (`beta`)**

   A practical full-catalog command is:

   ```bash
   python scripts/run_big_sparc_veil_test.py \
     --catalog results/SPARC/sparc_full_catalog.csv \
     --out results/SPARC
   ```

   Inspect:

   - `results/SPARC/beta_catalog.csv`

   Check that `beta` has real dispersion (not all values identical) and evaluate its relation with `F3_SCM`.

5. **Run out-of-sample (OOS) validation**

   Run the OOS step available in your branch/pipeline (for example `python scripts/scm_oos_validation.py` where available), then inspect:

   - `results/oos_validation/oos_generalization_results.csv`

   Compare models (`baseline` vs `SCM`) on:

   - `RMSE_out` (lower is better),
   - `MAE_out` (lower is better),
   - `delta_logL` (higher is better).

6. **Inspect ΔRMSE distribution**

   Open:

   - `hist_delta_rmse_out.pdf`

   with:

   - `ΔRMSE = RMSE_SCM - RMSE_baseline`

   A histogram shifted toward negative values indicates SCM improvement.

7. **Check Wilcoxon significance**

   Read `p-value` from OOS logs/CSV and interpret quickly:

   - `> 0.1`: inconclusive
   - `< 0.05`: evidence
   - `< 0.01`: strong evidence
   - `< 0.001`: very strong evidence

8. **Read the final executive summary**

   Inspect:

   - `results/executive_summary.txt`

   This is the paper/PR-ready synthesis (improvement fraction, median `ΔRMSE_out`, Wilcoxon `p-value`).

#### Ultra-short runbook (copy/paste)

```bash
# 1) Validate SPARC rotmod coverage (~175 expected)
find data/SPARC/rotmod -name '*_rotmod.dat' | wc -l

# 2) Build full homogeneous catalog
python scripts/build_sparc_full_catalog.py --data-root data/SPARC --out results/SPARC

# 3) Generate F3 catalog from a contract-compliant table
python scripts/generate_f3_catalog_from_contract.py --input data/big_sparc/contract/big_sparc_contract.parquet --out results/SPARC

# 4) Compute deep-regime beta catalog
python scripts/run_big_sparc_veil_test.py --catalog results/SPARC/sparc_full_catalog.csv --out results/SPARC

# 5) Run your branch-specific OOS validation script
python scripts/<your_oos_validation_script>.py

# 6) Inspect generated artifacts
ls results/SPARC

# 7) Read final synthesis
cat results/executive_summary.txt
```

Replace `<your_oos_validation_script>` with the OOS entrypoint available in your branch/pipeline.

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
