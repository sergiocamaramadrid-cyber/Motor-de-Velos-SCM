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
- HC3-robust OLS regression for environmental dependence in high-mass subsamples.
- Versioned, machine‑readable outputs and logging to support audit and replication.

Design goals
- Reproducible: runs record input checksums and git commit hashes when generating results.
- Deterministic: deterministic preprocessing and evaluation steps.
- Audit-friendly: clear inputs/outputs and diagnostics.
- Version-controlled: code and analysis scripts tracked in the repository.

---

## Repository structure

```
Motor-de-Velos-SCM/
├── data/
│   ├── sparc_subset.csv          ← 79-galaxy SPARC subset (versioned fixture)
│   ├── little_things_global.csv  ← LITTLE THINGS global properties
│   └── raw/                      ← small raw reference tables
├── docs/
│   ├── HISTORICAL_NOTE_MOTOR_DE_VELOS.md
│   ├── Hipotesis_2.4_Sensibilidad_al_entorno.md
│   ├── Hipotesis_2.5_Consistencia_proxies.md
│   ├── Section3_results.md
│   └── paper1/                   ← manuscript draft assets
├── results/                      ← generated outputs (not versioned in git)
├── scripts/                      ← CLI analysis scripts (see below)
├── src/                          ← core model library
│   ├── scm_models.py
│   ├── scm_analysis.py
│   ├── sensitivity.py
│   └── lt/lt_dust_hinge_analysis.py
├── tests/                        ← pytest test suite (443 tests, all passing)
├── requirements.txt
└── COMMERCIAL_USE.md
```

---

## Installation

### Requirements

- Python 3.10 or later.
- System tools: git.
- Dependencies: see `requirements.txt` (includes `numpy`, `pandas`, `scipy`,
  `statsmodels`, `matplotlib`, `pyarrow`, `pytest`, `tqdm`).

### Setup

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# upgrade pip and install all dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the test suite

```bash
python -m pytest -q
# Expected: 443 passed
```

---

## Data Policy

Raw datasets (e.g., full SPARC catalogue, LITTLE THINGS) are **not versioned**.  
Generated results are **not versioned**.  
Small curated fixtures (e.g., `data/sparc_subset.csv`, N = 79) **are versioned**.  
Download and preprocessing scripts are provided for reproducibility (see `scripts/download_sparc_data.py`).

---

## Scripts inventory

| Script | Description | Tests |
|---|---|---|
| `plot_sparc_split_mass.py` | δF₃ vs δ_mass scatter, split at median logM | 56 |
| `plot_sparc_mass_scan.py` | Sweep logM cuts, find optimal threshold | 51 |
| `sparc_ols_regression.py` | OLS regression (HC3) on high-mass subsample | 52 |
| `deep_slope_test.py` | Deep-regime slope diagnostic | 18 |
| `generate_f3_catalog.py` | Per-galaxy F3/β catalog from SPARC data files | 12 |
| `generate_f3_catalog_from_contract.py` | F3/β catalog (v2) from Parquet contract | 12 |
| `ingest_sparc_contract.py` | Ingest standard SPARC data → Parquet contract | 11 |
| `ingest_big_sparc_contract.py` | Ingest BIG-SPARC data → Parquet contract | 6 |
| `f3_catalog_analysis.py` | Statistical analysis of the F3 catalog | 12 |
| `compare_nu_models.py` | AICc-based model comparison | 11 |
| `blind_test_little_things.py` | LITTLE THINGS blind-test pipeline | 12 |
| `generate_env_figure.py` | Environmental correlation figure generator | — |
| `download_sparc_data.py` | Download raw SPARC data from the web | — |
| `contract_utils.py` | Shared data-contract utilities | — |

---

## Running the analyses

### SPARC Subset — Split-by-Mass Environmental Analysis

A curated subset of 79 SPARC galaxies (`data/sparc_subset.csv`) is committed to
the repository for fully reproducible environmental analyses.
Columns: `galaxy`, `logM`, `delta_mass_std`, `slope_tail`.

Generate the two-panel δF₃ vs δ_mass,std figure split at the sample median of logM:

```bash
python scripts/plot_sparc_split_mass.py
# with explicit paths:
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
low-mass regime does not.

### SPARC Subset — Mass Threshold Scan

Sweeps logM cuts from 10.0 to 11.3 (step 0.05), computes Spearman ρ for
galaxies with `logM ≥ m_cut`, and identifies the cut that maximises the
composite signal score `|ρ| × √N × (−log₁₀ p)`.

```bash
python scripts/plot_sparc_mass_scan.py
# with explicit paths and custom range:
python scripts/plot_sparc_mass_scan.py \
  --csv data/sparc_subset.csv \
  --out results/sparc_mass_scan.png \
  --m-start 10.0 --m-stop 11.3 --m-step 0.05 --n-min 15
```

Key result on the committed 79-galaxy catalog:

| logM cut | N  | ρ (Spearman) | p-value | Score |
|---|---|---|---|---|
| **10.05** (best) | 56 | **−0.48** | **1.8 × 10⁻⁴** | 13.46 |

All 21 evaluated cuts show ρ < 0 and p < 0.05.

### High-Mass OLS Regression (HC3)

Fits two nested OLS models on the N = 56 galaxies with logM ≥ 10.05, using
HC3 heteroscedasticity-robust standard errors:

- **Model 1** (simple): `δF₃ ~ δ_mass_std`
- **Model 2** (mass-controlled): `δF₃ ~ δ_mass_std + logM`

```bash
python scripts/sparc_ols_regression.py
# save summary to file:
python scripts/sparc_ols_regression.py \
  --csv data/sparc_subset.csv \
  --m-crit 10.05 \
  --out results/ols_summary.txt
```

Key result: the `delta_mass_std` coefficient is **negative and p < 0.05 (HC3)**
in both models, confirming the environmental signal survives mass control.

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
- HC3-robust OLS for environmental subgroup analyses
- Deterministic merge contracts (Parquet v2 pipeline)
- Explicit deep-regime slope test
- Versioned output naming

---

## Reproducibility

Each run should record:

- Git commit hash  
- Input file checksums  
- Command-line arguments  
- Parameter values (e.g., g0, thresholds, m_crit)

Outputs should be written under:
```
results/<module>/<artifact>-v<semver>.csv
```

---

## Limitations

The framework evaluates rotation-curve behaviour; it does not claim cosmological completeness.  
Statistical validation is dataset-dependent.  
Interpretation remains separate from computational reproducibility.

---

## Citation

See `COMMERCIAL_USE.md` and the Zenodo archive (DOI when available).

---

## License

Refer to the LICENSE file.

---

## Contact

Author: Sergio Cámara Madrid  
Repository: https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM