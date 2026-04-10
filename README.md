# Motor-de-Velos-SCM

**Author:** Sergio Cámara Madrid  
**Master repository** — single source of truth for all scripts, data, results, and documentation.

For the historical and conceptual background of the SCM (Motor de Velos; Fluid Condensation Model) see
`docs/HISTORICAL_NOTE_MOTOR_DE_VELOS.md`.

---

## Overview

Motor-de-Velos-SCM provides a reproducible, auditable pipeline to evaluate galaxy rotation curves under
the SCM model. Three independent analysis blocks form the core of the evidence:

| Block | Description | Key script | Key result |
|-------|-------------|-----------|------------|
| **SPARC environment** | Environmental modulation of outer-disk slopes across ~175 late-type galaxies | `scripts/scm_tr_regime_test.py` | `results/main/scm_tr_summary.csv` |
| **Yang cross-match** | Replication in the Yang et al. 2007 group catalogue (~89 galaxies) | `scripts/scm_tr_regime_test.py` | `results/yang/scm_tr_yang_dataset.csv` |
| **Gaia / MW shield** | Milky Way Cepheid outer rotation curve (Gaia DR3, 21 control points) | `scripts/mw_delta_f3.py` | `results/gaia/mw_radial_scan.csv` |

Full results and interpretation: `docs/paper1/SCM_TR_results.md`

---

## Repository structure

```
Motor-de-Velos-SCM/
├── README.md
├── requirements.txt
├── .github/workflows/          # CI (ci.yml, sparc_validation.yml)
├── data/
│   ├── raw/                    # Unmodified source data (LITTLE THINGS, Gaia)
│   ├── processed/              # Clean, pipeline-ready CSVs
│   └── README.md
├── results/
│   ├── diagnostics/            # Model-comparison and deep-slope diagnostics
│   ├── main/                   # Primary SPARC SCM-TR results
│   ├── yang/                   # Yang 2007 cross-match results
│   ├── gaia/                   # MW Gaia radial-scan results
│   ├── scm_environment/        # Environment-proxy sensitivity outputs
│   ├── scm_oos/                # Out-of-sample validation outputs
│   ├── extreme_25/             # Extreme-quartile sub-sample outputs
│   ├── delta_f3/               # Delta-F3 catalog and analysis outputs
│   ├── paper1_environment/     # Paper-1 environment-block final figures and tables
│   ├── blind_test_lt/          # LITTLE THINGS blind-test outputs
│   └── lt_dust_hinge/          # LITTLE THINGS dust-hinge analysis
├── scripts/                    # Canonical entry-point scripts (see below)
├── src/                        # Core library modules (scm_analysis, scm_models, sensitivity)
├── tests/                      # pytest test suite
└── docs/
    ├── paper1/                 # Paper-1 manuscript assets
    │   ├── SCM_TR_results.md   # Three-shield test results (main reference)
    │   ├── abstract.md
    │   ├── methods_delta_mass.md
    │   ├── results_summary.md
    │   └── figures/
    └── notes/                  # Working notes and hypothesis documents
```

---

## Canonical scripts

| Script | Purpose | Default output |
|--------|---------|----------------|
| `scripts/scm_tr_regime_test.py` | SCM-TR three-shield test (Spearman, bootstrap, HC3, mass scan) | `results/main/scm_tr_summary.json`, `mass_scan.csv` |
| `scripts/mw_delta_f3.py` | MW Cepheid delta-F3 pipeline with radial scan | `results/mw_delta_f3.{png,pdf}` |
| `scripts/plot_sparc_high_mass_regression.py` | OLS scatter figure for SPARC high-mass sub-sample | `results/scm_high_mass_regression.{png,pdf}` |
| `scripts/build_galaxy_catalog_env.py` | Merge SPARC + slopes + env proxy into master catalog | `data/galaxy_catalog_env.csv` |
| `scripts/compute_slope_tail.py` | Compute outer log-slope (slope_tail) for each galaxy | `results/slope_tail.csv` |
| `scripts/mass_split_analysis.py` | Mass-split histogram and summary | `results/mass_split/` |
| `scripts/env_mass_scan.py` | Environment x mass grid scan | `results/env_mass_scan/` |
| `scripts/run_full_pipeline.py` | Run all four entry-point steps in sequence | — |
| `scripts/run_pipeline.py` | Subprocess-based pipeline orchestrator (dry-run, skip flags) | — |
| `scripts/deep_slope_test.py` | Deep-regime slope diagnostic | `results/diagnostics/deep_slope_test/` |
| `scripts/blind_test_little_things.py` | Pre-registered out-of-sample test on LITTLE THINGS | `results/blind_test_lt/` |

---

## Installation

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run the test suite to verify the environment:

```bash
python -m pytest -q
```

---

## Reproducing key results

### Block 1 — SPARC three-shield test

```bash
python scripts/scm_tr_regime_test.py
```

Outputs: `results/main/scm_tr_summary.json`, `results/main/mass_scan.csv`, figures.  
Pre-computed summary: `results/main/scm_tr_summary.csv`

### Block 2 — Milky Way Cepheid radial scan (Gaia DR3)

```bash
python scripts/mw_delta_f3.py
```

Default R_cut = 13 kpc; result: slope_tail = -0.164, delta-F3_MW = -0.664, p = 2.65e-16.  
Radial-scan best: R_crit = 16.5 kpc, slope = -0.197, p = 4.5e-13.  
Pre-computed scan: `results/gaia/mw_radial_scan.csv`

### Block 3 — High-mass regression figure

```bash
python scripts/plot_sparc_high_mass_regression.py
```

Output: `results/scm_high_mass_regression.{png,pdf}`

### Full pipeline (all entry-point steps)

```bash
python scripts/run_full_pipeline.py
# or with dry-run / skip options:
python scripts/run_pipeline.py --dry-run
```

---

## Data

| File | Description | Source |
|------|-------------|--------|
| `data/mw_cepheids.csv` | MW Cepheid rotation-curve control points | Gaia DR3 |
| `data/little_things_global.csv` | LITTLE THINGS global properties | Oh et al. 2015 |
| `data/raw/lt_oh2015/` | Individual LITTLE THINGS rotation curves | Oh et al. 2015 |
| `data/raw/lt_masses.csv` | LITTLE THINGS stellar masses | — |
| `data/raw/lt_metals.csv` | LITTLE THINGS metallicities | — |
| `data/raw/cigan2021_tdust.csv` | Dust temperatures (Cigan et al. 2021) | — |

Large raw datasets (SPARC, full Gaia) are not versioned. See `data/README.md` for download instructions.

---

## Statistical protocol

- Spearman correlation with 2000-resample bootstrap (BCa intervals).
- OLS with HC3 heteroscedasticity-robust standard errors.
- Fisher Z test for cross-dataset consistency.
- Mass split at log M_bar = 10.05 solar masses (M_CRIT_DEFAULT).
- MW: beta_ref = 0.5, R_cut_default = 13.0 kpc.
- All pre-registered decisions documented in `docs/paper1/`.

---

## Results summary

See `docs/paper1/SCM_TR_results.md` for the full three-shield summary.

| Shield | Dataset | N | rho_Spearman | p | Consistent? |
|--------|---------|---|-------------|---|-------------|
| Main | SPARC | 168 | 0.418 | 0.0085 | yes |
| Yang | Yang 2007 | 89 | 0.391 | 0.021 | yes |
| Gaia/MW | MW Cepheids | 21 | slope = -0.164 | 2.6e-16 | yes |

---

## License

Refer to the LICENSE file.  
Commercial use restrictions: see `COMMERCIAL_USE.md`.

## Contact

Author: Sergio Cámara Madrid  
Repository: https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM
