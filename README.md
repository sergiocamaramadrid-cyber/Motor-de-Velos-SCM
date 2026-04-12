# SCM — Motor de Velos: Environmental Modulation in Galaxy Rotation Curves

**Author:** Sergio Cámara Madrid · Independent researcher  
**Zenodo DOI:** [10.5281/zenodo.19455777](https://doi.org/10.5281/zenodo.19455777)

> For historical context and conceptual origins of the SCM framework, see  
> `docs/HISTORICAL_NOTE_MOTOR_DE_VELOS.md`

---

## Overview

This repository contains the data products and analysis results of the SCM (Motor de Velos) framework, a phenomenological approach to studying the outer dynamics of disk galaxies.

The central goal is to test whether environmental effects leave a measurable imprint on the outer rotation curve slope beyond what is explained by baryonic mass alone.

---

## Key Result

Using a cleaned SPARC-based sample (N = 61 galaxies):

**Global correlation**

- Spearman ρ ≈ −0.365, p ≈ 9.3 × 10⁻⁴

**Mass-regime transition** (threshold: log M ≈ 10.6)

| Regime | log M | ρ | p |
|--------|-------|---|---|
| Low mass | < 10.6 | not significant | — |
| High mass | ≥ 10.6 | −0.44 to −0.49 | 10⁻³ – 10⁻⁴ |

**Residual test (mass controlled)**

- Environmental signal persists in the high-mass regime
- Disappears in the low-mass regime

---

## Interpretation

> *"The baryonic mass does not generate the environmental signal, but rather masks it."*

Only when galaxies reach a sufficient mass scale (log M ≳ 10.6) does the outer disk become dynamically sensitive to environmental coupling.

Working hypothesis:

> *"Mass acts as an accumulation of signals; environment emerges once structural coupling exceeds noise."*

This work does not claim a complete physical model. It provides a testable observational signal:

> *"Environmental modulation of outer galaxy dynamics, emerging only above a critical mass scale."*

---

## Data Products

### Main dataset

`results/scm_master_final.csv` — 61 galaxies (G001–G061)

| Column | Description |
|--------|-------------|
| `galaxy` | Galaxy identifier |
| `logMbar` | Baryonic mass (log scale) |
| `env_proxy` | Environmental proxy |
| `slope_tail` | Outer rotation curve slope (d log V / d log r) |
| `delta_f3` | Deviation from reference slope (slope_tail − 0.5) |
| `regime` | Mass regime (`low` / `high`, threshold log M ≈ 10.6) |
| `quality_flag` | Data quality indicator |
| `n_tail_points` | Number of outer points used in fit |

### Supporting results

| File | Description |
|------|-------------|
| `results/sparc_mass_scan.csv` | Mass-threshold scan showing emergence of environmental signal |
| `results/nested_models_comparison.csv` | Model comparison (AIC, BIC, adjusted R²) including environmental term |
| `results/mw_radial_scan.csv` | Milky Way outer slope radial analysis (Gaia-based) |

---

## Method Summary

- Outer slope measured at r ≥ 0.7 R_max
- Minimum tail points: n ≥ 4–5
- Correlation: Spearman rank test
- Regression: OLS with HC3 robust errors
- Model selection: AIC / ΔAIC
- Residual analysis used to isolate environmental contribution

---

## Repository Structure

```
data/       input catalogs (SPARC, auxiliary)
results/    processed datasets and outputs
scripts/    analysis scripts
docs/       paper material and figures
tests/      unit and integration tests
```

---

## Installation

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run tests: `pytest`

---

## Reproducibility

This repository provides:

- Clean dataset ready for direct statistical testing
- Standardized column definitions (`delta_f3 = slope_tail − 0.5`, β_ref = 0.5)
- Fully reproducible analysis pipeline (see `scripts/`)

Anyone can test, replicate, extend, or challenge these results.

---

## Citation

```
Sergio Cámara Madrid (2026). SCM — Motor de Velos: Environmental Modulation
in Galaxy Rotation Curves. Zenodo. https://doi.org/10.5281/zenodo.19455777
```

---

## License

Refer to the LICENSE file.

---

**Keywords:** Galaxy dynamics · Rotation curves · SPARC · Environment · Dark matter · Astrophysics · Data analysis