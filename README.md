# SCM — Structured Residuals in Galaxy Kinematics

## Overview

This repository contains the implementation of the Structural Coupling Model (SCM), an empirical framework designed to test whether residuals from standard kinematic models in astrophysical systems are consistent with stochastic noise or exhibit structured behaviour.

The central question addressed is:

> Are deviations from global kinematic relations purely random, or do they contain statistically significant structure?

This project does not propose a new physical law. Instead, it provides a reproducible statistical test of residual structure across independent datasets.

---

## Core Result

Across multiple datasets, we find:

✔ Residuals are statistically structured (bootstrap ΔRSS confidence intervals exclude zero in SPARC)

✘ No statistically robust mass-threshold transition detected

✘ No statistically significant environmental modulation

✔ Existence of null regimes consistent with noise-dominated systems (e.g. LITTLE THINGS)

**Conclusion:**

> Residuals in galaxy kinematics are not purely stochastic, but their physical origin (mass-driven, environment-driven, or multi-variable coupling) cannot be established with current sample sizes and proxies.

---

## Datasets

| Dataset | Description | N | Result |
|---|---|---|---|
| SPARC | Galaxy rotation curves | 92 | Structured residuals (P1 ✔) |
| YANG | Group catalog + environment proxies | 79 | No significant environmental dependence (P3 ✘) |
| LITTLE THINGS | Dwarf irregular galaxies | 26 | Null result (noise-dominated regime) |
| MOJAVE | Relativistic jet sample | ~65 | Weak / inconclusive signal |

All datasets used are publicly available and referenced below.

---

## Methodology

### Baseline Model

A global linear model is fitted:

```
y = β₀ + β₁ x + ε
```

where ε represents the residual component.

### Residual Structure Test (P1)

We test whether the residuals are structured using:

- Bootstrap resampling (≥ 1000 iterations)
- Confidence interval of ΔRSS = RSS_global − RSS_model

**Criterion:**
- ✔ Structured if IC95% does not include 0
- ✘ Not structured otherwise

### Threshold Test (P2)

A piecewise model is evaluated:

- Grid search over threshold parameter τ
- Permutation test (≥ 500 iterations)

**Criterion:**
- ✔ Transition if p_perm < 0.05 and σ(τ) ≤ 0.15
- ✘ Otherwise not robust

### Environmental Test (P3)

Regression of residuals vs environmental proxy, stratified by mass quartiles and evaluated in the high-mass regime (Q4).

**Criterion:**
- ✔ Significant if β_env < 0 and p < 0.05
- ✘ Otherwise not supported

---

## Falsifiability Criteria

The SCM framework is considered falsified if any of the following occur:

1. **No residual structure** — IC95%(ΔRSS) includes 0 across datasets
2. **Instability under resampling** — results fail under bootstrap / permutation / out-of-sample tests
3. **False positives in control datasets** — null systems (e.g. LITTLE THINGS) show significant structure
4. **Disappearance with larger samples** — signal weakens systematically as N increases

These criteria are explicitly tested in this repository.

---

## Final Results Summary

| Test | Result | Interpretation |
|---|---|---|
| P1 — Residual structure (SPARC) | ✔ Confirmed | Residuals are not noise |
| P2 — Threshold | ✘ Not robust | No evidence of mass transition |
| P3 — Environment (YANG) | ✘ Not significant | No confirmed modulation |

See `results/scm_final_results.csv` for the full validated results table.

---

## Reproducibility

All results can be reproduced using:

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_scm.py
```

Outputs include: bootstrap distributions, permutation statistics, final results table, and figures.

---

## Repository Structure

- `scripts/` → analysis scripts (main pipeline entry points)
- `tests/` → unit and integration tests
- `data/` → processed input catalogs
- `results/` → validated output tables and figures
- `docs/` → technical documentation and development notes
- `notebooks/experimental/` → exploratory pipelines (not production)

---

## Scientific Interpretation

The key finding is:

> Standard kinematic models fail systematically in a structured way, but this structure cannot yet be attributed to a single physical driver.

This reframes the problem:

- Not "find the law"
- But "characterise the failure of existing laws"

---

## Limitations

- Sample sizes are small, especially in the high-mass regime
- Environmental proxies may be incomplete representations of true environment
- The linear baseline model may underfit complex dynamics

---

## Future Work

To test the physical origin of residual structure:

- Increase N in high-mass regime
- Improve environmental metrics
- Explore multi-variable coupling models

---

## Positioning

This work is intentionally conservative:

- No new physics is claimed
- No overinterpretation of marginal signals
- All conclusions follow directly from statistical tests

---

## Citation

> DOI: 10.5281/zenodo.19897353

If you use this work, please also cite the underlying datasets:

- Lelli et al. 2016 (SPARC)
- Yang et al. 2007 (YANG)
- Hunter et al. 2012 (LITTLE THINGS)
- Lister et al. 2019 (MOJAVE)

See also `CITATION.cff` and the Zenodo archive.

---

## Installation

### Requirements

- Python 3.10 or later
- Dependencies: see `requirements.txt`

---

## License

Refer to the LICENSE file.

---

## Author

Sergio Cámara Madrid — Independent Researcher  
Repository: https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM

---

> This repository demonstrates that kinematic residuals contain structured information not captured by standard models.  
> Determining the physical origin of this structure remains an open problem.