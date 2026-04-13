# SCM — Motor de Velos: Framework Final (2026-04-13)

**Author:** Sergio Cámara Madrid  
**Release date:** 2026-04-13  
**DOI:** 10.5281/zenodo.19455777

---

## Overview

This package contains the final, self-contained release of the **SCM (Motor de Velos / Fluid Condensation Model)** computational framework for evaluating galaxy rotation curves and environmental modulation effects.

The framework implements:
- Per-galaxy deep-regime slope β fitting via the F3 relation.
- Gas-fraction analysis (`análisis_gas.py`).
- Environment proxy catalog construction (`build_e_env.py`).
- A unified entry-point pipeline (`run_scm.py`).

---

## Repository structure

```
SCM_Framework_Final_20260413_165000/
│
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
│
├── data/
│   ├── galaxy_catalog.csv      ← per-galaxy catalog (logM, logVobs, log_gbar, beta, …)
│   └── sparc_basic.csv         ← SPARC-derived baryonic summary table
│
├── results/
│   ├── coeficientes_finales.csv   ← final fitted SCM coefficients
│   ├── resultados_gas.csv         ← gas-fraction analysis outputs
│   └── figura_campanada.png       ← β distribution ("bell") figure
│
├── scripts/
│   ├── run_scm.py        ← MAIN: end-to-end pipeline entry point
│   ├── análisis_gas.py   ← gas-velocity / gas-fraction analysis module
│   └── build_e_env.py    ← environment-proxy catalog builder
│
└── docs/
    ├── paper_summary.md       ← condensed scientific summary
    └── resumen_ejecutivo.txt  ← executive summary (Spanish)
```

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

---

## Quick start

```bash
# Run the full pipeline (reads data/, writes results/)
python scripts/run_scm.py

# Gas analysis only
python scripts/análisis_gas.py --catalog data/galaxy_catalog.csv --out results/resultados_gas.csv

# Build environment proxy catalog
python scripts/build_e_env.py --catalog data/sparc_basic.csv --out data/galaxy_catalog.csv
```

---

## Key results

| Metric | Value |
|--------|-------|
| N galaxies (reliable β) | 162 |
| Mean β | 0.503 ± 0.041 |
| Δ from β=0.5 | +0.003 |
| p-value (t-test vs β=0.5) | 0.47 |
| Spearman ρ (env vs β) | 0.48 |
| ΔAIC vs baryonic-only | 6.2 |

---

## Citation

Please cite using `CITATION.cff` or:

> Cámara Madrid, S. (2026). *SCM — Motor de Velos: Environmental Modulation in Galaxy Rotation Curves*. Zenodo. https://doi.org/10.5281/zenodo.19455777

---

## License

MIT — see `LICENSE`.
