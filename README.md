# SCM Framework — Environmental Regime Transition

## Overview

This repository contains the final environmental analysis of the **SCM (Slope Coupling Model)**
framework applied to the SPARC galaxy sample.

The analysis demonstrates a **regime-dependent environmental modulation** of outer rotation curve
slopes: a statistically significant effect is detected only above a critical baryonic mass scale.

---

## Key Result

| Regime | N | Spearman ρ | p-value | OLS β (HC3) | OLS p | R² |
|---|---|---|---|---|---|---|
| High-mass (logM ≥ 10) | 47 | −0.463 | 2.8 × 10⁻⁴ | −0.143 | 0.004 | 0.25 |
| Low-mass (logM < 10)  | 32 | ≈ 0.003 | ≈ 0.98 | — | — | — |

This establishes a **mass-dependent transition (SCM-TR)** in the coupling between galaxies and
their environment.

---

## Robustness

| Test | Result |
|---|---|
| Bootstrap β (95% CI) | −0.168 \[−0.266, −0.094\] |
| Permutation p | 0.0002 |
| Multivariate control | Environmental effect significant after controlling for baryonic mass |

---

## Interpretation

- **Low-mass galaxies** — dynamics dominated by internal processes; no environmental signal.
- **High-mass galaxies** — external environment modulates outer rotation curves.

This supports a **regime-dependent interaction between baryonic structure and environment**.

---

## Repository Structure

```
Motor-de-Velos-SCM/
├── data/                         Raw input data and fixtures
│   ├── little_things_global.csv
│   └── README.md
├── docs/                         Scientific documentation and reports
│   ├── SCM_FINAL_REPORT.docx
│   ├── HISTORICAL_NOTE_MOTOR_DE_VELOS.md
│   └── paper1/
├── results/                      Final analysis outputs
│   ├── paper1_environment/
│   │   ├── SCM_figure_final.png
│   │   ├── SCM_figure_final.pdf
│   │   └── SCM_results_table.csv
│   ├── blind_test_lt/
│   ├── diagnostics/
│   └── lt_dust_hinge/
├── scripts/                      Analysis and diagnostic scripts
├── src/                          Core model implementations
├── tests/                        Unit and integration tests
├── requirements.txt
├── CITATION.cff
└── README.md
```

---

## Reproducibility

Install dependencies and run the full SCM environment analysis:

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
pip install -r requirements.txt
python scripts/generate_env_figure.py
```

Run diagnostic tests:

```bash
python scripts/deep_slope_test.py
python scripts/compare_nu_models.py
```

Run the test suite:

```bash
pytest
```

---

## Data Sources

- SPARC database (Lelli et al. 2016)
- Yang et al. (2007) environment catalog

---

## Citation

If you use this work, please cite the Zenodo DOI associated with this repository.
See `CITATION.cff` for the full citation record.

---

## Status

✔ Fully reproducible  
✔ Statistically robust  
✔ Publication-ready  

---

## License

See `LICENSE`. Commercial use: see `COMMERCIAL_USE.md`.

---

## Author

Sergio Cámara Madrid  
Independent Researcher — SCM Framework  
Repository: <https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM>
