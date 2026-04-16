# SCM-spectral-analysis

Spectral analysis of circular-velocity profiles for the **Motor de Velos SCM**
(Fluid Condensation Model) using a cleaned 13-galaxy sample.

## Overview

The Motor de Velos SCM predicts that the 1-D power spectrum of a galaxy's
circular-velocity profile follows a power law:

```
P(k) ∝ k^{-α}
```

where **α** (the *spectral index*) encodes how kinetic energy is distributed
across spatial scales.  This sub-project tests whether α correlates with
baryonic mass and local environment, as expected from the SCM framework.

## Directory structure

```
SCM-spectral-analysis/
│
├── data/
│   └── scm_spectral_clean_13_galaxies.csv   # cleaned 13-galaxy catalog
│
├── scripts/
│   └── spectral_analysis.py                 # main analysis script
│
├── figures/
│   └── (generated figures go here)
│
├── README.md      (this file)
├── CITATION.cff
└── LICENSE
```

## Data

`data/scm_spectral_clean_13_galaxies.csv` — cleaned catalog of 13 dwarf
galaxies with the following columns:

| Column | Units | Description |
|---|---|---|
| `galaxy` | — | Galaxy identifier |
| `logM` | log₁₀(M☉) | Baryonic mass |
| `V_max` | km/s | Maximum circular velocity |
| `sigma_v` | km/s | Line-of-sight velocity dispersion |
| `slope_inner` | — | SCM velocity slope (inner regime) |
| `slope_tail` | — | SCM velocity slope (tail regime) |
| `spectral_index` | — | Power-law index α of velocity power spectrum |
| `env_proxy` | — | Local environment overdensity proxy |
| `quality_flag` | 0/1 | 1 = reliable fit, 0 = uncertain |

## Running the analysis

```bash
# from the repository root
python SCM-spectral-analysis/scripts/spectral_analysis.py

# with explicit paths
python SCM-spectral-analysis/scripts/spectral_analysis.py \
    --csv SCM-spectral-analysis/data/scm_spectral_clean_13_galaxies.csv \
    --out SCM-spectral-analysis/figures
```

This will:
1. Load and validate the 13-galaxy spectral catalog.
2. Compute Spearman ρ between α and logM, and between α and env_proxy.
3. Fit OLS regression: α = a·logM + b.
4. Write diagnostic figures to `figures/`:
   - `spectral_index_vs_logM.png` / `.pdf`
   - `spectral_index_histogram.png` / `.pdf`
5. Write `figures/spectral_summary.csv` with all computed statistics.

## Dependencies

See `requirements.txt` in the repository root:

```
numpy
pandas
scipy
matplotlib
```

## Citation

See `CITATION.cff` in this directory.

## License

MIT — see `LICENSE`.
