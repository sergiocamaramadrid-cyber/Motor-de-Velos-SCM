# SCM-BH: Jet Structure and Doppler-Regime Modulation in MOJAVE AGN

Author: Sergio Cámara Madrid  
Status: Exploratory — not yet a final publication claim  
Tag: `scm-bh-v0.1`

---

## Overview

This repository contains the exploratory SCM-BH analysis of AGN jet geometry using MOJAVE jet-opening measurements and Wu & Shen DR16Q physical properties.

The main result is that observed jet opening angles are dominated by intrinsic structural scale r15, while black-hole mass, bolometric luminosity, and Eddington ratio do not show robust direct or residual correlations. A secondary Doppler-regime modulation is detected: δ correlates with θ_jet globally and strongly in the low-δ regime, while the relation saturates at high δ.

---

## Hypothesis (SCM-BH)

> The observable geometry of the jet in AGN is not governed directly by black-hole mass or Eddington rate, but primarily by the internal structure of the flow.

The observed opening angle θ_jet depends on the structural scale of the jet, r15, while the Doppler factor δ acts as a secondary observational modulator, producing a regime transition: at low δ the θ_jet–δ relation is significant, while at high δ it saturates.

---

## Central Result

### Structural layer

| Relation | Value |
|---|---|
| θ_jet ~ log(r15) | R² ≈ 0.25–0.32 |

### Engine (no signal)

| Variable | Result |
|---|---|
| LOGEDD_RATIO vs θ_jet | Not robust, N=77 |
| LOGMBH vs θ_jet | No signal |
| LOGLBOL vs θ_jet | No signal |

### Relativistic modulation

| Subsample | ρ | p |
|---|---|---|
| Global δ vs θ_jet | 0.31 | 0.011 |
| LOW δ | 0.52 | 0.002 |
| HIGH δ | 0.02 | 0.91 |

### Combined model

| Model | Value |
|---|---|
| θ_jet ~ log(r15) + δ + δ·log(r15) | R² ≈ 0.405 |
| Global p | ≈ 2×10⁻⁶ |

---

## Verdict

SCM-BH does **not** support a direct engine-driven model for jet opening angle.

Instead, the evidence favours a two-layer interpretation:

1. **Structural layer:** jet opening angle is primarily governed by intrinsic jet scale r15.
2. **Relativistic observational layer:** Doppler factor δ modulates the observed geometry, with a regime transition between low-δ and high-δ jets.

This is a working result, not yet a final MNRAS paper.

---

## Repository Structure

```
SCM-BH/
├── data/
│   └── processed/
│       ├── scm_bh_stage4_mojave_full_wushen.csv   # MOJAVE + Wu & Shen matched sample (N=77)
│       ├── wu_shen_2022_compact.csv               # Wu & Shen DR16Q compact reference table
│       └── scm_bh_stage4_rescued_master.csv       # Rescued master sample after radius diagnostic
├── results/
│   ├── scm_bh_stage3_final_summary.json           # Stage 3 numerical summary
│   ├── scm_bh_stage4_match_radius_diagnostic.csv  # Match-radius diagnostic table
│   ├── scm_bh_stage4_rescue_summary.json          # Stage 4 rescue summary
│   └── figure_theta_r15_delta.png                 # Main result figure
├── docs/
│   └── SCM_BH_PROGRESS_REPORT.md                  # Full progress report
├── README.md
└── CITATION.cff
```

---

## Data Sources

- **MOJAVE** jet opening angles: Lister et al. (2019), 15 GHz VLBI imaging survey.  
  <https://www.cv.nrao.edu/MOJAVE/>
- **Wu & Shen DR16Q**: Wu & Shen (2022), SDSS DR16 quasar physical properties.  
  <https://doi.org/10.3847/1538-4365/ac9ded>

---

## Citation

See `CITATION.cff` in this folder.

---

## License

MIT — see top-level LICENSE file.
