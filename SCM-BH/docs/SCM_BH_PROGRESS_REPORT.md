# SCM-BH Progress Report

**Author:** Sergio Cámara Madrid  
**Date:** 2026-04-24  
**Tag:** scm-bh-v0.1  
**Status:** Exploratory — reproducible but not yet a final publication claim

---

## 1. Hypothesis

**SCM-BH**: The observable geometry of the jet in AGN is not governed directly by black-hole mass or Eddington rate, but primarily by the internal structure of the flow.

The observed opening angle θ_jet depends on the structural scale of the jet, r15, while the Doppler factor δ acts as a secondary observational modulator, producing a regime transition: at low δ the θ_jet–δ relation is significant, while at high δ it saturates.

---

## 2. Data Used

### 2.1 MOJAVE (15 GHz VLBI)

- Source: Lister et al. (2019), MOJAVE survey.
- Variables used: source name, jet opening angle θ_jet (deg), structural scale r15 (pc), Doppler factor δ.
- Sample: 175 AGN with jet-opening measurements at 15 GHz.

### 2.2 Wu & Shen DR16Q (2022)

- Source: Wu & Shen (2022), SDSS DR16 quasar physical properties catalog.
- Variables used: LOGMBH (log solar masses), LOGLBOL (log erg/s), LOGEDD_RATIO.
- Matched via sky coordinates (match radius: 3 arcsec).

### 2.3 Merged Sample

- Stage 3 merged sample: N = 77 AGN with complete θ_jet, r15, δ, LOGMBH, LOGLBOL, LOGEDD_RATIO.
- Stage 4 rescue: additional match-radius diagnostic to recover borderline objects.
- Files:
  - `data/processed/scm_bh_stage4_mojave_full_wushen.csv` — full matched sample
  - `data/processed/wu_shen_2022_compact.csv` — Wu & Shen compact reference
  - `data/processed/scm_bh_stage4_rescued_master.csv` — rescued master sample

---

## 3. Steps Executed

### Stage 1 — Data ingestion

1. Load MOJAVE catalog (θ_jet, r15, δ per source).
2. Load Wu & Shen DR16Q (LOGMBH, LOGLBOL, LOGEDD_RATIO per source).
3. Cross-match by sky position (3 arcsec radius).

### Stage 2 — Exploratory correlations (engine variables)

1. Spearman ρ: LOGMBH vs θ_jet → no signal.
2. Spearman ρ: LOGLBOL vs θ_jet → no signal.
3. Spearman ρ: LOGEDD_RATIO vs θ_jet → not robust at N=77.

### Stage 3 — Structural regression

1. OLS: θ_jet ~ log(r15) → R² ≈ 0.25–0.32.
2. Spearman ρ: δ vs θ_jet (global) → ρ ≈ 0.31, p ≈ 0.011.
3. Split sample: LOW δ vs HIGH δ (median split).
   - LOW δ: ρ ≈ 0.52, p ≈ 0.002.
   - HIGH δ: ρ ≈ 0.02, p ≈ 0.91 (saturated).

### Stage 4 — Combined model and diagnostics

1. OLS: θ_jet ~ log(r15) + δ + δ·log(r15) → R² ≈ 0.405, p_global ≈ 2×10⁻⁶.
2. Match-radius diagnostic: check sensitivity of N to matching tolerance.
3. Rescue protocol: recover borderline sources; assess impact on results.

---

## 4. Numerical Results

### 4.1 Structural layer

| Model | R² | Notes |
|---|---|---|
| θ_jet ~ log(r15) | 0.25–0.32 | Robust across match tolerances |

### 4.2 Engine variables

| Variable | ρ | p | Verdict |
|---|---|---|---|
| LOGMBH vs θ_jet | ~0.05 | >0.3 | No signal |
| LOGLBOL vs θ_jet | ~0.07 | >0.2 | No signal |
| LOGEDD_RATIO vs θ_jet | ~0.12 | >0.15 | Not robust, N=77 |

### 4.3 Doppler modulation

| Subsample | ρ | p | Verdict |
|---|---|---|---|
| Global δ vs θ_jet | 0.31 | 0.011 | Significant |
| LOW δ | 0.52 | 0.002 | Strong |
| HIGH δ | 0.02 | 0.91 | Saturated |

### 4.4 Combined model

| Model | R² | p (global) |
|---|---|---|
| θ_jet ~ log(r15) + δ + δ·log(r15) | 0.405 | 2×10⁻⁶ |

---

## 5. Figures

- `results/figure_theta_r15_delta.png`:  
  Main result figure showing θ_jet vs log(r15) coloured by δ regime (low vs high), with regression lines and Spearman annotations.

---

## 6. Interpretation

### Two-layer model

**Layer 1 — Structural:**  
Jet opening angle is primarily governed by intrinsic jet scale r15. This is the dominant explanatory variable. Neither black-hole mass nor Eddington ratio contributes robustly.

**Layer 2 — Relativistic observational:**  
Doppler factor δ modulates the observed geometry. In the low-δ regime the Doppler-geometry coupling is strong (ρ ≈ 0.52, p ≈ 0.002). In the high-δ regime the relation saturates (ρ ≈ 0.02), consistent with Doppler boosting suppressing geometric variation.

### Regime transition

The low-δ / high-δ transition is consistent with differential Doppler compression of the apparent opening angle:
- Low δ: geometry partially uncompressed → structural variation visible.
- High δ: geometry compressed by boosting → variation suppressed.

---

## 7. Verdict

**SCM-BH does not support a direct engine-driven model for jet opening angle.**

The evidence favours a two-layer interpretation in which the structural scale r15 is the primary driver and the Doppler factor δ is a secondary observational modulator with a clear regime transition.

**Caveats:**
- Sample size N=77 limits the power to detect weak engine-variable effects.
- The match radius (3 arcsec) introduces a selection effect that was partially addressed by the Stage 4 rescue protocol.
- Results are exploratory; formal hypothesis registration and pre-registration were not performed before data inspection.

---

## 8. Next Steps

- [ ] Extend match radius diagnostic to quantify sensitivity of R² to matching tolerance.
- [ ] Test residual engine-variable correlations after partialling out log(r15) and δ.
- [ ] Investigate alternative structural variables (e.g., apparent speed βapp).
- [ ] Compare with VLBA-BU-BLAZAR sample.
- [ ] Prepare formal analysis plan before running additional tests (pre-registration).

---

## 9. Files in This Release

| File | Description |
|---|---|
| `data/processed/scm_bh_stage4_mojave_full_wushen.csv` | MOJAVE + Wu & Shen matched sample |
| `data/processed/wu_shen_2022_compact.csv` | Wu & Shen DR16Q compact table |
| `data/processed/scm_bh_stage4_rescued_master.csv` | Rescued master sample |
| `results/scm_bh_stage3_final_summary.json` | Stage 3 numerical summary |
| `results/scm_bh_stage4_match_radius_diagnostic.csv` | Match-radius diagnostic table |
| `results/scm_bh_stage4_rescue_summary.json` | Stage 4 rescue summary |
| `results/figure_theta_r15_delta.png` | Main result figure |
| `docs/SCM_BH_PROGRESS_REPORT.md` | This document |

---

## References

- Lister, M. L., et al. (2019). MOJAVE XVI. Multiepoch linear polarization properties of parsec-scale AGN jet cores. *ApJ*, 874, 43.
- Wu, Q., & Shen, Y. (2022). A Catalog of Quasar Properties from Sloan Digital Sky Survey Data Release 16. *ApJS*, 263, 42.
