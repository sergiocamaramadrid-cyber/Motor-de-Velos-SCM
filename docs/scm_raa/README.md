# SCM-RAA v2.6-experimental

## Overview

SCM-RAA is a structural classification framework designed to evaluate relationships between variables.

It separates:

- **Foreground** → robust structural signal
- **Midground** → weak or diffuse structure
- **Background** → noise / no structure

---

## Pipeline

1. CRTT (threshold detection via AIC)
2. Regime Signature:
   - N
   - delta_aic
   - delta_aic_per_n
   - iqr_frac
   - strong_rate
   - weak_rate
   - failure_rate
3. RAA classification
4. Bootstrap stability
5. Decision Layer

---

## Decision Layer Criteria

foreground_confirmed:
- strong_rate ≥ 0.70
- failure_rate ≤ 0.10

background_confirmed:
- failure_rate ≥ 0.70

midground_candidate:
- all other cases

---

## Clean validation table

`scm_raa_v2_6_experimental_report_table_CLEAN.csv`

---

## Results

| Dataset | Decision |
|---------|---------|
| SP500 | foreground_confirmed |
| ENERGY_SPAIN | midground_candidate |
| SPARC | midground_candidate |
| MOJAVE | midground_candidate |
| ECONOMY | midground_candidate |
| YANG | midground_candidate |
| SYNTH_H0 | background_confirmed |
| SYNTH_LINEAR | background_confirmed |
| LITTLE_THINGS | background_confirmed |

---

## What this framework does NOT do

- does not infer causality
- does not replace domain-specific analysis
- does not guarantee detection of all real signals
- does not treat weak signals as confirmed evidence

---

## Status

Experimental but reproducible.

Includes explicit false-positive control validated against synthetic noise.
