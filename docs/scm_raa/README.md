# SCM-RAA v3 (Experimental)

## Overview

SCM-RAA is a structural classification framework designed to detect and validate relationships between variables across different domains.

It separates:

- **Foreground** → robust structural signal
- **Midground** → weak or diffuse structure
- **Background** → noise / no structure

## Pipeline

1. CRTT (threshold detection via AIC)
2. Regime Signature (quantitative vector)
3. RAA (classification)
4. Sampling (bootstrap stability)
5. Decision Layer (final verdict)

## Architecture

### 1. CRTT

Detects if a structural transition exists (piecewise vs linear):

- ΔAIC
- Optimal threshold

### 2. Regime Signature

Quantitative vector of regime behaviour:

- N
- ΔAIC
- ΔAIC/N
- `iqr_frac` (threshold stability)

### 3. RAA

Converts metrics into:

- `status`: strong / weak / failure
- `layer`: foreground / midground / background

### 4. Sampling (bootstrap)

Evaluates real stability:

- `strong_rate`
- `weak_rate`
- `failure_rate`

### 5. Decision Layer

Avoids self-deception:

- `foreground_confirmed`
- `midground_candidate`
- `background_confirmed`

## Key Principle

> Weak signals are not considered evidence.  
> Only stable strong signals are confirmed.

## Validation

### Control H0 (noise)

| Dataset              | Result                 |
|----------------------|------------------------|
| SYNTH_LINEAR_NOISE   | background_confirmed   |
| SYNTH_H0_WHITE_NOISE | background_confirmed   |

✔ No false positives.

### Real data

| Dataset       | Result                |
|---------------|-----------------------|
| SP500         | foreground_confirmed  |
| MOJAVE        | midground_candidate   |
| SPARC         | midground_candidate   |
| YANG          | midground_candidate   |
| ENERGY_SPAIN  | midground_candidate   |
| ECONOMY       | midground_candidate   |
| LITTLE_THINGS | background_confirmed  |

The system correctly distinguishes:

- ✔ Strong signal → SP500
- ✔ Diffuse signal → SPARC / YANG / MOJAVE / ENERGY
- ✔ Noise → SYNTH / LITTLE THINGS

## Decision layer

RAA outputs are converted into information layers:

- `foreground` → main analysis
- `midground` → directed exploration
- `background` → control/reference

The framework does not discard data.  
It organizes datasets by information content.

## Limitations

- Designed for structural transitions (CRTT-based)
- Does not model continuous relations explicitly
- High `weak_rate` can appear in noise — mitigated by the decision layer
- No adaptive learning yet (passive memory only)

## Strengths

- Explicit control of false positives
- Domain-independent (astrophysics, economics, energy, …)
- Reproducible
- Interpretable (not a black box)
- Scalable

## Status

Experimental — reproducible and validated.
