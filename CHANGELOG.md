# Changelog

All notable changes to this project are documented here.

---

## [v2.8.0] — 2026-04-30

### Added

- **Expanded Structural Signal Classification system** — six distinct classes:
  - `global_structured` — strong, uniform signal across full state space
  - `regime_dependent` — signal exists only above a critical mass/energy threshold
  - `regime_fragmented` — mixed or irregular structure, non-global behavior
  - `confirm_noise` — apparent signal disappears after proper controls
  - `derived_bias` — signal is an artefact of dataset construction
  - `mediated_signal` — signal exists only through a confounding causal chain
- Multi-dataset validation pipeline (5 independent datasets)
- LITTLE THINGS analysis (N=25 irregular galaxies)
- Galaxy Cluster analysis (N=1959, M500–L500 correlation)
- Nebulae mediated-signal control test
- Regime-dependent signal classification pipeline
- Critical mass threshold detection (SPARC)
- Bulk vs outlier separation
- Residual-based validation pipeline
- NASA false positive control test

### Key Results

**SPARC (N=79)**
- Regime-dependent signal confirmed
- ρ ≈ -0.65 (high-mass regime), p ≈ 1.1 × 10⁻⁴, R² ≈ 0.33
- Mass threshold: logM ≈ 9.8–10.0
- Classification: `regime_dependent`

**LITTLE THINGS (N=25)**
- Mixed / fragmented structure; irregular galaxies show non-global behavior
- Classification: `regime_fragmented`

**NASA Exoplanet False Positives (KOI, N≈4500)**
- Initial signal disappears after control (SNR, duration)
- Classification: `confirm_noise`

**Galaxy Clusters (N=1959)**
- Strong global M500–L500 correlation; residual fully explained by redshift
- Classification: `derived_quantity_bias`

**Nebulae (control)**
- Signal recovered only through mediated channel (OH → Te → flux)
- Classification: `mediated_signal`

### Validation summary

| Dataset | N | Classification |
|---|---|---|
| SPARC | 79 | `regime_dependent` |
| LITTLE THINGS | 25 | `regime_fragmented` |
| NASA KOI FP | ~4500 | `confirm_noise` |
| Galaxy Clusters | 1959 | `derived_bias` |
| Nebulae | control | `mediated_signal` |

- No false positives detected
- Correct signal detection and noise rejection confirmed

### Changed

- `README.md` updated: subtitle, full classification table, per-dataset results, validation table
- Classification taxonomy expanded from 5 classes to 6 (replacing earlier `global_critical`/`transition`/`non_linear` with new operational classes)

### Notes

- v2.6 decision layer (`foreground_confirmed`, `midground_candidate`, `background_confirmed`) is preserved and remains fully valid
- v2.8 extends v2.6 — it does not replace it
- Nebulae test is used as method validation, not as a physical claim
- Cluster analysis reveals dataset construction bias, not a new astrophysical relation

---

## [v2.6.0-experimental] — 2026-04-28

### Added

- SCM-RAA (Regime-Aware Analysis) experimental module
- CRTT (piecewise vs linear AIC comparison)
- Regime Signature quantitative vector
- Bootstrap stability analysis
- Decision layer with explicit false-positive control:
  - `foreground_confirmed` — `strong_rate ≥ 0.70` and `failure_rate ≤ 0.10`
  - `background_confirmed` — `failure_rate ≥ 0.70`
  - `midground_candidate` — all other cases
- Validated across 9 datasets (synthetic noise → `background_confirmed`, no false positives)
- Reproducible Colab notebook: `SCM_RAA_v2_6_experimental_Colab.ipynb`
- Independence note in README: *This module extends the SCM framework but is developed and validated independently.*

---

## Earlier versions

See commit history for prior development milestones.

