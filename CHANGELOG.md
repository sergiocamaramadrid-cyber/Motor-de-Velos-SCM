# Changelog

All notable changes to this project are documented here.

---

## [v2.7.0] — 2026-04-29

### Added

- **Structural Signal Classification layer** — the framework now classifies the *structure* of signal, not only its presence:
  - `global_critical` — criticality is uniform across the full state space
  - `regime_dependent` — criticality is fragmented across mass/energy regimes
  - `transition` — system is near a structural boundary
  - `non_linear` — signal exists but lacks a linear critical structure
  - `none` — no detectable structure
- Regime-dependent signal classification pipeline
- Critical mass threshold detection (SPARC)
- Bulk vs outlier separation
- Residual-based validation pipeline
- NASA false positive control test

### Key Results (SPARC, N=79)

- No global environmental law
- Critical mass threshold at logM ≈ 9.8–10.0
- Strong environmental modulation above threshold
- Peak signal at logM ≈ 10.75:
  - ρ ≈ -0.65, p ≈ 1.1 × 10⁻⁴, β_env ≈ -0.061, R² ≈ 0.33
- No detectable correlation below threshold

### Cross-dataset validation

- SPARC → `regime_dependent` (fragmented criticality)
- YANG → `regime_dependent`
- MOJAVE → `global_critical`
- SP500 → `transition`
- ECONOMY → `non_linear`

### Validation (negative controls)

- NASA Exoplanet False Positives → CONFIRM_NOISE ✔
- SPARC outliers → no structure ✔
- SPARC bulk → structured regime-dependent signal ✔
- No false positives detected

### Changed

- `README.md` fully updated with v2.7 key results, classification table, validation summary, and DOI

### Notes

- v2.6 decision layer (`foreground_confirmed`, `midground_candidate`, `background_confirmed`) is preserved and remains fully valid
- v2.7 extends v2.6 — it does not replace it

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

