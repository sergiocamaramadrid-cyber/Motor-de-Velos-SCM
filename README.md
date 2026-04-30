# SCM — Motor de Velos

## v2.7 — Structural Signal Classification (Regime & Criticality)

The SCM Framework is a reproducible empirical system designed to detect, classify, and validate structural signals in complex datasets.

It does not assume signals — it tests whether they exist.

---

## Key Result (SPARC)

Analysis of 79 SPARC galaxies shows:

- No global environmental law
- Clear regime-dependent behavior
- Critical mass threshold at:

> logM ≈ 9.8–10.0

Above this threshold:

- Strong negative correlation between environment and outer slope

Best regime:

- logM ≈ 10.75
- ρ ≈ -0.65
- p ≈ 1.1 × 10⁻⁴
- β_env ≈ -0.061
- R² ≈ 0.33

Below threshold:

- No detectable correlation

---

## Interpretation

Environmental modulation is not universal.

It emerges only in a high-mass regime, indicating a transition in galaxy dynamics.

---

## Structural Signal Classification (v2.7)

The framework now classifies not only the presence of signal, but its structure:

| Class | Description |
|---|---|
| `global_critical` | Criticality is uniform across the full state space |
| `regime_dependent` | Criticality is fragmented across mass/energy regimes |
| `transition` | System is near a structural boundary |
| `non_linear` | Signal exists but lacks a linear critical structure |
| `none` | No detectable structure |

**Key insight:** Signal is not uniformly distributed in state space. Some systems concentrate criticality, while others fragment it across regimes.

Validated across: SPARC (`regime_dependent`), YANG (`regime_dependent`), MOJAVE (`global_critical`), SP500 (`transition`), ECONOMY (`non_linear`).

---

## v2.6 — SCM-RAA (preserved)

An experimental structural classification layer designed to evaluate relationships between variables.

This module extends the SCM framework but is developed and validated independently.

SCM-RAA classifies relationships into three levels:

- **Foreground** → robust structural signal
- **Midground** → weak or diffuse structure
- **Background** → noise / no detectable structure

Key features:

- CRTT (piecewise vs linear model comparison using AIC)
- Regime Signature (quantitative vector)
- Bootstrap stability analysis
- Decision layer with explicit false-positive control

> v2.7 is an evolution of v2.6, not a replacement. The v2.6 decision layer (`foreground_confirmed`, `midground_candidate`, `background_confirmed`) remains fully valid.

---

## Validation

The framework was stress-tested against:

- NASA Exoplanet False Positives → CONFIRM_NOISE
- SPARC Outliers → no structure
- SPARC Bulk → structured signal

This demonstrates:

- no false positives
- correct signal detection
- correct rejection of noise

---

## Core Principle

> The SCM does not search for signals.  
> It determines whether structure exists — and where.

---

## Outputs

All results are reproducible and stored in `/SCM_WORK/`, including:

- `scm_sparc_final_figure.png`
- `sparc_bulk_mass_threshold_scan.csv`
- `sparc_bulk_2d_grid.csv`
- `sparc_outliers.csv`
- `scm_nasa_fp_session.json`
- `SCM_v2_7_paper_results_summary.csv`

---

## Repository structure

- `notebooks/experimental/` → reproducible pipelines
- `data/processed/` → validated outputs
- `docs/scm_raa/` → technical documentation
- `scripts/` → analysis scripts
- `tests/` → unit and integration tests

---

## Installation

### Requirements

- Python 3.10 or later
- Dependencies: see `requirements.txt`

### Setup

```bash
git clone https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM.git
cd Motor-de-Velos-SCM
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Citation

> DOI: 10.5281/zenodo.19897353

See also `CITATION.cff` and the Zenodo archive.

---

## Status

Framework validated. Results reproducible. Ready for publication.

---

## License

Refer to the LICENSE file.

---

## Author

Sergio Cámara Madrid — Independent Researcher  
Repository: https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM