# SCM — Motor de Velos

## v2.7 — Structural Signal Classification (Regime, Noise, Bias & Mediation)

The SCM Framework is a reproducible empirical system designed to detect, classify, and validate structural signals in complex datasets.

It does not assume signals — it tests whether they exist.

---

## Core Results

### SPARC (N=79)

- Regime-dependent signal confirmed
- ρ ≈ -0.65 (high-mass regime)
- p ≈ 1.1 × 10⁻⁴
- R² ≈ 0.33
- Mass threshold: logM ≈ 9.8–10.0
- Classification: `regime_dependent`

### LITTLE THINGS (N=25)

- Mixed / fragmented structure
- Irregular galaxies show non-global behavior
- Classification: `regime_fragmented`

### NASA Exoplanet False Positives (KOI, N≈4500)

- Initial signal disappears after control (SNR, duration)
- Classification: `confirm_noise`

### Galaxy Clusters (N=1959)

- Strong global correlation detected (M500–L500)
- Residual fully explained by redshift
- Classification: `derived_quantity_bias`

### Nebulae (control test)

- Signal exists only through mediated channel (OH → Te → flux)
- Fully recovered when isolating causal chain
- Classification: `mediated_signal`

---

## SCM Classification System (v2.7)

The framework now distinguishes six structural regimes:

| Class | Description |
|---|---|
| `global_structured` | Strong, uniform signal across full state space |
| `regime_dependent` | Signal exists only above a critical mass/energy threshold |
| `regime_fragmented` | Mixed or irregular structure, non-global behavior |
| `confirm_noise` | Apparent signal disappears after proper controls |
| `derived_bias` | Signal is an artefact of dataset construction |
| `mediated_signal` | Signal exists only through a confounding causal chain |

**Key insight:** SCM does not search for strong signals in clean data. It classifies structure in noisy, rejected, or ambiguous datasets — separating real structure from false positives, construction biases, and mediated dependencies.

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

The framework was stress-tested across five independent datasets:

| Dataset | N | Result |
|---|---|---|
| SPARC | 79 | `regime_dependent` — ρ ≈ -0.65, p ≈ 1.1×10⁻⁴ |
| LITTLE THINGS | 25 | `regime_fragmented` — irregular, non-global |
| NASA KOI false positives | ~4500 | `confirm_noise` — signal vanishes under control |
| Galaxy Clusters | 1959 | `derived_bias` — residual explained by redshift |
| Nebulae | control | `mediated_signal` — causal chain recovered |

This demonstrates:

- no false positives
- correct signal detection
- correct rejection of noise and construction biases
- recovery of mediated causal structure

> Nebulae test is used as method validation, not as a physical claim. Cluster analysis reveals dataset construction bias, not a new astrophysical relation.

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