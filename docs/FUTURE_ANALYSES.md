# Future Analyses and Exploratory Proxies

This document describes exploratory analyses and robustness checks that extend
beyond the canonical results of the paper. These are provided for transparency
and reproducibility but are **not** part of the main claims.

---

## Yang et al. Environmental Proxy (`delta_mass_yang`)

**Script:** `scripts/crossmatch_yang_proxy.py`

**Purpose:**  
Cross-match SPARC galaxies against the Yang et al. (2007, 2012) SDSS group
catalog to derive an alternative large-scale-structure environmental proxy
(`delta_mass_yang`). This is an exploratory robustness check for the canonical
`delta_mass` proxy used in the paper.

**IMPORTANT — Scope and Limitations:**

- `delta_mass_yang` is **NOT** the canonical `delta_mass` used in any paper result.
- It is an independent robustness/sensitivity test only.
- The canonical `delta_mass` is derived from the local density field at 3 Mpc
  scale as described in `docs/paper1/methods_delta_mass.md`.
- Do **not** replace or rename `delta_mass` with `delta_mass_yang` in any
  pipeline or figure.

**Proxy derivation logic:**

1. If the Yang catalog contains a direct density-like column (`DELTA_3MPC`,
   `DELTA_5MPC`, `delta`, `local_density`), it is used directly as
   `delta_mass_yang`.
2. Otherwise, a pseudo-overdensity is computed from a group-multiplicity column
   (`NGROUP`, `MGROUP`, etc.) via:

   ```
   delta_mass_yang = (raw_proxy / median(raw_proxy)) − 1
   ```

   This normalised form is analogous in spirit to the canonical overdensity
   definition but is derived from group membership rather than a density field.

**Output columns** (`results/delta_mass_yang_sparc.csv`):

| Column | Description |
|--------|-------------|
| `galaxy` | SPARC galaxy name |
| `delta_mass_yang` | Environmental proxy (overdensity or normalised group value) |
| `<proxy_col>` | Raw value of the chosen Yang catalog column |
| `match_sep_arcsec` | Angular separation to matched Yang entry (arcsec) |
| `proxy_source` | Name of the Yang column used |
| `proxy_mode` | `density` or `group_proxy` |
| `match_radius_arcsec` | Search radius used (default 15 arcsec) |

**Usage:**

```bash
python scripts/crossmatch_yang_proxy.py
```

Ensure the input files exist at the configured paths before running:

- `data/SPARC/sparc_basic.csv` — SPARC catalog with `galaxy`, `ra`, `dec` columns
- `data/environment/yang_group_catalog.fits` — Yang group catalog (FITS or CSV)

**Column semantics reminder:**

- `delta_mass` — canonical LSS proxy (ρ_local/⟨ρ⟩ − 1 at 3 Mpc); used in the paper
- `delta_mass_yang` — exploratory Yang-catalog proxy; robustness check only
- `delta_dyn` / `delta_g` — log10(g_obs) − log10(g_bar); dynamical quantity
- `F3` / `friction_slope` — deep-regime slope β from `generate_f3_catalog.py`

Never rename `delta_dyn` as `delta_mass`, and never substitute `delta_mass_yang`
for `delta_mass` in paper figures or tables.
