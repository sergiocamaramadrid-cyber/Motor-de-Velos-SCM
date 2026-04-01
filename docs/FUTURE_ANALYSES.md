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

---

## Yang et al. Environmental Proxy (`delta_mass_yang`) — Uso práctico

**Script:** `scripts/crossmatch_yang_proxy.py`

### Preparación de archivos

1. **Catálogo de Yang**
   - Descarga una versión del catálogo de grupos de Yang en formato FITS o CSV
     (por ejemplo, desde VizieR o datos derivados de SDSS).
   - Asegúrate de que contiene columnas de coordenadas (RA, DEC) y al menos una
     columna de densidad (p.ej., `DELTA_3MPC`, `DELTA_5MPC`) o multiplicidad de
     grupo (`NGROUP`, `MGROUP`).
   - Colócalo en `data/environment/yang_group_catalog.fits` (o ajusta la ruta en
     el script).

2. **Catálogo básico de SPARC**
   - El script espera `data/SPARC/sparc_basic.csv` con las columnas `galaxy`,
     `ra`, `dec`.
   - Si tu archivo SPARC procesado (`sparc_processed.csv`) ya contiene esas
     columnas, puedes generarlo así:

     ```python
     import pandas as pd

     sparc = pd.read_csv("data/SPARC/sparc_processed.csv")
     required = ["galaxy", "ra", "dec"]
     missing = [c for c in required if c not in sparc.columns]
     if missing:
         raise ValueError(f"Missing required columns: {missing}")

     basic = sparc[required].drop_duplicates()
     basic.to_csv("data/SPARC/sparc_basic.csv", index=False)
     ```

### Ejecución

```bash
python scripts/crossmatch_yang_proxy.py
```

### Verificación de resultados

Una vez ejecutado, abre `results/delta_mass_yang_sparc.csv` y comprueba:

- **Número de matches** (filas del CSV o salida del script).
- **Mediana de `match_sep_arcsec`**: si es > 30 arcsec, el cruce puede no ser
  fiable o el radio es demasiado grande.
- **`proxy_source`**: columna del catálogo Yang utilizada.
- **`proxy_mode`**:
  - `density` → densidad directa
  - `group_proxy` → proxy derivado (exploratorio)

### Sanity check rápido

Verifica que:

- `delta_mass_yang` no sea constante ni mayoritariamente `NaN`.
- La distribución tenga valores positivos y negativos (sobredensidad /
  subdensidad).

### Nota importante

Este proxy (`delta_mass_yang`) es exploratorio y no sustituye al `delta_mass`
canónico utilizado en el paper. Su finalidad es servir como test de robustez
ambiental usando una fuente externa independiente.

---

## Yang et al. Environmental Proxy (`delta_mass_yang`) — Practical Use

**Script:** `scripts/crossmatch_yang_proxy.py`

### File preparation

1. **Yang catalog**
   - Download a version of the Yang group catalog in FITS or CSV format (for
     example, from VizieR or SDSS-derived resources).
   - Ensure that it contains coordinate columns (`RA`, `DEC`) and at least one
     density-like column (e.g. `DELTA_3MPC`, `DELTA_5MPC`) or group-multiplicity
     column (`NGROUP`, `MGROUP`).
   - Place it at `data/environment/yang_group_catalog.fits` (or adjust the path
     in the script).

2. **Basic SPARC catalog**
   - The script expects `data/SPARC/sparc_basic.csv` with columns `galaxy`,
     `ra`, `dec`.
   - If your processed SPARC table already contains these columns, you can
     generate it with:

     ```python
     import pandas as pd

     sparc = pd.read_csv("data/SPARC/sparc_processed.csv")
     required = ["galaxy", "ra", "dec"]
     missing = [c for c in required if c not in sparc.columns]
     if missing:
         raise ValueError(f"Missing required columns: {missing}")

     basic = sparc[required].drop_duplicates()
     basic.to_csv("data/SPARC/sparc_basic.csv", index=False)
     ```

### Execution

```bash
python scripts/crossmatch_yang_proxy.py
```

### Output verification

After execution, inspect `results/delta_mass_yang_sparc.csv` and verify:

- **Match count**: number of SPARC galaxies successfully cross-matched.
- **Median `match_sep_arcsec`**: if it is > 30 arcsec, the match may be
  unreliable or the search radius may be too large.
- **`proxy_source`**: which Yang catalog column was used.
- **`proxy_mode`**:
  - `density` → direct density-like quantity
  - `group_proxy` → exploratory proxy derived from group multiplicity or mass

### Quick sanity check

Confirm that:

- `delta_mass_yang` is not constant and not mostly `NaN`.
- Its distribution includes both positive and negative values (overdensity /
  underdensity behaviour).

### Important note

`delta_mass_yang` is an exploratory external robustness proxy. It is not
identical to the canonical `delta_mass` used in the paper and must not replace
it in the main manuscript, figures, or tables without explicit redefinition.
