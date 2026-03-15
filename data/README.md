# SPARC data notes

## Origen de datos

- `data/sparc_175_master_sample.csv` es una muestra sintética/curada del catálogo maestro SPARC para pruebas reproducibles del bloque F3.
- La misma muestra se conserva en `results/SPARC/sparc_175_master_sample.csv` para compatibilidad con pipelines históricos.
- El catálogo maestro completo se genera con `scripts/build_sparc_master_catalog.py` en `data/sparc_175_master.csv`.

## Columnas principales

Columnas esperadas en el catálogo SPARC del framework:

- `logSigmaHI_out`
- `logMbar`
- `logRd`
- `F3`
- `f3_scm`
- `delta_f3`
- `fit_ok`
- `quality_flag`
- `beta`
- `beta_err`
- `reliable`
- `friction_slope`
- `velo_inerte_flag`

## Formato esperado

- CSV con encabezados en primera fila.
- Una fila por galaxia.
- Tipos numéricos para magnitudes físicas (`log*`, `F3`, `beta`, `beta_err`).
- Flags booleanos para `fit_ok`, `reliable`, `velo_inerte_flag`.
- `quality_flag` como etiqueta corta (`ok`, `warn`, `fail`).

Para el contrato formal de columnas y reglas, ver `docs/DATA_CONTRACT_SPARC.md`.
