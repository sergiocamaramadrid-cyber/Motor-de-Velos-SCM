# DATA CONTRACT — SPARC Master Catalog

Este contrato define las columnas mínimas necesarias para el bloque F3.

## Columnas obligatorias

1. `logSigmaHI_out`
2. `logMbar`
3. `logRd`
4. `F3`
5. `f3_scm`
6. `delta_f3`
7. `fit_ok`
8. `quality_flag`
9. `beta`
10. `beta_err`
11. `reliable`
12. `friction_slope`
13. `velo_inerte_flag`

## Reglas mínimas

- Sin columnas faltantes.
- Sin NaN en columnas obligatorias.
- `beta_err >= 0`.
- `fit_ok=False` no debe coexistir con `reliable=True`.
- `friction_slope` y `beta` deben ser consistentes dentro de tolerancia numérica.

## Archivos de referencia

- Muestra: `data/sparc_175_master_sample.csv`
- Copia de resultados: `results/SPARC/sparc_175_master_sample.csv`
- Maestro generado: `data/sparc_175_master.csv`
