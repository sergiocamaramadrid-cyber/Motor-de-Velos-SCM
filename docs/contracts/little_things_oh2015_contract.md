# LITTLE THINGS Oh+2015 — contrato de ingestión para Framework SCM

## Fuente
Catálogo: J/AJ/149/180 (Oh et al. 2015)

## Archivos requeridos
- data/LITTLE_THINGS_Oh2015/ReadMe
- data/LITTLE_THINGS_Oh2015/table1.dat
- data/LITTLE_THINGS_Oh2015/table2.dat
- data/LITTLE_THINGS_Oh2015/rotdmbar.dat

## Objetivo
Construir dos salidas intermedias reproducibles:

1. `results/LITTLE_THINGS_Oh2015/little_things_galaxy_table.csv`
2. `results/LITTLE_THINGS_Oh2015/little_things_rotcurves.csv`

## Clave primaria
- `galaxy`

## Tabla por galaxia (`little_things_galaxy_table.csv`)
Columnas mínimas:

- `galaxy`
- `dist_mpc`
- `vsys_kms`
- `vsys_err_kms`
- `pa_deg`
- `pa_err_deg`
- `incl_deg`
- `incl_err_deg`
- `vmag_abs`
- `oh_12log`
- `oh_err`
- `logsfr_ha`
- `logsfr_ha_err`
- `logsfr_fuv`
- `logsfr_fuv_err`
- `rmax_kpc`
- `r03_kpc`
- `v_rmax_kms`
- `viso_rmax_kms`
- `rmax_over_hibeam`
- `z0_kpc`
- `c_nfw`
- `c_nfw_err`
- `c_m07`
- `v200_kms`
- `v200_err_kms`
- `v200m07_kms`
- `v200m07_err_kms`
- `rc_kpc`
- `rc_err_kpc`
- `rho0_1e3_msun_pc3`
- `rho0_err_1e3_msun_pc3`
- `alphamin`
- `alphamin_err`
- `alphamin_flag`
- `alpha36`
- `alpha36_err`
- `alpha36_flag`
- `mgas_1e7_msun`
- `mstar_k_1e7_msun`
- `mstar_sed_1e7_msun`
- `logmdyn`
- `logm200`

## Tabla de curvas (`little_things_rotcurves.csv`)
Columnas mínimas:

- `galaxy`
- `data_type`
- `r03_kpc`
- `v03_kms`
- `r_scaled`
- `v_scaled`
- `ev_scaled`

## Reglas
- `galaxy` debe conservar el identificador exacto del catálogo.
- `data_type` debe filtrarse a `Data` para observaciones reales.
- No mezclar `Model` con observaciones en el CSV principal de curvas.
- Valores faltantes deben guardarse como NaN.
- No renombrar galaxias.
- No alterar unidades.

## Uso esperado en el Framework
- `little_things_galaxy_table.csv`: controles físicos y metadatos.
- `little_things_rotcurves.csv`: base para estimar observables SCM/F3.
