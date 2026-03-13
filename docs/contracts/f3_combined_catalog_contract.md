# F3 Combined Catalog Contract

Archivo generado:

results/combined/f3_combined_catalog.csv

## Columnas mínimas

galaxy
source_catalog
framework_stage
science_role

dist_mpc
incl_deg

rmax_kpc
r03_kpc
v_rmax_kms

mgas_1e7_msun
mstar_proxy_1e7_msun
logmdyn

alphamin
rotcurve_available

f3_scm
delta_f3
tail_slope
n_tail_points
tail_rmin
fit_method
fit_ok
fit_ok_reason
quality_flag

## Definiciones oficiales

- `f3_scm`: pendiente externa `dlog(V)/dlog(R)` en la cola de la curva
- `delta_f3 = f3_scm - 0.5`
- `tail_rmin`: umbral mínimo en radio escalado para definir la cola externa
- `fit_ok`: indica si el ajuste externo es utilizable
- `quality_flag`: etiqueta resumida de calidad del cálculo

## Valores esperados de quality_flag

- `ok`
- `minimal_tail`
- `restricted_tail`
- `no_valid_tail_fit`
- `no_rotcurve_data`
