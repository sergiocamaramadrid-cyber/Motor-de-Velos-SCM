# F3 Combined Catalog

Calcula observables del Framework SCM sobre el catálogo combinado.

## Ejecución

```bash
python scripts/compute_f3_combined_catalog.py
```

## Entradas

results/combined/framework_master_catalog.csv

results/LITTLE_THINGS_Oh2015/little_things_rotcurves.csv


## Salida

results/combined/f3_combined_catalog.csv


## Columnas clave

f3_scm

delta_f3

tail_slope

n_tail_points

tail_rmin

fit_method

fit_ok

fit_ok_reason

quality_flag


## Definición actual

delta_f3 = f3_scm - 0.5

cola externa definida por r_scaled >= 0.7 por defecto
