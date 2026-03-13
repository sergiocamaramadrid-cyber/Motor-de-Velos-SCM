# Framework SCM – Master Catalog Contract

Este catálogo unifica todas las muestras observacionales usadas por el Framework.

Actualmente incluye:

- SPARC (Lelli+2016)
- LITTLE THINGS (Oh+2015)

Objetivo: construir una tabla homogénea para análisis y cálculo de observables SCM.

---

## Archivo generado

results/combined/framework_master_catalog.csv

---

## Clave primaria

galaxy

---

## Columnas obligatorias

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

---

## Definiciones

source_catalog

SPARC
LITTLE_THINGS

framework_stage

early_validation

science_role

main_disk_sample
dwarf_validation_sample

rotcurve_available

True si existe curva de rotación utilizable para el Framework.

---

## Uso en el Framework

Este catálogo sirve como base para:

compute_f3_combined_catalog.py

que calculará:

F3_SCM
ΔF3
