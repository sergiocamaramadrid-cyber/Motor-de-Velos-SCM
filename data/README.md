# Archivos de datos

## mw_cepheids.csv

Muestra de Cefeidas de la Vía Láctea usada en el pipeline principal SCM.
Columnas: `R_kpc`, `Vc_kms`, `e_Vc`, `source`.
Generado a partir de Gaia DR3 + compilaciones de la literatura.
Consumido por `scripts/mw_delta_f3.py`.

## mw_gaia_master_AB.csv (no incluido)

Archivo generado localmente en Colab a partir de consultas directas a Gaia DR3,
usado únicamente para un test exploratorio de anisotropía hemisférica (hemisferio
A vs. B en coordenadas galácticas).  **No forma parte del pipeline principal ni
del conjunto de datos del paper.**  Se excluye del repositorio intencionalmente
(véase `.gitignore`); puede regenerarse desde Gaia Archive si se necesita repetir
el test.

---

# Datos SPARC

Los datos del catálogo SPARC (Spitzer Photometry & Accurate Rotation Curves,
Lelli et al. 2016) no se incluyen en este repositorio por razones de tamaño y
licencia.

## Descarga

1. Visita la página oficial del catálogo SPARC:
   <http://astroweb.cwru.edu/SPARC/>

2. Descarga el archivo de la tabla de galaxias `SPARC_Lelli2016c.mrt` (o la
   versión `.csv` si está disponible) y colócalo en esta carpeta.

3. Descarga los archivos de curvas de rotación individuales (`*_rotmod.dat`)
   y colócalos en `data/SPARC/raw/`.

## Estructura esperada

```
data/SPARC/
├── SPARC_Lelli2016c.csv      ← tabla principal de galaxias
└── raw/
    ├── NGC0300_rotmod.dat
    ├── NGC0891_rotmod.dat
    └── ...
```

## Referencia

Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016).
*SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry
and Accurate Rotation Curves.*
AJ, 152, 157.  doi:10.3847/0004-6256/152/6/157
