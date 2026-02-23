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
