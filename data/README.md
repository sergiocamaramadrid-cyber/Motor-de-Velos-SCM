# Datos SPARC

Los datos del catálogo SPARC (Spitzer Photometry & Accurate Rotation Curves,
Lelli et al. 2016) no se incluyen en este repositorio por razones de tamaño y
licencia.

## Descarga

1. Visita la página oficial del catálogo SPARC:
   <http://astroweb.cwru.edu/SPARC/>

2. Coloca en `data/SPARC/metadata/` las tablas:
   - `SPARC_Lelli2016c.mrt`
   - `CDR_Lelli2016b.mrt`
   - `BTFR_Lelli2019.mrt`
   - `MassModels_Lelli2016c.mrt`

3. Descarga los archivos de curvas de rotación individuales (`*_rotmod.dat`)
   y colócalos en `data/SPARC/rotmod/` (ruta preferida). También se acepta
   `data/SPARC/raw/` por compatibilidad con estructuras históricas.

## Comprobación rápida antes de descargar

Antes de descargar nada, verifica si los datos ya existen en tu clon local:

```bash
find . -iname "*sparc*"
find . -name "*rotmod.dat"
find . -name "*.csv"
find . -iname "*ddo*"
```

Si `find . -name "*rotmod.dat"` devuelve ~175 archivos, SPARC ya está disponible
y puedes reutilizarlo directamente.

Ejemplo para lanzar el test sin reconstrucción:

```bash
python scripts/run_big_sparc_veil_test.py --catalog data/SPARC/sparc_full.csv --out results
```

Si no aparecen `*_rotmod.dat`, reconstruye el catálogo completo con:

```bash
python scripts/build_sparc_full_catalog.py
```

## Troubleshooting de enlaces (404 / Not Found)

Si un enlace devuelve `404`, suele deberse a que:

- es una URL privada o temporal,
- requiere sesión iniciada en GitHub,
- o la ruta (por ejemplo `/tasks/...`) no es pública.

En ese caso, comparte una de estas tres opciones para desbloquear la revisión:

1. una captura de pantalla,
2. el texto exacto del error,
3. o el enlace público correcto al commit, PR, issue o archivo.

## Estructura esperada

```
data/SPARC/
├── metadata/
│   ├── SPARC_Lelli2016c.mrt
│   ├── CDR_Lelli2016b.mrt
│   ├── BTFR_Lelli2019.mrt
│   └── MassModels_Lelli2016c.mrt
└── rotmod/
    ├── NGC0300_rotmod.dat
    ├── NGC0891_rotmod.dat
    └── ...
```

## Referencia

Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016).
*SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry
and Accurate Rotation Curves.*
AJ, 152, 157.  doi:10.3847/0004-6256/152/6/157
