# Datos SPARC

Los datos del catálogo SPARC (Spitzer Photometry & Accurate Rotation Curves,
Lelli et al. 2016) no se incluyen en este repositorio por razones de tamaño y
licencia, pero se pueden descargar automáticamente con el script incluido.

## Descarga automática (recomendada)

Ejecuta el script desde la raíz del repositorio:

```bash
python scripts/download_sparc_data.py --out data/SPARC
```

El script descarga:
- **Tabla de galaxias** (`SPARC_Lelli2016c.mrt` → `SPARC_Lelli2016c.csv`)
  con fotometría 3.6 µm, distancias, inclinaciones y masas gaseosas de
  175 galaxias.
- **Curvas de rotación** (`*_rotmod.dat`) con velocidad observada y
  contribuciones bariónicos (estrellas + gas), en `data/SPARC/raw/`.

### Fuentes disponibles

| Fuente | URL | Descripción |
|--------|-----|-------------|
| CWRU (primaria) | <https://astroweb.cwru.edu/SPARC/> | Servidor oficial de la universidad |
| Zenodo (respaldo) | <https://doi.org/10.5281/zenodo.16284118> | Archivo de largo plazo |

El script intenta CWRU primero y cae automáticamente a Zenodo si el servidor
principal no está disponible. Para forzar una fuente concreta:

```bash
# Solo Zenodo
python scripts/download_sparc_data.py --out data/SPARC --source zenodo

# Solo CWRU
python scripts/download_sparc_data.py --out data/SPARC --source cwru
```

## Descarga manual

1. Visita la página oficial: <http://astroweb.cwru.edu/SPARC/>
2. Descarga `SPARC_Lelli2016c.mrt` y colócalo en esta carpeta.
3. Descarga los archivos `*_rotmod.dat` (o `Rotmod_LTG.zip`) y descomprime
   en `data/SPARC/raw/`.

## Estructura esperada

```
data/SPARC/
├── SPARC_Lelli2016c.csv      ← tabla principal de galaxias (generada por el script)
├── SPARC_Lelli2016c.mrt      ← archivo original CDS/MRT (descargado)
└── raw/
    ├── NGC0300_rotmod.dat
    ├── NGC0891_rotmod.dat
    └── ... (175 archivos)
```

## Referencia

Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016).
*SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry
and Accurate Rotation Curves.*
AJ, 152, 157. doi:[10.3847/0004-6256/152/6/157](https://doi.org/10.3847/0004-6256/152/6/157)
