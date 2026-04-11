# Catálogo de Grupos IATE FoF+Halo (FINAL_Group.dat)

Catálogo de grupos de galaxias construido con el método **Friends-of-Friends
(FoF) + Halo** sobre el SDSS, distribuido por el
**Instituto de Astronomía Teórica y Experimental (IATE–CONICET, UNC)**.

## Descarga

Ejecutar el script de descarga incluido en el repositorio:

```bash
python scripts/download_iate_group_catalog.py
```

Esto descarga el archivo original `FINAL_Group.dat` desde el servidor
del IATE y lo guarda como `data/iate/iate_group_catalog.csv`.

Parámetro opcional:

```bash
python scripts/download_iate_group_catalog.py --out data/iate/iate_group_catalog.csv
```

## Fuente

| Campo        | Detalle                                                                 |
|------------- |-------------------------------------------------------------------------|
| URL          | https://catalogs.iate.conicet.unc.edu.ar/fofandhalo/FINAL_Group.dat   |
| Formato      | ASCII whitespace-separated (`.dat`), con encabezado en líneas `#`       |
| Referencia   | Rodríguez, F. & Merchán, M. (2020), *A&A*, **636**, A61                |

## Columnas esperadas

El catálogo estándar de la publicación contiene las siguientes columnas
(cuando el archivo incluye encabezado; de lo contrario se aplican estos
nombres como fallback):

| Columna        | Descripción                                             | Unidad   |
|----------------|---------------------------------------------------------|----------|
| `GroupID`      | Identificador único del grupo                           | —        |
| `RA_deg`       | Ascensión recta del centro del grupo                    | grados   |
| `Dec_deg`      | Declinación del centro del grupo                        | grados   |
| `z`            | Redshift espectroscópico del grupo                      | —        |
| `N_members`    | Número de galaxias miembro                              | —        |
| `sigma_v_kms`  | Dispersión de velocidades                               | km/s     |
| `log_Mh_Msun`  | Logaritmo de la masa del halo (abundancia matching)     | M☉       |
| `R200_Mpc`     | Radio virial R₂₀₀                                      | Mpc      |

> **Nota:** Si el archivo FINAL_Group.dat incluye una línea de encabezado
> con `#`, el parser la detecta automáticamente y usa los nombres de columna
> originales del archivo.

## Referencia bibliográfica

```
@ARTICLE{Rodriguez2020,
  author  = {Rodríguez, Fernanda and Merchán, Manuel},
  title   = {Galaxy groups in the SDSS DR12},
  journal = {A\&A},
  year    = {2020},
  volume  = {636},
  pages   = {A61},
  doi     = {10.1051/0004-6361/201936568}
}
```
