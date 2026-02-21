# Data — Motor de Velos SCM

## Included synthetic galaxy

`GXY_D13.8_V144_SCM_01.csv`  
Synthetic rotation curve for a representative galaxy with:

| Property | Value |
|---|---|
| Distance | 13.8 Mpc |
| MOND asymptotic velocity | ~144 km/s |
| Disk scale length *h_d* | 9 kpc |
| True Υ_disk | 0.5 |
| True *V_ext* (universal term) | 30 km/s |
| Random seed | 20260211 |

**Columns**

| Column | Units | Description |
|---|---|---|
| `r` | kpc | Galactocentric radius |
| `Vobs` | km/s | Observed rotation velocity |
| `eVobs` | km/s | Measurement uncertainty |
| `Vdisk` | km/s | Stellar-disk contribution at Υ=1 (Freeman exponential disk) |
| `Vgas` | km/s | HI gas contribution (face value) |
| `Vbul` | km/s | Bulge contribution at Υ_bul=1 (0 for this galaxy) |

## SPARC dataset

The full SPARC rotation-curve database (175 galaxies) used in the paper is
available at <http://astroweb.cwru.edu/SPARC/> (Lelli, McGaugh & Schombert
2016, AJ, 152, 157).  To reproduce the full analysis, download the SPARC
files and place them in this directory before running:

```bash
python scm_sparc_full_analysis.py --data-dir data/ --outdir results/
```
