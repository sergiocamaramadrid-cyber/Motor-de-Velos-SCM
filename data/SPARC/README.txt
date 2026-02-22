# SPARC Rotation Curve Data (Sample)

This directory contains sample rotation curve data files in the SPARC/Iorio format.

## File Format

Each `.txt` file contains whitespace-separated columns:

| Column | Name   | Units   | Description                        |
|--------|--------|---------|------------------------------------|
| 0      | R      | kpc     | Galactocentric radius              |
| 1      | Vobs   | km/s    | Observed rotation velocity         |
| 2      | errV   | km/s    | Uncertainty on Vobs                |
| 3      | Vgas   | km/s    | Gas rotation contribution          |
| 4      | Vdisk  | km/s    | Stellar disk contribution          |
| 5      | Vbul   | km/s    | Stellar bulge contribution         |

Lines beginning with `#` are comments and are ignored.

## Sample Galaxies

The following sample galaxies are included for testing:

- DDO154, IC2574, NGC0024, NGC0055, NGC0300
- NGC0891, NGC2403, NGC3198, NGC6503, NGC7793
- UGC02259, UGC05764

## Real SPARC Data

The full SPARC dataset (175 galaxies) is available at:
  http://astroweb.cwru.edu/SPARC/

Download `Rotmod_LTG.zip` and place the extracted `.txt` files in this directory.

## Running the Pipeline

    python -m src.scm_analysis --data-dir data/SPARC --out results/
    python -m src.sensitivity  --data-dir data/SPARC --out results/sensitivity/
