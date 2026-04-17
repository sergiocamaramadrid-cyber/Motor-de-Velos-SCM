# data/processed

This folder contains processed datasets generated from SPARC raw data and
other inputs. All files here are reproducible outputs from scripts in `/scripts`.

**These files are NOT versioned.** Run the pipeline to regenerate them.

## Expected contents

```
data/processed/
  sparc_combined_radial.csv    # Merged radial profiles for all SPARC galaxies
  sparc_175_master.csv         # Master catalog with derived quantities (175 galaxies)
  galaxy_catalog_env.csv       # Galaxy catalog with environment proxy (N~79)
```

## How to regenerate

```bash
# Step 1 – compute slope tails from SPARC rotation curves
python scripts/compute_slope_tail.py

# Step 2 – build master catalog with environment proxy
python scripts/build_master_catalog.py

# Or run the full pipeline in one command:
python scripts/run_full_pipeline.py
```

See `/scripts/README.md` (if present) or the main `README.md` for full
pipeline documentation.
