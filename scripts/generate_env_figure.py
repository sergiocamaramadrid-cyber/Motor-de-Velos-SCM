"""#!/usr/bin/env python3
"""

#!/usr/bin/env python3
"""
Generate the environmental correlation figure (δ_mass vs rho_lag1) from
available results / per-galaxy catalogs. Saves PDF to
docs/paper1/figures/figure_env_correlation.pdf when successful.

Behavior:
- Searches for candidate tables in results/ and top-level CSV/Parquet files.
- Accepts several column-name heuristics for delta_mass and rho_lag1.
- If found, computes Spearman rho and p-value and plots scatter + linear fit
  with matplotlib (no seaborn), white background, clear labels.
- Exits non-zero if no suitable source table is found; writes a diagnostics
  file docs/paper1/figures/MISSING_SOURCE.txt listing attempted files.

This script is intended to be run in CI (Actions) or locally.
"""
from pathlib import Path
import re
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path("docs/paper1/figures/figure_env_correlation.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
DIAG = OUT.parent / "MISSING_SOURCE.txt"

COLUMN_X_CAND = [r"delta_mass", r"delta.*mass", r"deltamass", r"delta", r"\bddelta\b"]
COLUMN_Y_CAND = [r"rho_lag1", r"rho.*lag", r"lag_1", r"lag1", r"persistence", r"autocorr", r"rho"]

# Search candidate files
search_paths = list(Path('.').rglob('results/**/*.csv')) + list(Path('.').rglob('results/*.csv')) + list(Path('.').glob('*.csv')) + list(Path('.').rglob('*.parquet'))
tried = []

def find_columns(df):
    cols_lower = {c.lower(): c for c in df.columns}
    # exact-match priority
    if 'delta_mass' in cols_lower and 'rho_lag1' in cols_lower:
        return cols_lower['delta_mass'], cols_lower['rho_lag1']
    # heuristics
    xcol = None
    ycol = None
    for patt in COLUMN_X_CAND:
        for c in df.columns:
            if re.search(patt, c, re.I):
                xcol = c
                break
        if xcol:
            break
    for patt in COLUMN_Y_CAND:
        for c in df.columns:
            if re.search(patt, c, re.I):
                ycol = c
                break
        if ycol:
            break
    if xcol and ycol:
        return xcol, ycol
    return None


def try_load(path: Path):
    try:
        if path.suffix.lower() in ['.csv']:
            df = pd.read_csv(path)
        elif path.suffix.lower() in ['.parquet', '.pq']:
            df = pd.read_parquet(path)
        else:
            return None
    except Exception as e:
        tried.append(f"FAILED_READ: {path} -> {e}")
        return None
    cols = find_columns(df)
    if cols is None:
        tried.append(f"NO_MATCH: {path}")
        return None
    xcol, ycol = cols
    sub = df[[xcol, ycol]].dropna()
    if len(sub) < 10:
        tried.append(f"TOO_SMALL (<10 rows): {path} (found cols {xcol},{ycol}, n={len(sub)})")
        return None
    return sub, path

found = None
for p in search_paths:
    res = try_load(p)
    if res is not None:
        found = res
        break

if found is None:
    # Fallback: try common output names from pipeline
    pipeline_candidates = [Path('results/per_galaxy_summary.csv'), Path('results/f3_catalog_real.csv'), Path('results/f3_beta_catalog.parquet'), Path('results/f3_catalog_sparc_from_contract.parquet')]
    for p in pipeline_candidates:
        if p.exists():
            res = try_load(p)
            if res is not None:
                found = res
                break

if found is None:
    DIAG.write_text('\n'.join(tried) + '\n\nNo suitable table with delta_mass and rho_lag1 found.\nPlease provide a table with columns (delta_mass, rho_lag1) or run the pipeline that produces them.\n', encoding='utf-8')
    print(f"No suitable source found. Wrote diagnostics to {DIAG}")
    sys.exit(2)

sub, src = found
x = pd.to_numeric(sub.iloc[:,0], errors='coerce').dropna().to_numpy()
y = pd.to_numeric(sub.iloc[:,1], errors='coerce').dropna().to_numpy()
# align lengths (keep only paired indices)
min_n = min(len(x), len(y))
if len(x) != len(y):
    # attempt to align by index if sub has index preserving
    x = x[:min_n]
    y = y[:min_n]

# compute Spearman
from scipy.stats import spearmanr
rho, pval = spearmanr(x, y)

# Plot
plt.figure(figsize=(6,5), facecolor='w')
ax = plt.gca()
ax.set_facecolor('white')
plt.scatter(x, y, s=25, alpha=0.75, color='black', edgecolors='none')
# linear fit (ordinary polyfit)
coef = np.polyfit(x, y, 1)
fit = np.poly1d(coef)
x_fit = np.linspace(np.min(x), np.max(x), 200)
plt.plot(x_fit, fit(x_fit), linewidth=2, color='C1')
plt.xlabel(r'$\delta_{\rm mass}$')
plt.ylabel(r'$\rho_{\rm lag1}$')
plt.title('Environmental correlation in SPARC outer disks')
text = f'$\\rho$ = {rho:.2f}, p = {pval:.1e}'
plt.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment='top', fontsize=10)
plt.tight_layout()
plt.savefig(OUT, format='pdf', dpi=300)
plt.close()
print(f"Wrote figure to {OUT} from source {src} (n={len(x)})")
# exit zero
sys.exit(0)