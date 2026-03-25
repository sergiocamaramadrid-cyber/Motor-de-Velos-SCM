#!/usr/bin/env python3
"""
scripts/generate_sparc_radial_profiles.py
==========================================
Generate ``data/sparc_full_radial.csv`` — per-galaxy radial ΔF₃ profiles
needed as the primary input to ``run_h3_experiment.py``.

For every SPARC galaxy the script:

1. Loads the rotation-curve file (``<sparc_dir>/<Galaxy>_rotmod.dat`` or in
   ``<sparc_dir>/raw/``).
2. Fits the best-fit disk mass-to-light ratio ``upsilon_disk`` using the SCM
   pipeline (or uses the catalogue value if a pre-computed results CSV is
   supplied via ``--results-csv``).
3. Computes per-radial-point baryonic and observed accelerations:

      g_bar(r) = V_bar(r)² / r   (with upsilon_disk applied to disk)
      g_obs(r) = V_obs(r)² / r

4. Derives:

      ΔF₃(r) = log₁₀(g_obs(r)) − log₁₀(g_bar(r))

5. Writes one row per valid (galaxy, r) point.

Output CSV columns
------------------
galaxy, r, delta_F3

Usage
-----
::

    python scripts/generate_sparc_radial_profiles.py \\
        --sparc-dir data/SPARC \\
        --out data/sparc_full_radial.csv

With a pre-computed SCM results CSV (skips re-fitting upsilon_disk)::

    python scripts/generate_sparc_radial_profiles.py \\
        --sparc-dir data/SPARC \\
        --results-csv results/per_galaxy_summary.csv \\
        --out data/sparc_full_radial.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so this script can be run from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.scm_analysis import (
    load_galaxy_table,
    load_rotation_curve,
    fit_galaxy,
    _CONV,
    _MIN_RADIUS_KPC,
)
from src.scm_models import v_baryonic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_delta_f3(rc: pd.DataFrame, upsilon_disk: float) -> pd.DataFrame:
    """Return a DataFrame with columns ``r`` and ``delta_F3`` for one galaxy.

    Only radial points where both g_bar > 0 and g_obs > 0 are included.
    """
    r_arr = rc['r'].values
    v_obs_arr = rc['v_obs'].values
    vb_arr = v_baryonic(
        r_arr,
        rc['v_gas'].values,
        rc['v_disk'].values,
        rc['v_bul'].values,
        upsilon_disk=upsilon_disk,
        upsilon_bul=0.7,
    )

    safe_r = np.maximum(r_arr, _MIN_RADIUS_KPC)
    g_bar = vb_arr ** 2 / safe_r * _CONV   # m/s²
    g_obs = v_obs_arr ** 2 / safe_r * _CONV  # m/s²

    valid = (g_bar > 0) & (g_obs > 0)
    if not np.any(valid):
        return pd.DataFrame(columns=['r', 'delta_F3'])

    rows = pd.DataFrame({
        'r': r_arr[valid],
        'delta_F3': np.log10(g_obs[valid]) - np.log10(g_bar[valid]),
    })
    return rows


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate_radial_profiles(
    sparc_dir: str | Path,
    out_path: str | Path,
    results_csv: str | Path | None = None,
    a0: float = 1.2e-10,
    verbose: bool = True,
) -> pd.DataFrame:
    """Generate ``sparc_full_radial.csv`` from SPARC rotation-curve files.

    Parameters
    ----------
    sparc_dir : str or Path
        Root directory containing SPARC data.
    out_path : str or Path
        Destination CSV path.
    results_csv : str or Path or None
        Optional per-galaxy SCM results CSV containing an ``upsilon_disk``
        column.  If provided the fitting step is skipped for matched galaxies.
    a0 : float
        Characteristic acceleration used when fitting upsilon_disk.
    verbose : bool
        Print per-galaxy progress.

    Returns
    -------
    pd.DataFrame
        The generated radial profiles table.
    """
    sparc_dir = Path(sparc_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    galaxy_table = load_galaxy_table(sparc_dir)
    galaxy_names = galaxy_table['Galaxy'].tolist()

    # Pre-computed upsilon_disk look-up (optional, avoids re-fitting)
    ud_lookup: dict[str, float] = {}
    if results_csv is not None:
        res = pd.read_csv(results_csv)
        if 'upsilon_disk' in res.columns and 'galaxy' in res.columns:
            ud_lookup = dict(zip(res['galaxy'], res['upsilon_disk']))
        elif 'upsilon_disk' in res.columns and 'Galaxy' in res.columns:
            ud_lookup = dict(zip(res['Galaxy'], res['upsilon_disk']))

    all_rows: list[pd.DataFrame] = []
    n_ok = 0
    n_skip = 0

    for name in galaxy_names:
        try:
            rc = load_rotation_curve(sparc_dir, name)
        except FileNotFoundError:
            if verbose:
                print(f'  [skip] {name}: rotation curve not found', file=sys.stderr)
            n_skip += 1
            continue

        # Obtain upsilon_disk
        if name in ud_lookup:
            upsilon_disk = float(ud_lookup[name])
        else:
            fit = fit_galaxy(rc, a0=a0)
            upsilon_disk = fit['upsilon_disk']

        rows = _compute_delta_f3(rc, upsilon_disk)
        if rows.empty:
            if verbose:
                print(f'  [skip] {name}: no valid radial points', file=sys.stderr)
            n_skip += 1
            continue

        rows.insert(0, 'galaxy', name)
        all_rows.append(rows)
        n_ok += 1

        if verbose:
            print(f'  {name}: {len(rows)} radial points, '
                  f'delta_F3 range [{rows["delta_F3"].min():.3f}, '
                  f'{rows["delta_F3"].max():.3f}]')

    if not all_rows:
        raise RuntimeError(
            'No radial profiles could be generated.  Check that '
            f'{sparc_dir} contains valid SPARC rotation-curve files.'
        )

    profiles_df = pd.concat(all_rows, ignore_index=True)
    profiles_df.to_csv(out_path, index=False)

    print(f'\nGeneradas {len(profiles_df)} filas para {n_ok} galaxias '
          f'({n_skip} omitidas).')
    print(f'CSV guardado en: {out_path}')

    return profiles_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Genera data/sparc_full_radial.csv con perfiles ΔF₃ radiales '
                    'para cada galaxia SPARC.',
    )
    parser.add_argument(
        '--sparc-dir',
        default='data/SPARC',
        help='Directorio raíz de datos SPARC (default: data/SPARC)',
    )
    parser.add_argument(
        '--out',
        default='data/sparc_full_radial.csv',
        help='Ruta del CSV de salida (default: data/sparc_full_radial.csv)',
    )
    parser.add_argument(
        '--results-csv',
        default=None,
        help='CSV opcional con upsilon_disk pre-calculados (ahorra tiempo de ajuste)',
    )
    parser.add_argument(
        '--a0',
        type=float,
        default=1.2e-10,
        help='Aceleración característica a₀ en m/s² (default: 1.2e-10)',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suprimir progreso por galaxia',
    )
    args = parser.parse_args()

    generate_radial_profiles(
        sparc_dir=args.sparc_dir,
        out_path=args.out,
        results_csv=args.results_csv,
        a0=args.a0,
        verbose=not args.quiet,
    )
