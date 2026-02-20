"""run_blind_test_final.py — Motor de Velos SCM blind test on LITTLE THINGS galaxies.

Applies the SCM Hinge model with fixed, pre-calibrated SPARC coefficients to
the LITTLE THINGS dwarf-galaxy rotation curves without any per-galaxy
parameter tuning (genuine blind test).

Usage
-----
    python scripts/run_blind_test_final.py

Results are printed to the terminal and appended to ``veredicto_final.txt``
in the current working directory.
"""

import datetime
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow importing this module from repository root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

# ==========================================
# 1. CONFIGURACIÓN Y COEFICIENTES (SPARC)
# ==========================================
# Cambia esta ruta a la carpeta donde tienes los .dat o .csv de LITTLE THINGS
DATA_DIR = "data/little_things/"

SCM_FIXED_PARAMS = {
    'a': -0.42,
    'b': 0.12,   # reserved: asymmetric-hinge slope correction (not used in v1)
    'd': 1.85,
    'logg0': -10.45,
    'C': 1.58
}

# ==========================================
# 2. LOADER ROBUSTO (Carga de datos reales)
# ==========================================
def load_lt_file(galaxy_id):
    """Load a LITTLE THINGS rotation-curve file (.dat or .csv).

    Accepts any whitespace- or comma-separated file; comment lines starting
    with ``#`` are ignored.  Column names are mapped to the canonical set used
    by the SCM calculation engine.

    Parameters
    ----------
    galaxy_id : str
        Galaxy identifier (filename without extension), e.g. ``"DDO154"``.

    Returns
    -------
    df : pd.DataFrame
        Validated DataFrame with columns ``r_kpc``, ``vobs``, ``evobs``,
        ``vgas``, ``vdisk`` and derived ``vbary``.

    Raises
    ------
    FileNotFoundError
        If neither ``<galaxy_id>.dat`` nor ``<galaxy_id>.csv`` is found in
        ``DATA_DIR``.
    """
    # Busca .dat o .csv
    path = os.path.join(DATA_DIR, f"{galaxy_id}.dat")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, f"{galaxy_id}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró {galaxy_id}.dat ni {galaxy_id}.csv en '{DATA_DIR}'"
        )

    # Lectura inteligente (detecta separadores automáticamente)
    df = pd.read_csv(path, sep=None, engine='python', comment='#')
    df.columns = [c.strip().lower() for c in df.columns]

    # Mapeo de columnas para LITTLE THINGS
    rename_map = {
        'rad': 'r_kpc', 'r': 'r_kpc',
        'vrot': 'vobs', 'vobs': 'vobs',
        'e_vrot': 'evobs', 'err': 'evobs', 'evobs': 'evobs',
        'vgas': 'vgas', 'vdisk': 'vdisk', 'vstar': 'vdisk'
    }
    df.rename(columns=rename_map, inplace=True)

    # Cálculo de vbary (Gas + Estrellas) si no viene pre-calculado
    if 'vbary' not in df.columns:
        vgas  = df['vgas'].values  if 'vgas'  in df.columns else np.zeros(len(df))
        vdisk = df['vdisk'].values if 'vdisk' in df.columns else np.zeros(len(df))
        df['vbary'] = np.sqrt(vgas**2 + vdisk**2)

    return df.dropna(subset=['r_kpc', 'vobs', 'evobs'])


# ==========================================
# 3. MOTOR DE CÁLCULO SCM (Hinge Dinámico)
# ==========================================
def predict_scm_blind(df, coeffs):
    """Predict SCM circular velocity using the Velos Hinge model.

    The model adds a "hinge" term that activates at low baryonic accelerations
    (the deep-MOND / Motor-de-Velos regime):

        log_gbar   = log10(V_bary² / r)
        hinge_term = d · max(0, logg0 − log_gbar)
        log_V_scm  = a · log_gbar + hinge_term + C
        V_total    = sqrt(V_bary² + V_scm²)

    No free parameters are adjusted per galaxy; only the globally calibrated
    SPARC coefficients are used.

    Parameters
    ----------
    df : pd.DataFrame
        Galaxy data with columns ``vbary`` [km/s] and ``r_kpc`` [kpc].
    coeffs : dict
        SCM fixed coefficients: ``a``, ``d``, ``logg0``, ``C``.

    Returns
    -------
    v_total : ndarray
        Predicted circular velocity [km/s] in quadrature sum.
    """
    vbar = df['vbary'].values
    r    = df['r_kpc'].values
    # g_bar local (aceleración bariónica)
    log_gbar = np.log10(np.maximum((vbar**2) / np.maximum(r, 0.1), 1e-15))

    # El Hinge: Presión de Velos activada por baja aceleración
    hinge_term = coeffs['d'] * np.maximum(0, coeffs['logg0'] - log_gbar)

    # Ecuación de estado del Velo Inerte
    log_vscm = coeffs['a'] * log_gbar + hinge_term + coeffs['C']
    v_scm = 10**log_vscm

    # Retorna suma en cuadratura (Modelo de Condensación Fluida)
    return np.sqrt(vbar**2 + v_scm**2)


# ==========================================
# 4. EL NOTARIO (Registro de auditoría)
# ==========================================
def log_veredicto(res, coeffs, filename="veredicto_final.txt"):
    """Append the galaxy verdict to an audit log file.

    Creates the file with a header on first call; subsequent calls append one
    data line per galaxy.

    Parameters
    ----------
    res : dict
        Result dict with keys ``galaxy``, ``rmse_bar``, ``rmse_scm``,
        ``delta_chi2``.
    coeffs : dict
        SCM fixed coefficients logged in the file header.
    filename : str, optional
        Path to the audit file (default ``"veredicto_final.txt"``).
    """
    header = not os.path.isfile(filename)
    with open(filename, "a") as f:
        if header:
            f.write("# VEREDICTO TEST CIEGO - FRAMEWORK SCM\n")
            f.write(f"# Coeffs: {coeffs}\n")
            f.write(f"# Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n#\n")
            f.write(f"{'galaxia':<15} {'rmse_bar':<10} {'rmse_scm':<10} {'delta_chi2':<12} {'mejora'}\n")
            f.write("-" * 65 + "\n")

        mejora = "SÍ" if res['delta_chi2'] < -5 else "NO"
        f.write(f"{res['galaxy']:<15} {res['rmse_bar']:<10.2f} {res['rmse_scm']:<10.2f} "
                f"{res['delta_chi2']:<12.2f} {mejora}\n")


# ==========================================
# 5. EJECUCIÓN (EL BANCO DE PRUEBAS)
# ==========================================
def run_batch(galaxias, coeffs=SCM_FIXED_PARAMS, verdict_file="veredicto_final.txt"):
    """Run the blind SCM test on a list of galaxies and return results.

    Parameters
    ----------
    galaxias : list[str]
        Galaxy identifiers to process.
    coeffs : dict, optional
        SCM fixed coefficients.  Defaults to :data:`SCM_FIXED_PARAMS`.
    verdict_file : str, optional
        Path to the output audit file.

    Returns
    -------
    results : list[dict]
        Per-galaxy result dicts with keys ``galaxy``, ``rmse_bar``,
        ``rmse_scm``, ``delta_chi2``.
    """
    results = []
    for g_id in galaxias:
        try:
            df = load_lt_file(g_id)
            v_obs  = df['vobs'].values
            e_vobs = df['evobs'].values
            v_bar  = df['vbary'].values

            # 1. Modelo Nulo (Solo Bariones)
            chi2_bar = np.sum(((v_obs - v_bar) / e_vobs)**2)
            rmse_bar = np.sqrt(np.mean((v_obs - v_bar)**2))

            # 2. Modelo SCM (Test Ciego sin ajustes locales)
            v_scm    = predict_scm_blind(df, coeffs)
            chi2_scm = np.sum(((v_obs - v_scm) / e_vobs)**2)
            rmse_scm = np.sqrt(np.mean((v_obs - v_scm)**2))

            res = {
                'galaxy':     g_id,
                'rmse_bar':   float(rmse_bar),
                'rmse_scm':   float(rmse_scm),
                'delta_chi2': float(chi2_scm - chi2_bar),
            }

            log_veredicto(res, coeffs, filename=verdict_file)
            mejora = "SÍ" if res['delta_chi2'] < -5 else "NO"
            print(f"✅ {g_id}: Δχ² = {res['delta_chi2']:.2f} | Mejora: {mejora}")
            results.append(res)

        except Exception as e:
            print(f"❌ Error procesando {g_id}: {e}")

    return results


if __name__ == "__main__":
    # Primero solo DDO154 para el Sanity Check. Si sale bien, añade las demás.
    galaxias = ["DDO154", "DDO53", "NGC2366", "IC2574", "WLM"]

    print(f"🚀 Iniciando Batch: {len(galaxias)} galaxias en el visor.")

    run_batch(galaxias)

    print("\n🏁 Proceso terminado. Revisa 'veredicto_final.txt'.")
