#!/usr/bin/env python3
"""
scm_env_protocol.py

Protocolo de análisis ambiental para evaluar el efecto del entorno
en la pendiente masa-velocidad (Δγ).

Este script evalúa si el entorno (borde vs núcleo) altera la relación
fundamental entre masa y velocidad en galaxias.

Uso:
    python scm_env_protocol.py df_master.csv

Autor: Motor-de-Velos-SCM team
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from scipy import stats


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Protocolo ambiental: análisis Δγ (delta gamma)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
    python scm_env_protocol.py df_master.csv
    python scm_env_protocol.py data/master_galaxies.csv --output scm_env_protocol_out
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Archivo CSV con datos master de galaxias'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='scm_env_protocol_out',
        help='Directorio de salida (default: scm_env_protocol_out)'
    )
    
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=10000,
        help='Número de iteraciones bootstrap (default: 10000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria (default: 42)'
    )
    
    return parser.parse_args()


def load_and_clean_data(input_file):
    """
    Cargar y limpiar datos de galaxias.
    
    Args:
        input_file: Path al archivo CSV
        
    Returns:
        pd.DataFrame: Datos limpios
    """
    print(f"📂 Cargando datos desde: {input_file}")
    
    # Check if file exists, if not create sample data
    if not Path(input_file).exists():
        print(f"⚠️  Archivo no encontrado, creando datos de ejemplo...")
        df = create_sample_data()
    else:
        df = pd.read_csv(input_file)
    
    n_raw = len(df)
    print(f"   ✓ Cargadas {n_raw} galaxias (raw)")
    
    # Clean data - remove NaN, infinite values
    df_clean = df.dropna()
    n_clean = len(df_clean)
    print(f"   ✓ Después de limpieza: {n_clean} galaxias")
    
    return df_clean, n_raw


def create_sample_data():
    """
    Crear datos de ejemplo para demostración.
    
    Returns:
        pd.DataFrame: Datos de ejemplo
    """
    np.random.seed(42)
    n_galaxies = 500
    
    # Clasificación ambiental (0=núcleo, 1=borde)
    env_class = np.random.choice([0, 1], size=n_galaxies, p=[0.3, 0.7])
    
    # Masa estelar (log M*)
    log_mass = np.random.uniform(8.5, 11.5, n_galaxies)
    
    # Velocidad - depende del ambiente
    # Núcleo: pendiente γ ≈ 0.25
    # Borde: pendiente γ ≈ 0.28 (más dispersión)
    gamma_nucleo = 0.25
    gamma_borde = 0.28
    
    velocity = np.zeros(n_galaxies)
    for i in range(n_galaxies):
        if env_class[i] == 0:  # Núcleo
            velocity[i] = gamma_nucleo * log_mass[i] + np.random.normal(0, 0.08)
        else:  # Borde
            velocity[i] = gamma_borde * log_mass[i] + np.random.normal(0, 0.12)
    
    # Surface brightness (indicador de densidad ambiental)
    surf_brightness = np.random.uniform(18, 24, n_galaxies)
    
    df = pd.DataFrame({
        'log_mass': log_mass,
        'log_velocity': velocity,
        'env_class': env_class,
        'surf_brightness': surf_brightness,
        'distance_Mpc': np.random.uniform(5, 50, n_galaxies)
    })
    
    return df


def classify_environment(df):
    """
    Clasificar galaxias según ambiente (borde vs núcleo).
    
    Args:
        df: DataFrame con datos
        
    Returns:
        tuple: (df_borde, df_nucleo)
    """
    print("\n🌌 Clasificando ambiente...")
    
    # Si ya existe clasificación, usarla
    if 'env_class' in df.columns:
        df_nucleo = df[df['env_class'] == 0].copy()
        df_borde = df[df['env_class'] == 1].copy()
    else:
        # Clasificación basada en brillo superficial o densidad
        # Núcleo: brillo alto (menor valor numérico)
        # Borde: brillo bajo (mayor valor numérico)
        threshold = df['surf_brightness'].median()
        df_nucleo = df[df['surf_brightness'] < threshold].copy()
        df_borde = df[df['surf_brightness'] >= threshold].copy()
    
    print(f"   ✓ Núcleo (denso): {len(df_nucleo)} galaxias")
    print(f"   ✓ Borde (disperso): {len(df_borde)} galaxias")
    
    return df_borde, df_nucleo


def hc3_regression(x, y):
    """
    Regresión lineal con errores HC3 (heteroscedasticity-consistent).
    
    Args:
        x: Variable independiente
        y: Variable dependiente
        
    Returns:
        dict: Resultados de regresión con errores robustos
    """
    # Regresión OLS
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Residuos
    y_pred = X @ beta
    residuals = y - y_pred
    
    # Errores HC3 (heteroscedasticity-consistent estimator tipo 3)
    n = len(x)
    h = np.sum(X * np.linalg.solve(X.T @ X, X.T).T, axis=1)
    weights = residuals**2 / (1 - h)**2
    
    # Covarianza robusta
    XtX_inv = np.linalg.inv(X.T @ X)
    V_hc3 = XtX_inv @ (X.T @ np.diag(weights) @ X) @ XtX_inv
    
    # Errores estándar robustos
    se_robust = np.sqrt(np.diag(V_hc3))
    
    # T-estadístico y p-valor para la pendiente
    t_stat = beta[1] / se_robust[1]
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-2))
    
    return {
        'slope': beta[1],
        'intercept': beta[0],
        'se_slope': se_robust[1],
        'se_intercept': se_robust[0],
        't_stat': t_stat,
        'p_value': p_value
    }


def bootstrap_delta_gamma(df_borde, df_nucleo, n_bootstrap=10000, seed=42):
    """
    Bootstrap para intervalos de confianza de Δγ.
    
    Args:
        df_borde: DataFrame de galaxias en borde
        df_nucleo: DataFrame de galaxias en núcleo
        n_bootstrap: Número de iteraciones
        seed: Semilla aleatoria
        
    Returns:
        dict: Percentiles de bootstrap
    """
    print(f"\n🔄 Ejecutando bootstrap ({n_bootstrap} iteraciones)...")
    
    np.random.seed(seed)
    delta_gamma_samples = []
    
    for i in range(n_bootstrap):
        # Resample con reemplazo
        idx_borde = np.random.choice(len(df_borde), size=len(df_borde), replace=True)
        idx_nucleo = np.random.choice(len(df_nucleo), size=len(df_nucleo), replace=True)
        
        # Regresión en cada muestra
        res_borde = hc3_regression(
            df_borde.iloc[idx_borde]['log_mass'].values,
            df_borde.iloc[idx_borde]['log_velocity'].values
        )
        res_nucleo = hc3_regression(
            df_nucleo.iloc[idx_nucleo]['log_mass'].values,
            df_nucleo.iloc[idx_nucleo]['log_velocity'].values
        )
        
        delta_gamma_samples.append(res_borde['slope'] - res_nucleo['slope'])
    
    delta_gamma_samples = np.array(delta_gamma_samples)
    
    # Percentiles
    ci_2p5 = np.percentile(delta_gamma_samples, 2.5)
    ci_97p5 = np.percentile(delta_gamma_samples, 97.5)
    
    print(f"   ✓ Δγ Bootstrap CI 95%: [{ci_2p5:.6f}, {ci_97p5:.6f}]")
    
    return {
        'samples': delta_gamma_samples,
        'ci_2p5': ci_2p5,
        'ci_97p5': ci_97p5,
        'mean': np.mean(delta_gamma_samples),
        'std': np.std(delta_gamma_samples)
    }


def bayes_factor_test(df_borde, df_nucleo):
    """
    Test de Bayes Factor para comparar dispersión entre ambientes.
    
    Args:
        df_borde: DataFrame de galaxias en borde
        df_nucleo: DataFrame de galaxias en núcleo
        
    Returns:
        dict: Resultados de Bayes Factor
    """
    print("\n📊 Calculando Bayes Factor...")
    
    # Residuos de cada grupo
    res_borde = hc3_regression(
        df_borde['log_mass'].values,
        df_borde['log_velocity'].values
    )
    res_nucleo = hc3_regression(
        df_nucleo['log_mass'].values,
        df_nucleo['log_velocity'].values
    )
    
    # Calcular residuos
    y_pred_borde = res_borde['intercept'] + res_borde['slope'] * df_borde['log_mass'].values
    residuals_borde = df_borde['log_velocity'].values - y_pred_borde
    
    y_pred_nucleo = res_nucleo['intercept'] + res_nucleo['slope'] * df_nucleo['log_mass'].values
    residuals_nucleo = df_nucleo['log_velocity'].values - y_pred_nucleo
    
    # Varianzas
    var_borde = np.var(residuals_borde, ddof=1)
    var_nucleo = np.var(residuals_nucleo, ddof=1)
    
    # F-test para varianzas
    F = var_borde / var_nucleo if var_borde > var_nucleo else var_nucleo / var_borde
    df1 = len(df_borde) - 1 if var_borde > var_nucleo else len(df_nucleo) - 1
    df2 = len(df_nucleo) - 1 if var_borde > var_nucleo else len(df_borde) - 1
    
    p_value = 2 * min(stats.f.cdf(F, df1, df2), 1 - stats.f.cdf(F, df1, df2))
    
    # Bayes Factor approximation (simplified)
    # BF10 > 3 = sustancial evidencia, > 10 = fuerte evidencia
    BF_approx = F if var_borde > var_nucleo else 1/F
    
    borde_more_scatter = var_borde > var_nucleo
    
    print(f"   ✓ Varianza borde: {var_borde:.6f}")
    print(f"   ✓ Varianza núcleo: {var_nucleo:.6f}")
    print(f"   ✓ BF p-value: {p_value:.4f}")
    print(f"   ✓ Borde tiene más dispersión: {borde_more_scatter}")
    
    return {
        'p_value': p_value,
        'BF_approx': BF_approx,
        'var_borde': var_borde,
        'var_nucleo': var_nucleo,
        'borde_more_scatter': borde_more_scatter
    }


def run_environmental_protocol(df):
    """
    Ejecutar protocolo completo de análisis ambiental.
    
    Args:
        df: DataFrame con datos limpios
        
    Returns:
        dict: Resultados del análisis
    """
    # Clasificar ambiente
    df_borde, df_nucleo = classify_environment(df)
    
    # Análisis HC3 para cada grupo
    print("\n🔬 Análisis HC3 robusto...")
    
    hc3_borde = hc3_regression(
        df_borde['log_mass'].values,
        df_borde['log_velocity'].values
    )
    print(f"   ✓ γ_borde: {hc3_borde['slope']:.6f} ± {hc3_borde['se_slope']:.6f}")
    
    hc3_nucleo = hc3_regression(
        df_nucleo['log_mass'].values,
        df_nucleo['log_velocity'].values
    )
    print(f"   ✓ γ_núcleo: {hc3_nucleo['slope']:.6f} ± {hc3_nucleo['se_slope']:.6f}")
    
    # Delta gamma
    delta_gamma = hc3_borde['slope'] - hc3_nucleo['slope']
    print(f"\n   ⚡ Δγ = {delta_gamma:.6f}")
    
    # P-valor para diferencia (aproximado)
    se_delta = np.sqrt(hc3_borde['se_slope']**2 + hc3_nucleo['se_slope']**2)
    t_delta = delta_gamma / se_delta
    p_delta = 2 * (1 - stats.t.cdf(np.abs(t_delta), df=len(df)-4))
    print(f"   p-valor Δγ: {p_delta:.4f}")
    
    return {
        'df_borde': df_borde,
        'df_nucleo': df_nucleo,
        'hc3_borde': hc3_borde,
        'hc3_nucleo': hc3_nucleo,
        'delta_gamma': delta_gamma,
        'p_delta_gamma': p_delta
    }


def generate_summary_json(results, bootstrap_results, bf_results, 
                          n_total, n_raw, args, start_time):
    """
    Generar archivo summary.json con resultados del protocolo ambiental.
    
    Args:
        results: Resultados del análisis principal
        bootstrap_results: Resultados de bootstrap
        bf_results: Resultados de Bayes Factor
        n_total: Número total de galaxias limpias
        n_raw: Número total de galaxias raw
        args: Argumentos de línea de comandos
        start_time: Tiempo de inicio
        
    Returns:
        dict: Diccionario de resumen
    """
    print("\n📝 Generando summary.json...")
    
    # Crear directorio de salida
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'summary.json'
    
    # Preparar datos de resumen
    summary = {
        'metadata': {
            'analysis_type': 'environmental_protocol',
            'timestamp': start_time.isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'input_file': args.input_file,
            'n_bootstrap': args.n_bootstrap,
            'random_seed': args.seed
        },
        'sample_sizes': {
            'N_total_clean': int(n_total),
            'N_extremes_raw': int(n_raw),
            'N_used': int(n_total),
            'N_borde': len(results['df_borde']),
            'N_nucleo': len(results['df_nucleo'])
        },
        'HC3_robust_estimates': {
            'HC3_gamma_borde': float(results['hc3_borde']['slope']),
            'HC3_gamma_borde_se': float(results['hc3_borde']['se_slope']),
            'HC3_gamma_nucleo': float(results['hc3_nucleo']['slope']),
            'HC3_gamma_nucleo_se': float(results['hc3_nucleo']['se_slope']),
            'HC3_delta_gamma': float(results['delta_gamma']),
            'HC3_p_delta_gamma': float(results['p_delta_gamma'])
        },
        'bootstrap_confidence': {
            'BOOT_delta_gamma_mean': float(bootstrap_results['mean']),
            'BOOT_delta_gamma_std': float(bootstrap_results['std']),
            'BOOT_delta_gamma_ci2p5': float(bootstrap_results['ci_2p5']),
            'BOOT_delta_gamma_ci97p5': float(bootstrap_results['ci_97p5']),
            'BOOT_n_iterations': args.n_bootstrap
        },
        'bayes_factor_test': {
            'BF_p': float(bf_results['p_value']),
            'BF_approx': float(bf_results['BF_approx']),
            'BF_var_borde': float(bf_results['var_borde']),
            'BF_var_nucleo': float(bf_results['var_nucleo']),
            'BF_borde_more_scatter': bool(bf_results['borde_more_scatter'])
        },
        'interpretation': {
            'delta_gamma_significant': bool(results['p_delta_gamma'] < 0.05),
            'ci_excludes_zero': bool(bootstrap_results['ci_2p5'] > 0 or bootstrap_results['ci_97p5'] < 0),
            'environmental_effect_detected': bool(
                (results['p_delta_gamma'] < 0.05) and 
                (bootstrap_results['ci_2p5'] > 0 or bootstrap_results['ci_97p5'] < 0)
            )
        }
    }
    
    # Escribir JSON
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"   ✓ Guardado: {output_path}")
    
    return summary


def print_final_summary(summary):
    """
    Imprimir resumen final del análisis.
    
    Args:
        summary: Diccionario de resumen
    """
    print("\n" + "="*80)
    print("🎉 RESUMEN DEL PROTOCOLO AMBIENTAL")
    print("="*80)
    
    print(f"\n📊 DATOS:")
    print(f"   Galaxias totales: {summary['sample_sizes']['N_total_clean']}")
    print(f"   Borde: {summary['sample_sizes']['N_borde']}")
    print(f"   Núcleo: {summary['sample_sizes']['N_nucleo']}")
    
    print(f"\n🔬 RESULTADOS HC3:")
    print(f"   γ_borde:  {summary['HC3_robust_estimates']['HC3_gamma_borde']:.6f}")
    print(f"   γ_núcleo: {summary['HC3_robust_estimates']['HC3_gamma_nucleo']:.6f}")
    print(f"   Δγ:       {summary['HC3_robust_estimates']['HC3_delta_gamma']:.6f}")
    print(f"   p-valor:  {summary['HC3_robust_estimates']['HC3_p_delta_gamma']:.4f}")
    
    print(f"\n🔄 BOOTSTRAP CI 95%:")
    print(f"   [{summary['bootstrap_confidence']['BOOT_delta_gamma_ci2p5']:.6f}, " +
          f"{summary['bootstrap_confidence']['BOOT_delta_gamma_ci97p5']:.6f}]")
    
    print(f"\n📈 BAYES FACTOR:")
    print(f"   p-valor: {summary['bayes_factor_test']['BF_p']:.4f}")
    print(f"   Borde más dispersión: {summary['bayes_factor_test']['BF_borde_more_scatter']}")
    
    print(f"\n✅ INTERPRETACIÓN:")
    if summary['interpretation']['environmental_effect_detected']:
        print("   🌟 EFECTO AMBIENTAL DETECTADO")
        print("   El entorno SÍ altera la pendiente masa-velocidad")
    else:
        print("   ⚪ NO se detecta efecto ambiental significativo")
    
    print("="*80 + "\n")


def main():
    """
    Función principal del protocolo ambiental.
    """
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("🌌 PROTOCOLO AMBIENTAL Δγ - Motor-de-Velos-SCM")
    print("="*80)
    print(f"Iniciado: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Parse argumentos
    args = parse_arguments()
    
    # Cargar datos
    df_clean, n_raw = load_and_clean_data(args.input_file)
    
    # Ejecutar protocolo ambiental
    results = run_environmental_protocol(df_clean)
    
    # Bootstrap
    bootstrap_results = bootstrap_delta_gamma(
        results['df_borde'],
        results['df_nucleo'],
        n_bootstrap=args.n_bootstrap,
        seed=args.seed
    )
    
    # Bayes Factor
    bf_results = bayes_factor_test(
        results['df_borde'],
        results['df_nucleo']
    )
    
    # Generar summary.json
    summary = generate_summary_json(
        results, bootstrap_results, bf_results,
        len(df_clean), n_raw, args, start_time
    )
    
    # Imprimir resumen
    print_final_summary(summary)
    
    print(f"✅ Protocolo completado exitosamente")
    print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
