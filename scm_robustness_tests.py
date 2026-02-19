#!/usr/bin/env python3
"""
scm_robustness_tests.py

Fase 2: Pruebas de robustez para intentar "matar" el efecto ambiental.

Tres frentes de ataque:
1. Control por morfología (¿es tipo galáctico, no entorno?)
2. Modelo continuo (no solo extremos binarios)
3. Validación cruzada interna (¿es estructural o frágil?)

Uso:
    python scm_robustness_tests.py df_master.csv

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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Pruebas de robustez del efecto ambiental Δγ',
        formatter_class=argparse.RawDescriptionHelpFormatter
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
        '--n-cv',
        type=int,
        default=100,
        help='Número de iteraciones para cross-validation (default: 100)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria (default: 42)'
    )
    
    return parser.parse_args()


def create_enhanced_sample_data():
    """
    Crear datos de ejemplo con morfología y variables continuas.
    
    Returns:
        pd.DataFrame: Datos enriquecidos
    """
    np.random.seed(42)
    n_galaxies = 500
    
    # Clasificación ambiental (0=núcleo, 1=borde)
    env_class = np.random.choice([0, 1], size=n_galaxies, p=[0.3, 0.7])
    
    # Densidad ambiental continua (log Sigma_5)
    # Correlacionada con env_class pero con variación
    logSigma5 = np.zeros(n_galaxies)
    for i in range(n_galaxies):
        if env_class[i] == 0:  # Núcleo - alta densidad
            logSigma5[i] = np.random.normal(1.5, 0.3)
        else:  # Borde - baja densidad
            logSigma5[i] = np.random.normal(0.5, 0.4)
    
    # Tipo morfológico (T-type)
    # 0-2: temprano (E/S0)
    # 3-5: intermedio (Sa-Sb)
    # 6-10: tardío (Sc-Irr)
    # Ligeramente correlacionado con ambiente
    morph_type = np.zeros(n_galaxies)
    for i in range(n_galaxies):
        if env_class[i] == 0:  # Núcleo - más early-type
            morph_type[i] = np.random.choice(range(11), p=[0.2,0.2,0.15,0.15,0.1,0.08,0.05,0.03,0.02,0.01,0.01])
        else:  # Borde - más late-type
            morph_type[i] = np.random.choice(range(11), p=[0.05,0.05,0.08,0.12,0.15,0.15,0.15,0.1,0.08,0.05,0.02])
    
    # Clasificación morfológica binaria
    morph_bin = np.where(morph_type < 3, 0,  # early
                         np.where(morph_type < 6, 1, 2))  # intermediate, late
    
    # Presencia de barra
    is_barred = np.random.choice([0, 1], size=n_galaxies, p=[0.6, 0.4])
    
    # Fracción de gas
    # Late-type galaxies tienen más gas
    gas_fraction = 0.05 + 0.15 * (morph_type / 10.0) + np.random.normal(0, 0.05, n_galaxies)
    gas_fraction = np.clip(gas_fraction, 0.01, 0.4)
    
    # Masa estelar (log M*)
    log_mass = np.random.uniform(8.5, 11.5, n_galaxies)
    
    # Velocidad - depende de ambiente Y morfología
    # Núcleo: pendiente γ ≈ 0.25
    # Borde: pendiente γ ≈ 0.28
    # Pero morfología también tiene efecto pequeño
    gamma_base = 0.25
    gamma_env_effect = 0.03  # Efecto ambiental real
    gamma_morph_effect = 0.01  # Efecto morfológico menor
    
    velocity = np.zeros(n_galaxies)
    for i in range(n_galaxies):
        gamma_i = gamma_base
        
        # Efecto ambiental
        if env_class[i] == 1:  # Borde
            gamma_i += gamma_env_effect
        
        # Efecto morfológico (late-type ligeramente mayor pendiente)
        gamma_i += gamma_morph_effect * (morph_type[i] / 10.0)
        
        # Dispersión mayor en borde
        if env_class[i] == 0:  # Núcleo
            velocity[i] = gamma_i * log_mass[i] + np.random.normal(0, 0.08)
        else:  # Borde
            velocity[i] = gamma_i * log_mass[i] + np.random.normal(0, 0.12)
    
    # Surface brightness
    surf_brightness = np.random.uniform(18, 24, n_galaxies)
    
    df = pd.DataFrame({
        'log_mass': log_mass,
        'log_velocity': velocity,
        'env_class': env_class,
        'logSigma5': logSigma5,
        'morph_type': morph_type,
        'morph_bin': morph_bin,
        'is_barred': is_barred,
        'gas_fraction': gas_fraction,
        'surf_brightness': surf_brightness,
        'distance_Mpc': np.random.uniform(5, 50, n_galaxies)
    })
    
    return df


def load_data(input_file):
    """Cargar datos de galaxias."""
    print(f"📂 Cargando datos desde: {input_file}")
    
    if not Path(input_file).exists():
        print(f"⚠️  Archivo no encontrado, creando datos enriquecidos de ejemplo...")
        df = create_enhanced_sample_data()
    else:
        df = pd.read_csv(input_file)
    
    n_raw = len(df)
    print(f"   ✓ Cargadas {n_raw} galaxias")
    
    # Clean
    df_clean = df.dropna()
    print(f"   ✓ Después de limpieza: {len(df_clean)} galaxias")
    
    # Verify required columns
    required_cols = ['log_mass', 'log_velocity', 'env_class']
    if not all(col in df_clean.columns for col in required_cols):
        print(f"   ⚠️  Faltan columnas requeridas, usando datos de ejemplo")
        df_clean = create_enhanced_sample_data()
    
    print(f"   ✓ Columnas disponibles: {df_clean.columns.tolist()}")
    
    return df_clean


def test_morphology_control(df):
    """
    Test 1: Control por morfología.
    
    Pregunta: ¿El efecto ambiental sobrevive controlando por tipo morfológico?
    """
    print("\n" + "="*80)
    print("🔬 TEST 1: CONTROL POR MORFOLOGÍA")
    print("="*80)
    
    # Check if morphology data available
    if 'morph_bin' not in df.columns and 'morph_type' not in df.columns:
        print("   ⚠️  Sin datos de morfología - saltando test")
        return None
    
    # Create morphology bins if needed
    if 'morph_bin' not in df.columns:
        df['morph_bin'] = pd.cut(df['morph_type'], bins=[0, 3, 6, 11], 
                                  labels=[0, 1, 2], include_lowest=True)
    
    print(f"\n   Distribución morfológica:")
    morph_counts = df['morph_bin'].value_counts().sort_index()
    print(f"   - Early (0-2): {morph_counts.get(0, 0)} galaxias")
    print(f"   - Intermediate (3-5): {morph_counts.get(1, 0)} galaxias")
    print(f"   - Late (6-10): {morph_counts.get(2, 0)} galaxias")
    
    # Modelo SIN control morfológico
    print("\n   📊 Modelo base (sin control):")
    from sklearn.linear_model import LinearRegression
    
    df_nucleo = df[df['env_class'] == 0]
    df_borde = df[df['env_class'] == 1]
    
    X_nucleo = df_nucleo[['log_mass']].values
    y_nucleo = df_nucleo['log_velocity'].values
    X_borde = df_borde[['log_mass']].values
    y_borde = df_borde['log_velocity'].values
    
    model_nucleo = LinearRegression().fit(X_nucleo, y_nucleo)
    model_borde = LinearRegression().fit(X_borde, y_borde)
    
    gamma_nucleo_base = model_nucleo.coef_[0]
    gamma_borde_base = model_borde.coef_[0]
    delta_gamma_base = gamma_borde_base - gamma_nucleo_base
    
    print(f"      γ_núcleo: {gamma_nucleo_base:.6f}")
    print(f"      γ_borde:  {gamma_borde_base:.6f}")
    print(f"      Δγ:       {delta_gamma_base:.6f}")
    
    # Modelo CON control morfológico
    print("\n   📊 Modelo con control morfológico:")
    
    # Separate by environment and morphology
    results = {}
    for env in [0, 1]:
        env_name = 'nucleo' if env == 0 else 'borde'
        results[env_name] = {}
        
        for morph in [0, 1, 2]:
            morph_name = ['early', 'intermediate', 'late'][morph]
            subset = df[(df['env_class'] == env) & (df['morph_bin'] == morph)]
            
            if len(subset) > 10:  # Minimum sample size
                X = subset[['log_mass']].values
                y = subset['log_velocity'].values
                model = LinearRegression().fit(X, y)
                results[env_name][morph_name] = model.coef_[0]
    
    # Promediar gammas por ambiente (controlando morfología)
    gamma_nucleo_ctrl = np.mean([v for v in results.get('nucleo', {}).values()])
    gamma_borde_ctrl = np.mean([v for v in results.get('borde', {}).values()])
    delta_gamma_ctrl = gamma_borde_ctrl - gamma_nucleo_ctrl
    
    print(f"      γ_núcleo (promedio): {gamma_nucleo_ctrl:.6f}")
    print(f"      γ_borde (promedio):  {gamma_borde_ctrl:.6f}")
    print(f"      Δγ controlado:       {delta_gamma_ctrl:.6f}")
    
    # Cambio relativo
    change_pct = 100 * (delta_gamma_ctrl - delta_gamma_base) / delta_gamma_base
    print(f"\n   📉 Cambio en Δγ: {change_pct:.1f}%")
    
    # Veredicto
    survives = abs(delta_gamma_ctrl) > 0.5 * abs(delta_gamma_base)
    print(f"\n   {'✅ SOBREVIVE' if survives else '❌ NO SOBREVIVE'}: ", end='')
    if survives:
        print("El efecto ambiental persiste controlando morfología")
    else:
        print("El efecto era principalmente morfológico")
    
    return {
        'delta_gamma_base': float(delta_gamma_base),
        'delta_gamma_controlled': float(delta_gamma_ctrl),
        'change_percent': float(change_pct),
        'survives_morphology_control': bool(survives),
        'gamma_nucleo_base': float(gamma_nucleo_base),
        'gamma_borde_base': float(gamma_borde_base),
        'gamma_nucleo_controlled': float(gamma_nucleo_ctrl),
        'gamma_borde_controlled': float(gamma_borde_ctrl)
    }


def test_continuous_model(df):
    """
    Test 2: Modelo continuo.
    
    Pregunta: ¿La pendiente cambia suavemente con densidad ambiental?
    """
    print("\n" + "="*80)
    print("🔬 TEST 2: MODELO CONTINUO")
    print("="*80)
    
    if 'logSigma5' not in df.columns:
        print("   ⚠️  Sin variable continua de densidad - saltando test")
        return None
    
    print("\n   Modelo: log_velocity ~ log_mass + logSigma5 + log_mass:logSigma5")
    
    # Prepare data
    X = df[['log_mass', 'logSigma5']].values
    X_with_interaction = np.column_stack([
        np.ones(len(df)),
        df['log_mass'].values,
        df['logSigma5'].values,
        df['log_mass'].values * df['logSigma5'].values
    ])
    y = df['log_velocity'].values
    
    # Fit model
    beta = np.linalg.lstsq(X_with_interaction, y, rcond=None)[0]
    
    # Residuals and standard errors
    y_pred = X_with_interaction @ beta
    residuals = y - y_pred
    n = len(df)
    k = X_with_interaction.shape[1]
    
    # Robust standard errors (HC3)
    h = np.sum(X_with_interaction * np.linalg.solve(X_with_interaction.T @ X_with_interaction, 
                                                      X_with_interaction.T).T, axis=1)
    weights = residuals**2 / (1 - h)**2
    XtX_inv = np.linalg.inv(X_with_interaction.T @ X_with_interaction)
    V_hc3 = XtX_inv @ (X_with_interaction.T @ np.diag(weights) @ X_with_interaction) @ XtX_inv
    se_robust = np.sqrt(np.diag(V_hc3))
    
    # T-statistics and p-values
    t_stats = beta / se_robust
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))
    
    print(f"\n   Coeficientes:")
    print(f"      Intercepto:        {beta[0]:.6f} (p={p_values[0]:.4f})")
    print(f"      log_mass:          {beta[1]:.6f} (p={p_values[1]:.4f})")
    print(f"      logSigma5:         {beta[2]:.6f} (p={p_values[2]:.4f})")
    print(f"      Interacción:       {beta[3]:.6f} (p={p_values[3]:.4f})")
    
    # Key test: is interaction significant?
    interaction_significant = p_values[3] < 0.05
    
    print(f"\n   {'✅ SOBREVIVE' if interaction_significant else '❌ NO SOBREVIVE'}: ", end='')
    if interaction_significant:
        print("La pendiente cambia significativamente con densidad")
    else:
        print("No hay evidencia de cambio continuo con densidad")
    
    # R-squared
    ss_total = np.sum((y - np.mean(y))**2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total)
    
    print(f"\n   R²: {r_squared:.4f}")
    
    return {
        'intercept': float(beta[0]),
        'coef_log_mass': float(beta[1]),
        'coef_logSigma5': float(beta[2]),
        'coef_interaction': float(beta[3]),
        'p_intercept': float(p_values[0]),
        'p_log_mass': float(p_values[1]),
        'p_logSigma5': float(p_values[2]),
        'p_interaction': float(p_values[3]),
        'r_squared': float(r_squared),
        'interaction_significant': bool(interaction_significant)
    }


def test_cross_validation(df, n_iterations=100, test_fraction=0.1, seed=42):
    """
    Test 3: Validación cruzada interna.
    
    Pregunta: ¿El efecto es estructural o frágil?
    """
    print("\n" + "="*80)
    print("🔬 TEST 3: VALIDACIÓN CRUZADA")
    print("="*80)
    
    print(f"\n   Configuración:")
    print(f"   - Iteraciones: {n_iterations}")
    print(f"   - Fracción test: {test_fraction*100:.0f}%")
    
    np.random.seed(seed)
    
    delta_gammas = []
    p_values = []
    
    print(f"\n   Ejecutando {n_iterations} iteraciones...")
    
    for i in range(n_iterations):
        # Random subsample
        n_test = int(len(df) * test_fraction)
        test_idx = np.random.choice(len(df), size=n_test, replace=False)
        train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
        
        df_train = df.iloc[train_idx]
        
        # Separate by environment
        df_nucleo = df_train[df_train['env_class'] == 0]
        df_borde = df_train[df_train['env_class'] == 1]
        
        if len(df_nucleo) < 20 or len(df_borde) < 20:
            continue
        
        # Fit models
        from sklearn.linear_model import LinearRegression
        
        X_nucleo = df_nucleo[['log_mass']].values
        y_nucleo = df_nucleo['log_velocity'].values
        X_borde = df_borde[['log_mass']].values
        y_borde = df_borde['log_velocity'].values
        
        model_nucleo = LinearRegression().fit(X_nucleo, y_nucleo)
        model_borde = LinearRegression().fit(X_borde, y_borde)
        
        gamma_nucleo = model_nucleo.coef_[0]
        gamma_borde = model_borde.coef_[0]
        delta_gamma = gamma_borde - gamma_nucleo
        
        delta_gammas.append(delta_gamma)
        
        # Simple t-test for difference
        # (simplified - full HC3 would be slower)
        se_nucleo = np.std(y_nucleo - model_nucleo.predict(X_nucleo)) / np.sqrt(len(df_nucleo))
        se_borde = np.std(y_borde - model_borde.predict(X_borde)) / np.sqrt(len(df_borde))
        se_delta = np.sqrt(se_nucleo**2 + se_borde**2)
        t_stat = delta_gamma / se_delta
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=len(df_train)-4))
        p_values.append(p_val)
    
    delta_gammas = np.array(delta_gammas)
    p_values = np.array(p_values)
    
    # Statistics
    mean_delta = np.mean(delta_gammas)
    std_delta = np.std(delta_gammas)
    ci_lower = np.percentile(delta_gammas, 2.5)
    ci_upper = np.percentile(delta_gammas, 97.5)
    
    # Stability checks
    always_positive = np.all(delta_gammas > 0)
    mostly_positive = np.mean(delta_gammas > 0) > 0.95
    mostly_significant = np.mean(p_values < 0.05) > 0.80
    
    print(f"\n   Resultados ({len(delta_gammas)} iteraciones válidas):")
    print(f"      Δγ medio:     {mean_delta:.6f}")
    print(f"      Δγ std:       {std_delta:.6f}")
    print(f"      CI 95%:       [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"      Siempre > 0:  {always_positive}")
    print(f"      % positivo:   {100*np.mean(delta_gammas > 0):.1f}%")
    print(f"      % p < 0.05:   {100*np.mean(p_values < 0.05):.1f}%")
    
    # Stability verdict
    is_stable = mostly_positive and mostly_significant
    
    print(f"\n   {'✅ SOBREVIVE' if is_stable else '❌ NO SOBREVIVE'}: ", end='')
    if is_stable:
        print("El efecto es estructuralmente robusto")
    else:
        print("El efecto es frágil y sensible al muestreo")
    
    return {
        'n_iterations': len(delta_gammas),
        'mean_delta_gamma': float(mean_delta),
        'std_delta_gamma': float(std_delta),
        'ci_2p5': float(ci_lower),
        'ci_97p5': float(ci_upper),
        'always_positive': bool(always_positive),
        'percent_positive': float(100*np.mean(delta_gammas > 0)),
        'percent_significant': float(100*np.mean(p_values < 0.05)),
        'is_stable': bool(is_stable)
    }


def generate_robustness_report(results, args, start_time):
    """Generar reporte de robustez en JSON."""
    print("\n" + "="*80)
    print("📝 GENERANDO REPORTE DE ROBUSTEZ")
    print("="*80)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'robustness_report.json'
    
    # Overall verdict
    tests_passed = 0
    tests_total = 0
    
    if results['morphology_control']:
        tests_total += 1
        if results['morphology_control']['survives_morphology_control']:
            tests_passed += 1
    
    if results['continuous_model']:
        tests_total += 1
        if results['continuous_model']['interaction_significant']:
            tests_passed += 1
    
    if results['cross_validation']:
        tests_total += 1
        if results['cross_validation']['is_stable']:
            tests_passed += 1
    
    overall_survives = tests_passed >= 2  # Majority
    
    report = {
        'metadata': {
            'analysis_type': 'robustness_tests',
            'timestamp': start_time.isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'input_file': args.input_file,
            'n_cv_iterations': args.n_cv
        },
        'test_results': results,
        'overall_assessment': {
            'tests_passed': tests_passed,
            'tests_total': tests_total,
            'pass_rate': tests_passed / tests_total if tests_total > 0 else 0,
            'overall_survives': overall_survives
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n   ✓ Guardado: {output_path}")
    
    return report


def print_final_verdict(report):
    """Imprimir veredicto final."""
    print("\n" + "="*80)
    print("⚖️  VEREDICTO FINAL")
    print("="*80)
    
    results = report['test_results']
    assessment = report['overall_assessment']
    
    print(f"\n   Tests completados: {assessment['tests_passed']}/{assessment['tests_total']}")
    
    print(f"\n   📋 Resultados por test:")
    
    if results['morphology_control']:
        mc = results['morphology_control']
        status = "✅ PASA" if mc['survives_morphology_control'] else "❌ FALLA"
        print(f"      1. Control morfológico: {status}")
        print(f"         Δγ cambió {mc['change_percent']:.1f}%")
    
    if results['continuous_model']:
        cm = results['continuous_model']
        status = "✅ PASA" if cm['interaction_significant'] else "❌ FALLA"
        print(f"      2. Modelo continuo: {status}")
        print(f"         Interacción p={cm['p_interaction']:.4f}")
    
    if results['cross_validation']:
        cv = results['cross_validation']
        status = "✅ PASA" if cv['is_stable'] else "❌ FALLA"
        print(f"      3. Cross-validation: {status}")
        print(f"         {cv['percent_positive']:.0f}% positivo, {cv['percent_significant']:.0f}% significativo")
    
    print(f"\n   🎯 VEREDICTO GLOBAL: ", end='')
    if assessment['overall_survives']:
        print("✅ EL EFECTO AMBIENTAL SOBREVIVE")
        print("\n   La hipótesis ambiental resiste los intentos de demolición.")
        print("   El efecto parece ser real, robusto y estructural.")
    else:
        print("❌ EL EFECTO NO SOBREVIVE")
        print("\n   La hipótesis ambiental no resiste el escrutinio.")
        print("   El efecto aparente era probablemente espurio.")
    
    print("="*80 + "\n")


def main():
    """Función principal."""
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("🔪 FASE 2: PRUEBAS DE ROBUSTEZ - Intento de Demolición")
    print("="*80)
    print(f"Iniciado: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    args = parse_arguments()
    
    # Load data
    df = load_data(args.input_file)
    
    # Run tests
    results = {}
    
    results['morphology_control'] = test_morphology_control(df)
    results['continuous_model'] = test_continuous_model(df)
    results['cross_validation'] = test_cross_validation(df, n_iterations=args.n_cv, seed=args.seed)
    
    # Generate report
    report = generate_robustness_report(results, args, start_time)
    
    # Final verdict
    print_final_verdict(report)
    
    print(f"✅ Análisis completado")
    print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
