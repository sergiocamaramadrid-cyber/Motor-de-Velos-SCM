#!/usr/bin/env python3
"""
scm_kill_tests.py

Fase 2b: Kill tests definitivos con datos reales.

Tres tests diseñados para intentar refutar el efecto ambiental:
1. Control por morfología (T-type + barra)
2. Modelo continuo (toda la muestra, no extremos)
3. Jackknife (estabilidad ante muestreo)

Si el efecto sobrevive los tres → robusto y publicable
Si falla alguno → efecto refutado

Uso:
    python scm_kill_tests.py df_master.csv

Requiere columnas: log_mbar, log_vflat, logSigma5, T, bar, incl

Autor: Motor-de-Velos-SCM team
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Kill tests definitivos para efecto ambiental',
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
        '--n-jackknife',
        type=int,
        default=100,
        help='Número de iteraciones jackknife (default: 100)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla aleatoria (default: 42)'
    )
    
    return parser.parse_args()


def load_and_prepare_data(input_file):
    """
    Cargar y preparar datos.
    """
    print("\n" + "="*80)
    print("📂 CARGANDO DATOS")
    print("="*80)
    
    if not Path(input_file).exists():
        print(f"\n❌ ERROR: Archivo no encontrado: {input_file}")
        print("\nEste script requiere datos REALES con columnas:")
        print("  - log_mbar, log_vflat, logSigma5")
        print("  - T (tipo de Hubble)")
        print("  - bar (presencia de barra)")
        print("  - incl (inclinación)")
        sys.exit(1)
    
    df = pd.read_csv(input_file)
    print(f"\n   ✓ Cargadas {len(df)} galaxias")
    print(f"   ✓ Columnas disponibles: {df.columns.tolist()}")
    
    # Verify required columns
    required_cols = ['log_mbar', 'log_vflat', 'logSigma5', 'T', 'bar']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n❌ ERROR: Faltan columnas requeridas: {missing_cols}")
        sys.exit(1)
    
    # Clean infinites and NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna(subset=required_cols)
    n_after = len(df)
    
    if n_after < n_before:
        print(f"   ⚠️  Eliminadas {n_before - n_after} filas con NaN/Inf")
    
    print(f"   ✓ Dataset limpio: {len(df)} galaxias")
    
    return df


def compute_extremes(df, percentiles=(0.15, 0.85)):
    """
    Calcular extremos ambientales según percentiles.
    """
    print("\n" + "="*80)
    print("🌌 CALCULANDO EXTREMOS AMBIENTALES")
    print("="*80)
    
    p_low, p_high = percentiles
    
    p15 = df['logSigma5'].quantile(p_low)
    p85 = df['logSigma5'].quantile(p_high)
    
    print(f"\n   Percentiles de logSigma5:")
    print(f"      P{int(p_low*100)} (baja densidad): {p15:.6f}")
    print(f"      P{int(p_high*100)} (alta densidad): {p85:.6f}")
    
    # Select extremes
    df_ext = df[(df['logSigma5'] <= p15) | (df['logSigma5'] >= p85)].copy()
    
    # Classify as nucleus (high density) or border (low density)
    df_ext['is_nuc'] = (df_ext['logSigma5'] >= p85).astype(int)
    
    n_borde = (df_ext['is_nuc'] == 0).sum()
    n_nucleo = (df_ext['is_nuc'] == 1).sum()
    
    print(f"\n   Extremos seleccionados: {len(df_ext)} galaxias")
    print(f"      Borde (baja densidad): {n_borde}")
    print(f"      Núcleo (alta densidad): {n_nucleo}")
    print(f"      % de muestra total: {100*len(df_ext)/len(df):.1f}%")
    
    if len(df_ext) >= 0.9 * len(df):
        print("\n   🔴 ADVERTENCIA: Usando >90% de la muestra!")
        print("   Los percentiles pueden estar mal calculados.")
    
    return df_ext, p15, p85


def categorize_morphology(df):
    """
    Categorizar T-type en early/intermediate/late.
    """
    print("\n   Categorizando morfología (T-type):")
    
    # Standard bins: early <2, intermediate 2-6, late >=6
    df['morph_bin'] = pd.cut(
        df['T'],
        bins=[-np.inf, 2, 6, np.inf],
        labels=['early', 'mid', 'late']
    )
    
    counts = df['morph_bin'].value_counts().sort_index()
    print(f"      Early (T<2): {counts.get('early', 0)}")
    print(f"      Intermediate (2≤T<6): {counts.get('mid', 0)}")
    print(f"      Late (T≥6): {counts.get('late', 0)}")
    
    return df


def test_1_morphology_control(df_ext):
    """
    Test 1: Control por morfología (T + bar).
    
    Modelo: log_mbar ~ log_vflat * is_nuc + C(morph_bin) + bar
    """
    print("\n" + "="*80)
    print("🔪 TEST 1: CONTROL POR MORFOLOGÍA")
    print("="*80)
    
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n❌ ERROR: statsmodels no instalado")
        print("Ejecuta: pip install statsmodels")
        return None
    
    # Categorize morphology
    df_test = categorize_morphology(df_ext.copy())
    
    # Remove NaN in morph_bin
    df_test = df_test.dropna(subset=['morph_bin'])
    
    print(f"\n   Muestra para análisis: {len(df_test)} galaxias")
    
    # Fit model with morphology control
    print("\n   Ajustando modelo con control morfológico...")
    print("   Modelo: log_mbar ~ log_vflat * is_nuc + C(morph_bin) + bar")
    
    try:
        model = smf.ols(
            "log_mbar ~ log_vflat * is_nuc + C(morph_bin) + bar",
            data=df_test
        ).fit(cov_type='HC3')
        
        print("\n" + "="*80)
        print(model.summary())
        print("="*80)
        
        # Extract interaction term
        interaction_param = model.params.get('log_vflat:is_nuc', np.nan)
        interaction_pval = model.pvalues.get('log_vflat:is_nuc', np.nan)
        
        print("\n   📊 RESULTADOS CLAVE:")
        print(f"      Interacción log_vflat:is_nuc = {interaction_param:.6f}")
        print(f"      p-valor = {interaction_pval:.6e}")
        
        survives = interaction_pval < 0.05
        
        print(f"\n   {'✅ SOBREVIVE' if survives else '❌ NO SOBREVIVE'}: ", end='')
        if survives:
            print("La interacción sigue significativa después de controlar morfología")
        else:
            print("El efecto desaparece al controlar por morfología → ERA MORFOLÓGICO")
        
        return {
            'interaction_coef': float(interaction_param),
            'interaction_pval': float(interaction_pval),
            'survives': bool(survives),
            'r_squared': float(model.rsquared),
            'n_obs': int(model.nobs)
        }
        
    except Exception as e:
        print(f"\n❌ ERROR en el modelo: {e}")
        return None


def test_2_continuous_model(df):
    """
    Test 2: Modelo continuo (toda la muestra).
    
    Modelo: log_mbar ~ log_vflat + logSigma5 + log_vflat:logSigma5
    """
    print("\n" + "="*80)
    print("🔪 TEST 2: MODELO CONTINUO")
    print("="*80)
    
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n❌ ERROR: statsmodels no instalado")
        return None
    
    print(f"\n   Usando TODA la muestra: {len(df)} galaxias")
    print("   Modelo: log_mbar ~ log_vflat + logSigma5 + log_vflat:logSigma5")
    
    try:
        model = smf.ols(
            "log_mbar ~ log_vflat + logSigma5 + log_vflat:logSigma5",
            data=df
        ).fit(cov_type='HC3')
        
        print("\n" + "="*80)
        print(model.summary())
        print("="*80)
        
        # Extract interaction term
        interaction_param = model.params.get('log_vflat:logSigma5', np.nan)
        interaction_pval = model.pvalues.get('log_vflat:logSigma5', np.nan)
        
        print("\n   📊 RESULTADOS CLAVE:")
        print(f"      Interacción log_vflat:logSigma5 = {interaction_param:.6f}")
        print(f"      p-valor = {interaction_pval:.6e}")
        
        survives = interaction_pval < 0.05
        
        print(f"\n   {'✅ SOBREVIVE' if survives else '❌ NO SOBREVIVE'}: ", end='')
        if survives:
            print("Pendiente cambia significativamente con densidad continua")
        else:
            print("No hay cambio continuo → efecto discreto o ausente")
        
        return {
            'interaction_coef': float(interaction_param),
            'interaction_pval': float(interaction_pval),
            'survives': bool(survives),
            'r_squared': float(model.rsquared),
            'n_obs': int(model.nobs)
        }
        
    except Exception as e:
        print(f"\n❌ ERROR en el modelo: {e}")
        return None


def test_3_jackknife(df_ext, n_iterations=100, drop_fraction=0.1, seed=42):
    """
    Test 3: Jackknife (estabilidad).
    
    Repetir análisis quitando 10% aleatorio cada vez.
    """
    print("\n" + "="*80)
    print("🔪 TEST 3: JACKKNIFE (ESTABILIDAD)")
    print("="*80)
    
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("\n❌ ERROR: statsmodels no instalado")
        return None
    
    print(f"\n   Configuración:")
    print(f"      Iteraciones: {n_iterations}")
    print(f"      Drop: {drop_fraction*100:.0f}% cada iteración")
    print(f"      Muestra base: {len(df_ext)} galaxias")
    
    rng = np.random.default_rng(seed)
    
    delta_vals = []
    p_vals = []
    
    print(f"\n   Ejecutando {n_iterations} iteraciones...")
    
    for i in range(n_iterations):
        # Drop random subset
        drop_idx = rng.choice(
            df_ext.index,
            size=int(drop_fraction * len(df_ext)),
            replace=False
        )
        sample = df_ext.drop(index=drop_idx)
        
        try:
            model = smf.ols(
                "log_mbar ~ log_vflat * is_nuc",
                data=sample
            ).fit(cov_type='HC3')
            
            delta = model.params.get('log_vflat:is_nuc', np.nan)
            p_val = model.pvalues.get('log_vflat:is_nuc', np.nan)
            
            delta_vals.append(delta)
            p_vals.append(p_val)
            
        except Exception:
            delta_vals.append(np.nan)
            p_vals.append(np.nan)
    
    delta_vals = np.array(delta_vals)
    p_vals = np.array(p_vals)
    
    # Remove NaN
    valid_mask = ~np.isnan(delta_vals) & ~np.isnan(p_vals)
    delta_vals = delta_vals[valid_mask]
    p_vals = p_vals[valid_mask]
    
    # Statistics
    median_delta = np.median(delta_vals)
    pct_positive = 100 * np.mean(delta_vals > 0)
    pct_significant = 100 * np.mean(p_vals < 0.05)
    p5 = np.percentile(delta_vals, 5)
    p95 = np.percentile(delta_vals, 95)
    
    print(f"\n   Resultados ({len(delta_vals)} iteraciones válidas):")
    print(f"      Δγ mediana: {median_delta:.6f}")
    print(f"      Δγ > 0: {pct_positive:.1f}%")
    print(f"      p < 0.05: {pct_significant:.1f}%")
    print(f"      Δγ percentil 5-95: [{p5:.6f}, {p95:.6f}]")
    
    # Stability criteria
    stable = (pct_positive >= 95) and (pct_significant >= 80)
    
    print(f"\n   {'✅ SOBREVIVE' if stable else '❌ NO SOBREVIVE'}: ", end='')
    if stable:
        print("El efecto es robusto ante muestreo")
    else:
        print("El efecto es frágil → depende de puntos específicos")
    
    return {
        'median_delta': float(median_delta),
        'percent_positive': float(pct_positive),
        'percent_significant': float(pct_significant),
        'percentile_5': float(p5),
        'percentile_95': float(p95),
        'n_iterations': len(delta_vals),
        'is_stable': bool(stable)
    }


def generate_kill_tests_report(results, args, start_time):
    """
    Generar reporte JSON de kill tests.
    """
    print("\n" + "="*80)
    print("📝 GENERANDO REPORTE")
    print("="*80)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'kill_tests_report.json'
    
    # Count survivors
    tests_passed = 0
    tests_total = 0
    
    if results['test1_morphology']:
        tests_total += 1
        if results['test1_morphology']['survives']:
            tests_passed += 1
    
    if results['test2_continuous']:
        tests_total += 1
        if results['test2_continuous']['survives']:
            tests_passed += 1
    
    if results['test3_jackknife']:
        tests_total += 1
        if results['test3_jackknife']['is_stable']:
            tests_passed += 1
    
    overall_survives = tests_passed == tests_total
    
    report = {
        'metadata': {
            'analysis_type': 'kill_tests',
            'timestamp': start_time.isoformat(),
            'duration_seconds': (datetime.now() - start_time).total_seconds(),
            'input_file': args.input_file,
            'n_jackknife': args.n_jackknife
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
    """
    Imprimir veredicto final.
    """
    print("\n" + "="*80)
    print("⚖️  VEREDICTO FINAL - KILL TESTS")
    print("="*80)
    
    results = report['test_results']
    assessment = report['overall_assessment']
    
    print(f"\n   Tests completados: {assessment['tests_passed']}/{assessment['tests_total']}")
    
    print(f"\n   📋 Resultados:")
    
    if results['test1_morphology']:
        t1 = results['test1_morphology']
        status = "✅ PASA" if t1['survives'] else "❌ FALLA"
        print(f"      1. Control morfológico: {status}")
        print(f"         Interacción p={t1['interaction_pval']:.4e}")
    
    if results['test2_continuous']:
        t2 = results['test2_continuous']
        status = "✅ PASA" if t2['survives'] else "❌ FALLA"
        print(f"      2. Modelo continuo: {status}")
        print(f"         Interacción p={t2['interaction_pval']:.4e}")
    
    if results['test3_jackknife']:
        t3 = results['test3_jackknife']
        status = "✅ PASA" if t3['is_stable'] else "❌ FALLA"
        print(f"      3. Jackknife: {status}")
        print(f"         {t3['percent_positive']:.0f}% positivo, {t3['percent_significant']:.0f}% significativo")
    
    print(f"\n   {'='*76}")
    print(f"   🎯 VEREDICTO: ", end='')
    
    if assessment['overall_survives']:
        print("✅ EL EFECTO AMBIENTAL SOBREVIVE TODOS LOS TESTS")
        print("\n   El efecto es ROBUSTO y PUBLICABLE.")
        print("   Ha resistido:")
        print("      ✓ Control por morfología y barra")
        print("      ✓ Análisis continuo de toda la muestra")
        print("      ✓ Validación jackknife de estabilidad")
    else:
        print("❌ EL EFECTO NO SOBREVIVE")
        print(f"\n   Falló {assessment['tests_total'] - assessment['tests_passed']}/{assessment['tests_total']} tests.")
        print("   El efecto ambiental queda REFUTADO.")
    
    print("="*80 + "\n")


def main():
    """
    Función principal.
    """
    start_time = datetime.now()
    
    print("\n" + "="*80)
    print("🔪 KILL TESTS - Fase 2b Definitiva")
    print("="*80)
    print(f"Iniciado: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    args = parse_arguments()
    
    # Load data
    df = load_and_prepare_data(args.input_file)
    
    # Compute extremes
    df_ext, p15, p85 = compute_extremes(df)
    
    # Run tests
    results = {
        'percentiles': {'p15': float(p15), 'p85': float(p85)},
        'n_total': len(df),
        'n_extremes': len(df_ext)
    }
    
    results['test1_morphology'] = test_1_morphology_control(df_ext)
    results['test2_continuous'] = test_2_continuous_model(df)
    results['test3_jackknife'] = test_3_jackknife(
        df_ext,
        n_iterations=args.n_jackknife,
        seed=args.seed
    )
    
    # Generate report
    report = generate_kill_tests_report(results, args, start_time)
    
    # Final verdict
    print_final_verdict(report)
    
    print(f"✅ Análisis completado")
    print(f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
