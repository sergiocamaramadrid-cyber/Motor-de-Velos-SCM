#!/usr/bin/env python3
"""
verify_data_quality.py

Script para verificar si los datos son reales o sintéticos,
y si se aplicaron correctamente los percentiles ambientales.

Uso:
    python verify_data_quality.py df_master.csv

Autor: Motor-de-Velos-SCM team
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path


def verify_data_source(df):
    """
    Detectar si los datos son sintéticos o reales.
    
    Banderas de datos sintéticos:
    - Errores estándar muy pequeños (~0.007)
    - Distribuciones demasiado perfectas
    - Valores exactamente redondos
    """
    print("\n" + "="*80)
    print("🔍 VERIFICACIÓN DE ORIGEN DE DATOS")
    print("="*80)
    
    flags = []
    
    # Check 1: Sample size
    n = len(df)
    print(f"\n   Tamaño de muestra: {n}")
    
    if n == 500:
        flags.append("⚠️  N=500 exacto (típico de datos sintéticos)")
    
    # Check 2: Environmental variable statistics
    if 'logSigma5' in df.columns:
        sigma_mean = df['logSigma5'].mean()
        sigma_std = df['logSigma5'].std()
        print(f"\n   logSigma5 estadísticas:")
        print(f"      Media: {sigma_mean:.6f}")
        print(f"      Std:   {sigma_std:.6f}")
        
        # Synthetic data often has very specific distributions
        if abs(sigma_mean - 1.0) < 0.1 and abs(sigma_std - 0.5) < 0.1:
            flags.append("⚠️  logSigma5 tiene distribución típica de sintético")
    
    # Check 3: Perfect splits
    if 'env_class' in df.columns:
        n_nucleo = (df['env_class'] == 0).sum()
        n_borde = (df['env_class'] == 1).sum()
        ratio = n_borde / n if n > 0 else 0
        
        print(f"\n   Clasificación ambiental:")
        print(f"      Núcleo: {n_nucleo}")
        print(f"      Borde:  {n_borde}")
        print(f"      Ratio:  {ratio:.3f}")
        
        # Check for suspiciously perfect ratios
        perfect_ratios = [0.5, 0.6, 0.7, 0.8]
        if any(abs(ratio - pr) < 0.02 for pr in perfect_ratios):
            flags.append(f"⚠️  Ratio {ratio:.3f} sospechosamente perfecto")
    
    # Check 4: Velocity errors
    if 'log_velocity' in df.columns and 'log_mass' in df.columns:
        # Simple linear fit to check residuals
        from sklearn.linear_model import LinearRegression
        X = df[['log_mass']].values
        y = df['log_velocity'].values
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)
        residual_std = np.std(residuals)
        
        print(f"\n   Dispersión residual:")
        print(f"      Std residuos: {residual_std:.6f}")
        
        if residual_std < 0.15:
            flags.append(f"⚠️  Dispersión muy baja ({residual_std:.3f}) - típico de sintético")
    
    # Verdict
    print(f"\n   {'='*76}")
    if len(flags) == 0:
        print("   ✅ DATOS PARECEN REALES")
        print("   No se detectaron banderas de datos sintéticos")
        is_synthetic = False
    elif len(flags) <= 1:
        print("   ⚠️  POSIBLEMENTE REALES CON ADVERTENCIAS")
        for flag in flags:
            print(f"   {flag}")
        is_synthetic = False
    else:
        print("   🔴 DATOS PROBABLEMENTE SINTÉTICOS")
        for flag in flags:
            print(f"   {flag}")
        is_synthetic = True
    
    return is_synthetic, flags


def verify_percentile_filtering(df):
    """
    Verificar si se aplicaron correctamente los percentiles 15/85.
    """
    print("\n" + "="*80)
    print("🔍 VERIFICACIÓN DE FILTRADO POR PERCENTILES")
    print("="*80)
    
    if 'logSigma5' not in df.columns:
        print("\n   ⚠️  Columna 'logSigma5' no encontrada")
        print("   No se puede verificar filtrado de percentiles")
        return None
    
    # Statistics
    print(f"\n   Estadísticas de logSigma5:")
    print(df['logSigma5'].describe())
    
    # Percentiles
    p15 = df['logSigma5'].quantile(0.15)
    p85 = df['logSigma5'].quantile(0.85)
    
    print(f"\n   Percentiles:")
    print(f"      P15 (low density):  {p15:.6f}")
    print(f"      P85 (high density): {p85:.6f}")
    
    # Count extremes
    n_low = (df['logSigma5'] <= p15).sum()
    n_high = (df['logSigma5'] >= p85).sum()
    n_extremes_total = n_low + n_high
    
    print(f"\n   Conteo de extremos:")
    print(f"      N ≤ P15: {n_low}")
    print(f"      N ≥ P85: {n_high}")
    print(f"      Total extremos: {n_extremes_total}")
    print(f"      % de muestra: {100*n_extremes_total/len(df):.1f}%")
    
    # Verdict
    print(f"\n   {'='*76}")
    
    expected_extremes = int(0.30 * len(df))  # Should be ~30%
    
    if abs(n_extremes_total - len(df)) < 10:
        print("   🔴 ERROR: Se están usando TODAS las galaxias")
        print("   Los percentiles NO se aplicaron correctamente")
        print(f"   Esperado: ~{expected_extremes} galaxias")
        print(f"   Obtenido: {n_extremes_total} galaxias (≈100%)")
        correctly_filtered = False
    elif abs(n_extremes_total - expected_extremes) < 50:
        print("   ✅ CORRECTO: Se aplicaron los percentiles")
        print(f"   Extremos: {n_extremes_total} (~30% de {len(df)})")
        correctly_filtered = True
    else:
        print("   ⚠️  ADVERTENCIA: Número de extremos inesperado")
        print(f"   Esperado: ~{expected_extremes}")
        print(f"   Obtenido: {n_extremes_total}")
        correctly_filtered = False
    
    return {
        'p15': p15,
        'p85': p85,
        'n_low': n_low,
        'n_high': n_high,
        'n_extremes': n_extremes_total,
        'correctly_filtered': correctly_filtered
    }


def verify_environmental_classification(df):
    """
    Verificar cómo se hizo la clasificación ambiental.
    """
    print("\n" + "="*80)
    print("🔍 VERIFICACIÓN DE CLASIFICACIÓN AMBIENTAL")
    print("="*80)
    
    if 'env_class' not in df.columns:
        print("\n   ⚠️  Columna 'env_class' no encontrada")
        return None
    
    if 'logSigma5' not in df.columns:
        print("\n   ⚠️  Columna 'logSigma5' no encontrada")
        print("   No se puede verificar consistencia con densidad")
        return None
    
    # Check consistency
    print("\n   Verificando consistencia env_class vs logSigma5:")
    
    mean_sigma_nucleo = df[df['env_class'] == 0]['logSigma5'].mean()
    mean_sigma_borde = df[df['env_class'] == 1]['logSigma5'].mean()
    
    print(f"      logSigma5 medio (núcleo): {mean_sigma_nucleo:.6f}")
    print(f"      logSigma5 medio (borde):  {mean_sigma_borde:.6f}")
    
    # Núcleo debería tener MAYOR densidad (mayor logSigma5)
    if mean_sigma_nucleo > mean_sigma_borde:
        print("\n   ✅ CONSISTENTE: Núcleo tiene mayor densidad")
        consistent = True
    else:
        print("\n   🔴 INCONSISTENTE: Borde tiene mayor densidad")
        print("   La clasificación puede estar invertida")
        consistent = False
    
    return {
        'mean_sigma_nucleo': mean_sigma_nucleo,
        'mean_sigma_borde': mean_sigma_borde,
        'consistent': consistent
    }


def generate_verification_code(filename):
    """
    Generar código Python para verificación manual.
    """
    print("\n" + "="*80)
    print("📝 CÓDIGO DE VERIFICACIÓN MANUAL")
    print("="*80)
    
    code = f"""
# Ejecuta esto en Python para verificar tus datos:

import pandas as pd
df = pd.read_csv('{filename}')

print("=== VERIFICACIÓN DE DATOS ===")
print(f"N total: {{len(df)}}")

if 'logSigma5' in df.columns:
    print("\\nEstadísticas logSigma5:")
    print(df[['logSigma5']].describe())
    
    p15 = df['logSigma5'].quantile(0.15)
    p85 = df['logSigma5'].quantile(0.85)
    
    print(f"\\nPercentil 15: {{p15:.6f}}")
    print(f"Percentil 85: {{p85:.6f}}")
    
    n_low = (df['logSigma5'] <= p15).sum()
    n_high = (df['logSigma5'] >= p85).sum()
    
    print(f"\\nN < P15: {{n_low}}")
    print(f"N > P85: {{n_high}}")
    print(f"Total extremos: {{n_low + n_high}} ({{100*(n_low+n_high)/len(df):.1f}}%)")
    
    if (n_low + n_high) >= 0.9 * len(df):
        print("\\n🔴 ERROR: Estás usando casi todas las galaxias!")
        print("Los percentiles NO se aplicaron correctamente.")
    else:
        print("\\n✅ OK: Percentiles aplicados correctamente.")

if 'log_velocity' in df.columns and 'log_mass' in df.columns:
    from sklearn.linear_model import LinearRegression
    import numpy as np
    
    X = df[['log_mass']].values
    y = df['log_velocity'].values
    model = LinearRegression().fit(X, y)
    residuals = y - model.predict(X)
    
    print(f"\\nStd residuos: {{np.std(residuals):.6f}}")
    
    if np.std(residuals) < 0.15:
        print("🔴 ADVERTENCIA: Dispersión muy baja (posible sintético)")
    else:
        print("✅ OK: Dispersión realista")
"""
    
    print(code)
    
    # Save to file
    output_file = Path('verify_data.py')
    with open(output_file, 'w') as f:
        f.write(code)
    
    print(f"\n   ✓ Código guardado en: {output_file}")
    print(f"   Ejecuta: python {output_file}")


def main():
    """Función principal."""
    if len(sys.argv) < 2:
        print("Uso: python verify_data_quality.py df_master.csv")
        return 1
    
    input_file = sys.argv[1]
    
    print("\n" + "="*80)
    print("🔬 VERIFICADOR DE CALIDAD DE DATOS")
    print("="*80)
    print(f"Archivo: {input_file}")
    
    # Check if file exists
    if not Path(input_file).exists():
        print(f"\n❌ ERROR: Archivo no encontrado: {input_file}")
        print("\nNOTA: El análisis actual está usando DATOS SINTÉTICOS")
        print("generados automáticamente por create_sample_data().")
        print("\nPara análisis real, proporciona un archivo CSV con:")
        print("  - log_mass")
        print("  - log_velocity")
        print("  - logSigma5 (densidad ambiental)")
        return 1
    
    # Load data
    try:
        df = pd.read_csv(input_file)
        print(f"✓ Datos cargados: {len(df)} galaxias")
        print(f"✓ Columnas: {df.columns.tolist()}")
    except Exception as e:
        print(f"\n❌ ERROR al cargar datos: {e}")
        return 1
    
    # Run verifications
    is_synthetic, flags = verify_data_source(df)
    percentile_info = verify_percentile_filtering(df)
    env_info = verify_environmental_classification(df)
    
    # Generate verification code
    generate_verification_code(input_file)
    
    # Final summary
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL")
    print("="*80)
    
    if is_synthetic:
        print("\n🔴 DATOS SINTÉTICOS DETECTADOS")
        print("\nLos resultados del análisis NO son válidos para publicación.")
        print("Son útiles solo para pruebas y desarrollo del método.")
    else:
        if percentile_info and not percentile_info['correctly_filtered']:
            print("\n⚠️  DATOS REALES PERO FILTRADO INCORRECTO")
            print("\nDebes aplicar correctamente los percentiles 15/85")
            print("antes de ejecutar el análisis ambiental.")
        else:
            print("\n✅ DATOS PARECEN VÁLIDOS")
            print("\nPuedes proceder con confianza al análisis.")
    
    print("="*80 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
