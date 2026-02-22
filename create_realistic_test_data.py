#!/usr/bin/env python3
"""
create_realistic_test_data.py

Crear datos de prueba realistas con las columnas requeridas para kill tests.

Esto simula datos tipo SPARC con morfología y estructura ambiental realista.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# Número de galaxias (SPARC tiene ~175)
n_galaxies = 175

# T-type (Hubble type): -5 a 10
# Distribución realista: más galaxias tardías
T = np.random.choice(
    range(-5, 11),
    size=n_galaxies,
    p=[0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.02, 0.01, 0.005, 0.005]
)

# bar: presencia de barra (0/1)
# ~40% de galaxias tienen barra
bar = np.random.choice([0, 1], size=n_galaxies, p=[0.6, 0.4])

# incl: inclinación en grados (0-90)
# Evitar muy face-on (i < 30) por problemas de medición
incl = np.random.uniform(30, 80, n_galaxies)

# logSigma5: densidad ambiental
# Distribución bimodal: núcleo vs borde
env_type = np.random.choice([0, 1], size=n_galaxies, p=[0.35, 0.65])  # 0=núcleo, 1=borde

logSigma5 = np.zeros(n_galaxies)
for i in range(n_galaxies):
    if env_type[i] == 0:  # Núcleo - alta densidad
        logSigma5[i] = np.random.normal(1.3, 0.35)
    else:  # Borde - baja densidad
        logSigma5[i] = np.random.normal(0.4, 0.45)

# log_mass: masa estelar (SPARC range ~8.5-11.5)
log_mass = np.random.uniform(8.5, 11.3, n_galaxies)

# log_vflat: velocidad plana
# Relación masa-velocidad con efecto ambiental
gamma_base = 0.26
gamma_env_effect = 0.032  # Efecto ambiental real
gamma_morph_effect = 0.008  # Efecto morfológico menor

log_vflat = np.zeros(n_galaxies)
for i in range(n_galaxies):
    gamma_i = gamma_base
    
    # Efecto ambiental (núcleo vs borde)
    if env_type[i] == 1:  # Borde
        gamma_i += gamma_env_effect
    
    # Efecto morfológico pequeño
    gamma_i += gamma_morph_effect * (T[i] + 5) / 15.0  # Normalizado
    
    # Dispersión mayor en ambientes de baja densidad
    if env_type[i] == 0:  # Núcleo
        log_vflat[i] = gamma_i * log_mass[i] + np.random.normal(0, 0.09)
    else:  # Borde
        log_vflat[i] = gamma_i * log_mass[i] + np.random.normal(0, 0.14)

# log_mbar: masa bariónica (masa estelar + gas)
# Para simplificar, asumimos gas fraction pequeña
gas_fraction = 0.05 + 0.15 * (T + 5) / 15.0 + np.random.normal(0, 0.03, n_galaxies)
gas_fraction = np.clip(gas_fraction, 0.01, 0.35)

log_mbar = np.log10(10**log_mass * (1 + gas_fraction))

# Create DataFrame
df = pd.DataFrame({
    'log_mbar': log_mbar,
    'log_vflat': log_vflat,
    'logSigma5': logSigma5,
    'T': T,
    'bar': bar,
    'incl': incl,
    'log_mass': log_mass,
    'gas_fraction': gas_fraction
})

# Save
output_file = 'df_master_realistic.csv'
df.to_csv(output_file, index=False)

print(f"✓ Datos realistas guardados en: {output_file}")
print(f"  N = {len(df)} galaxias")
print(f"\nEstadísticas:")
print(df[['log_mbar', 'log_vflat', 'logSigma5', 'T']].describe())
print(f"\nPercentiles logSigma5:")
print(f"  P15 = {df['logSigma5'].quantile(0.15):.3f}")
print(f"  P85 = {df['logSigma5'].quantile(0.85):.3f}")
print(f"\nDistribución T-type:")
print(df['T'].value_counts().sort_index())
print(f"\nDistribución bar:")
print(f"  No barred: {(df['bar']==0).sum()}")
print(f"  Barred: {(df['bar']==1).sum()}")
