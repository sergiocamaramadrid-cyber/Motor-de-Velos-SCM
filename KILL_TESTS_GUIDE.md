# Kill Tests - Guía de Uso

## Objetivo

Los **kill tests** son tres pruebas diseñadas para **intentar refutar** el efecto ambiental detectado en Phase 1. Si el efecto sobrevive los tres tests, es robusto y publicable. Si falla alguno, queda refutado.

## Prerrequisitos

### Datos Requeridos

Tu archivo `df_master.csv` debe contener las siguientes columnas:

| Columna | Descripción | Tipo | Rango típico |
|---------|-------------|------|--------------|
| `log_mbar` | log₁₀(M_bar/M☉) | float | 8.5-11.5 |
| `log_vflat` | log₁₀(V_flat [km/s]) | float | 1.5-2.8 |
| `logSigma5` | log₁₀(Σ₅ [gal/Mpc²]) | float | -1.0-2.0 |
| `T` | Tipo de Hubble | int | -5 a 10 |
| `bar` | Presencia de barra | int | 0 o 1 |
| `incl` | Inclinación (grados) | float | 0-90 |

### Instalación

```bash
pip install pandas numpy scipy statsmodels scikit-learn
```

## Ejecución

### Con tus datos reales:

```bash
python scm_kill_tests.py df_master.csv
```

### Con datos de prueba:

```bash
# Generar datos realistas de prueba
python create_realistic_test_data.py

# Ejecutar kill tests
python scm_kill_tests.py df_master_realistic.csv
```

## Los Tres Tests

### Test 1: Control por Morfología

**Pregunta:** ¿El efecto es realmente ambiental o solo tipo galáctico?

**Método:**
```python
Model: log_mbar ~ log_vflat * is_nuc + C(morph_bin) + bar
```

**Categorización:**
- Early: T < 2
- Intermediate: 2 ≤ T < 6
- Late: T ≥ 6

**Criterio de éxito:** 
- Interacción `log_vflat:is_nuc` debe tener p < 0.05

**Interpretación:**
- ✅ **PASA**: El efecto persiste controlando morfología → es independiente del tipo galáctico
- ❌ **FALLA**: El efecto desaparece → era morfología disfrazada

---

### Test 2: Modelo Continuo

**Pregunta:** ¿La pendiente cambia suavemente con densidad o es un artefacto de dicotomizar?

**Método:**
```python
Model: log_mbar ~ log_vflat + logSigma5 + log_vflat:logSigma5
```

**Muestra:** Usa TODA la muestra (no solo extremos)

**Criterio de éxito:**
- Interacción `log_vflat:logSigma5` debe tener p < 0.05

**Interpretación:**
- ✅ **PASA**: Hay cambio continuo de pendiente con densidad
- ❌ **FALLA**: No hay efecto continuo → efecto discreto o ausente

---

### Test 3: Jackknife (Estabilidad)

**Pregunta:** ¿El efecto es robusto o depende de unos pocos puntos influyentes?

**Método:**
- 100 iteraciones
- Cada iteración: quitar 10% aleatorio
- Ajustar modelo base en cada submuestra

**Criterio de éxito:**
- Δγ > 0 en ≥95% de iteraciones
- p < 0.05 en ≥80% de iteraciones

**Interpretación:**
- ✅ **PASA**: El efecto es estructural y robusto
- ❌ **FALLA**: El efecto es frágil, depende del muestreo específico

## Interpretación de Resultados

### Veredicto Global

**Para que el efecto sea ROBUSTO y PUBLICABLE:**
- Debe PASAR los 3 tests (3/3)

**Si falla 1 o más tests:**
- El efecto ambiental queda REFUTADO
- No es publicable como efecto ambiental real
- Puede ser: confusión morfológica, artefacto estadístico, o puntos influyentes

### Ejemplo de Output Exitoso:

```
Tests completados: 3/3

📋 Resultados:
   1. Control morfológico: ✅ PASA (p=0.012)
   2. Modelo continuo: ✅ PASA (p=0.008)  
   3. Jackknife: ✅ PASA (98% positivo, 94% significativo)

🎯 VEREDICTO: ✅ EL EFECTO AMBIENTAL SOBREVIVE

El efecto es ROBUSTO y PUBLICABLE.
```

### Ejemplo de Output Fallido:

```
Tests completados: 0/3

📋 Resultados:
   1. Control morfológico: ❌ FALLA (p=0.866)
   2. Modelo continuo: ❌ FALLA (p=0.393)
   3. Jackknife: ❌ FALLA (25% positivo, 0% significativo)

🎯 VEREDICTO: ❌ EL EFECTO NO SOBREVIVE

El efecto ambiental queda REFUTADO.
```

## Archivos de Salida

### JSON Report: `scm_env_protocol_out/kill_tests_report.json`

```json
{
  "metadata": {
    "analysis_type": "kill_tests",
    "timestamp": "...",
    "input_file": "df_master.csv"
  },
  "test_results": {
    "test1_morphology": {
      "interaction_coef": 0.032,
      "interaction_pval": 0.012,
      "survives": true
    },
    "test2_continuous": {...},
    "test3_jackknife": {...}
  },
  "overall_assessment": {
    "tests_passed": 3,
    "tests_total": 3,
    "overall_survives": true
  }
}
```

## Troubleshooting

### Error: "Archivo no encontrado"

El script requiere datos REALES. Usa `create_realistic_test_data.py` para generar datos de prueba.

### Error: "Faltan columnas requeridas"

Verifica que tu CSV tiene: `log_mbar`, `log_vflat`, `logSigma5`, `T`, `bar`

### Warning: "Usando >90% de la muestra"

Tus percentiles pueden estar mal. Verifica que `logSigma5` sea continua y tenga variación real.

### Coeficientes nan o errores de convergencia

- Verifica que no haya NaN/Inf en tus datos
- Asegúrate de tener suficientes galaxias en cada categoría morfológica
- Revisa que la variable `bar` sea binaria (0/1)

## Próximos Pasos

### Si el efecto SOBREVIVE (3/3):

1. Documenta los resultados en el paper
2. Incluye las tres tablas de regresión
3. Menciona que pasó controles robustos
4. Discute implicaciones físicas

### Si el efecto NO SOBREVIVE:

1. **NO publiques** como efecto ambiental real
2. Analiza qué test falló y por qué
3. Considera análisis alternativos
4. Reporta honestamente la refutación

## Referencias

- HC3: MacKinnon & White (1985)
- Jackknife: Efron & Gong (1983)
- Statsmodels: Seabold & Perktold (2010)

---

**Versión:** 1.0  
**Última actualización:** 2026-02-19
