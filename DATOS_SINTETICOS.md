# ⚠️ ADVERTENCIA SOBRE DATOS SINTÉTICOS

## Estado Actual del Análisis

**🔴 LOS RESULTADOS ACTUALES USAN DATOS SINTÉTICOS**

### ¿Qué significa esto?

Los scripts `scm_env_protocol.py` y `scm_robustness_tests.py` actualmente generan **datos de ejemplo sintéticos** cuando no encuentran el archivo `df_master.csv`.

### Banderas Rojas Identificadas:

1. **N_total = 500, N_used = 500**
   - ❌ Indica que NO se aplicaron percentiles 15/85
   - ✅ Debería ser ~150 galaxias (30% de la muestra)

2. **Errores estándar ~0.007**
   - ❌ Demasiado pequeños para datos reales
   - ✅ Datos reales tienen SE ~0.02-0.03

3. **Split perfecto 346/154 = 69.2%/30.8%**
   - ❌ Sospechosamente cerca de 70/30
   - ✅ Datos reales tienen variación natural

4. **Dispersión residual baja**
   - ❌ Los datos sintéticos tienen ruido controlado
   - ✅ Datos reales tienen más dispersión

## Cómo Usar con Datos Reales

### 1. Preparar tu archivo CSV

Tu `df_master.csv` debe contener:

```csv
log_mass,log_velocity,logSigma5,morph_type,is_barred,gas_fraction
9.5,2.3,1.2,3,0,0.15
10.2,2.5,0.8,5,1,0.22
...
```

**Columnas requeridas:**
- `log_mass`: log10(M*/Msun)
- `log_velocity`: log10(V_flat) en km/s
- `logSigma5`: log10(Sigma_5) - densidad ambiental

**Columnas opcionales (para robustez):**
- `morph_type`: Tipo morfológico T (0-10)
- `is_barred`: Barred (1) o unbarred (0)
- `gas_fraction`: Fracción de gas

### 2. Verificar la calidad de tus datos

```bash
python verify_data_quality.py df_master.csv
```

Este script verificará:
- ✅ Si los datos son reales o sintéticos
- ✅ Si se aplicaron correctamente los percentiles
- ✅ Consistencia de la clasificación ambiental
- ✅ Dispersión realista

### 3. Ejecutar el análisis

**Análisis ambiental básico:**
```bash
python scm_env_protocol.py df_master.csv
```

**Pruebas de robustez (Fase 2):**
```bash
python scm_robustness_tests.py df_master.csv
```

## Diferencias: Sintético vs Real

| Aspecto | Datos Sintéticos | Datos Reales (SPARC) |
|---------|------------------|----------------------|
| N total | 500 | ~175 |
| N extremos | 500 (100%) | ~50-60 (30%) |
| SE típico | 0.007 | 0.02-0.03 |
| Dispersión | Baja (~0.10) | Media (~0.15-0.20) |
| Δγ | ~0.041 | ¿? (a determinar) |
| p-valor | 6.77e-05 | ¿? (a determinar) |

## Próximos Pasos

### Si tienes datos reales de SPARC:

1. **Verifica tus datos:**
   ```bash
   python verify_data_quality.py tu_archivo.csv
   ```

2. **Si el verificador da ✅:**
   - Ejecuta el protocolo ambiental
   - Ejecuta las pruebas de robustez
   - Los resultados serán válidos

3. **Si el verificador da 🔴:**
   - Revisa el filtrado de percentiles
   - Verifica las transformaciones logarítmicas
   - Asegúrate que `logSigma5` es continua

### Si solo quieres probar la metodología:

Los datos sintéticos son **válidos para:**
- ✅ Probar que el código funciona
- ✅ Entender la metodología
- ✅ Desarrollo de nuevas pruebas

Los datos sintéticos **NO son válidos para:**
- ❌ Publicación científica
- ❌ Conclusiones sobre física real
- ❌ Comparación con literatura

## Contacto

Si tienes preguntas sobre:
- Formato de datos requerido
- Cómo aplicar percentiles correctamente
- Interpretación de resultados

Abre un issue en el repositorio.

---

**Versión:** 1.0  
**Última actualización:** 2026-02-19
