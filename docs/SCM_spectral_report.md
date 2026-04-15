# Informe Técnico  
## Proceso de Extracción de Señales en el Framework SCM (Motor de Velos)

**Autor:** Sergio Cámara Madrid  
**Fecha:** 15 de abril de 2026  
**Repositorio:** https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM  
**DOI:** https://doi.org/10.5281/zenodo.19455777  

---

## 1. Resumen ejecutivo

El Framework SCM (Motor de Velos) ha sido desarrollado para estudiar la dinámica externa de galaxias. En fases previas se demostró que la pendiente externa de la curva de rotación (ΔF3) presenta una correlación negativa con el entorno (`env_proxy`) únicamente en galaxias de alta masa (logMbar ≥ 10.6), lo que indica la existencia de un umbral de masa crítico. Este resultado fue validado mediante bootstrap, permutación y tests de robustez.

En esta fase se ha explorado una segunda señal potencial: la presencia de estructuras periódicas (ondas) en las curvas de rotación, caracterizadas por una frecuencia dominante obtenida mediante análisis de Fourier (FFT). Se han procesado 525 curvas de rotación reales del catálogo SPARC, generando el archivo `sparc_frequency_analysis.csv` con la frecuencia dominante para cada galaxia.

El análisis incluye eliminación de tendencia polinómica, interpolación a malla uniforme, detección de picos espectrales y test de permutación para evaluar significancia.

La correlación de esta nueva métrica con la masa bariónica y el entorno no se ha completado por falta de acceso al archivo de metadatos en el entorno de trabajo. No obstante, el análisis espectral está finalizado y es completamente reproducible.

---

## 2. Metodología general del Framework SCM

El Framework SCM se basa en el observable:

$$
F_{\mathrm{SCM}} = \frac{d\log V}{d\log R}
$$

y su desviación respecto a un valor de referencia:

$$
\Delta F_3 = F_{\mathrm{SCM}} - 0.5
$$

La estrategia de análisis consiste en:

1. Controlar primero por masa bariónica (`logMbar`) mediante regresión.
2. Analizar los residuos para detectar modulación ambiental (`env_proxy`).
3. Dividir la muestra por umbral de masa para identificar regímenes.

---

## 3. Resultados en SPARC (señal ambiental)

- **Muestra:** 175 galaxias SPARC → 79 tras filtrado de calidad  
- **Umbral de masa crítico:** logMbar ≈ 10.6  

### Baja masa (logMbar &lt; 10.6)
- ρ ≈ 0.006  
- p ≈ 0.895  
→ Sin señal

### Alta masa (logMbar ≥ 10.6)
- ρ ≈ -0.44  
- p ≈ 0.001  
- β_env ≈ -0.112 (p = 0.010)

### Validación
- Bootstrap: mediana ρ ≈ -0.45  
  IC68% ≈ [-0.58, -0.29]  
- Permutación: p_emp ≈ 0.001  
- Outliers: ρ ≈ -0.47  

### Conclusión

La señal ambiental es robusta y emerge únicamente por encima de un umbral de masa.

---

## 4. Análisis espectral (FFT)

### Objetivo

Detectar periodicidades en las curvas de rotación que puedan reflejar estructura dinámica adicional.

---

### Pipeline aplicado

Para cada galaxia:

1. Carga de datos (R, Vobs)
2. Eliminación de tendencia (polinomio grado 2)
3. Interpolación a 512 puntos
4. FFT
5. Espectro de potencia
6. Detección de picos (> percentil 95)
7. Test de permutación (200 iteraciones)
8. Selección de frecuencia dominante

---

### Resultados

- **Curvas procesadas:** 525  
- **Archivo generado:** `sparc_frequency_analysis.csv`

Columnas:

- `galaxy`
- `dominant_freq` (1/kpc)
- `has_significant` (bool)

---

### Rango de frecuencias

- 0.02 → 1.73 (1/kpc)  
- Periodos: ~0.6 → 50 kpc  

---

### Interpretación

Muchas galaxias presentan picos significativos, lo que **sugiere la posible presencia de estructuras periódicas**, cuya naturaleza (física o instrumental) requiere validación adicional.

---

### Limitaciones del análisis espectral

- Muestreo radial irregular en SPARC  
- Extensión radial limitada  
- Sensibilidad al método de detrending  
- Posibilidad de artefactos FFT  

---

## 5. Estado actual

### Logros

- Señal ambiental validada (SCM v1.0)
- Pipeline espectral completo
- Datos procesados (525 galaxias)
- Script reproducible (`scm_spectral_analysis.py`)

---

### Pendiente

Correlacionar:

- `dominant_freq` vs `logMbar`
- `dominant_freq` vs `env_proxy`

---

### Nota crítica

El análisis espectral es actualmente **independiente del resultado principal SCM** y debe considerarse exploratorio hasta completar las correlaciones.

---

## 6. Próximos pasos

1. Recuperar metadatos (`scm_master_final.csv`)
2. Fusionar datasets:

```python
df = pd.merge(freq, meta, on="galaxy")
```

3. Calcular correlaciones
4. Visualizar
5. Evaluar si existe señal adicional

---

## 7. Conclusiones

- La señal ambiental está validada y es robusta
- Existe un umbral de masa crítico
- Se ha implementado un pipeline espectral completo
- Se detectan posibles estructuras periódicas
- La interpretación física de estas estructuras está abierta

---

## Estado del Framework SCM

- ✅ Señal ambiental validada
- ✅ Pipeline espectral implementado
- ⏳ Correlación espectral pendiente

---

## Conclusión final

El Framework SCM no es únicamente un resultado, sino un sistema reproducible capaz de seguir generando nuevas hipótesis observacionales.

---

*Fin del informe*
