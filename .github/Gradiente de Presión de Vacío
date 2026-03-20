# Hipótesis de Gradiente de Presión de Vacío – 4.0

**Status:** PRE-VALIDATION (SPARC execution pending)

---

## Enunciado

Planteamos que la evacuación coherente de los voids genera outflows radiales y un gradiente efectivo de presión del medio intergaláctico entre entornos de muy baja y alta densidad.

Este gradiente modula la turbulencia y la persistencia dinámica en discos galácticos, produciendo diferencias sistemáticas en:

- `rec_slope`
- `F₃ = σ_z / V_rot`

entre galaxias void-like y galaxias en filamentos.

---

## Observables

- `rec_slope = Δ log V_flat / Δ log σ_int`
- `F₃ = σ_z / V_rot`
- `δ = (ρ_local / ⟨ρ⟩) − 1`
- Proxy entorno (Fase A) = `logSigmaHI_outer` + clasificación LSB/aislada

---

## Predicción principal

A igualdad aproximada de masa y SFR:

- Galaxias en entornos de **baja densidad** → menor `rec_slope` y menor `F₃`
- Galaxias en entornos **densos** → mayor `rec_slope` y mayor `F₃`

Estas diferencias se plantean como **tendencias estadísticas**, no como umbrales rígidos.

---

## Estrategia estadística

Modelos anidados:

1. **Modelo base**  
   `rec_slope ~ log M_* + log SFR`

2. **Modelo intermedio**  
   `rec_slope ~ log M_* + log SFR + δ`

3. **Modelo completo**  
   `rec_slope ~ log M_* + log SFR + δ + δ² + δ × log SFR`

Condiciones:

- Predictores continuos estandarizados (media 0, desviación estándar 1)
- Errores robustos (HC3)
- Comparación de modelos mediante **AIC** (principal) y **BIC**

---

## Falsabilidad

- Permutación estratificada por bins de masa (manteniendo distribución de SFR)
- Bootstrap
- Evaluación de robustez

**Hipótesis nula (H₀):** el entorno no aporta información adicional sobre `rec_slope`.

---

## Criterio de decisión

Se considerará evidencia compatible con modulación ambiental si:

- `ΔAIC > 2` entre modelo con entorno y modelo base
- El coeficiente asociado a `δ` es significativo y con el signo esperado

---

## Fases

- **Fase A (SPARC):** uso de proxies de entorno (LSB / aisladas y `logSigmaHI_outer`)
- **Fase B (BIG-SPARC):** extensión a muestra ampliada
- **Fase C (TNG300):** contraste mecanístico con simulaciones

En la Fase A, `logSigmaHI_outer` se emplea como proxy de entorno en ausencia de mediciones directas de densidad (`δ`). En fases posteriores se utilizará `δ` explícito para validar la consistencia del efecto.

---

## Interpretación (si se confirma)

Resultados positivos serían consistentes con:

- Modulación ambiental de la dinámica galáctica
- Gradiente efectivo de presión entre voids y filamentos
- Diferencias en confinamiento y reciclaje de gas

---

## Parametrización fenomenológica (ansatz)

\[
\frac{\sigma_z}{V_{\text{rot}}} = \kappa \left(\frac{P_{\text{IGM}}}{P_0}\right)^\beta \left(1 + \frac{\dot{M}_{\text{infall}}}{\dot{M}_*}\right)^\gamma
\]

Interpretado como una **parametrización fenomenológica**, no como una ley establecida.

---

## Resumen

Si el entorno influye en la dinámica galáctica, debe manifestarse como una variación sistemática de `rec_slope` y `F₃` con la densidad ambiental.

Esta hipótesis es:

- Medible
- Falsable
- Operativa con datos actuales (SPARC)
- Extensible a datasets mayores y simulaciones
