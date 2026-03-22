**Hipótesis derivada 2.5: Consistencia entre proxies ambientales y mejora con δ**

**Enunciado**  
Si el gradiente de presión ambiental modula F₃, los proxies disponibles en SPARC (logΣ_HI_outer y LSB/HSB) deben mostrar tendencias coherentes con la señal esperada de δ (sobredensidad local). Al incorporar δ en submuestras cross-match (SDSS o BIG-SPARC), la señal debería fortalecerse en términos de ajuste estadístico y estabilidad de los coeficientes.

**Observables**  
Nota: ρ_local puede estimarse operativamente en SDSS mediante conteo de vecinos dentro de ~3 Mpc (con corte en velocidad) o mediante reconstrucciones de campo de densidad como NEXUS+ o DisPerSE.

- **Target**: F₃ (primario) o σ_int / V_flat (secundario)  
- **Proxies**:  
  - logΣ_HI_outer (continuo)  
  - LSB/HSB (binaria: 0=LSB, 1=HSB; proxy cualitativo)  
- **Controles**: log M_*, log SFR  
- **δ**: (ρ_local / ⟨ρ⟩) − 1 (escala ~3 Mpc; SDSS/NEXUS+/DisPerSE)

**Signo esperado (condicional)**  
- δ alto (filamentos/nodos) → mayor F₃  
- logΣ_HI_outer: se espera correlación negativa con F₃ si refleja stripping o menor gas en entornos densos (puede variar según definición operacional)  
- LSB/HSB: LSB → F₃ menor; HSB → F₃ mayor

**Predicciones**  
1. **Coherencia de signo**: signo(β_δ) consistente con el de los proxies  
2. **Mejora con δ**: modelos con δ muestran menor AIC, mayor estabilidad de coeficientes y mayor distancia de β respecto a 0  
3. **Comparación de efecto (expectativa, no requisito rígido)**: |β_δ| ≥ |β_proxy| en variables estandarizadas, sujeto a ruido estadístico y colinealidad  
4. **Contingencia proxy–δ**: Si Spearman |ρ(proxy, δ)| < 0.3, la señal ambiental debe recuperarse al usar δ; LSB/HSB actúa como proxy cualitativo independiente

**Estrategia estadística**  
1. Correlación proxy–δ: Spearman + bootstrap (IC 95%)  
2. Modelos anidados (errores robustos HC3):  
   - Base: F₃ ~ log M_* + log SFR  
   - Proxy: + proxy  
   - δ: + δ  
3. Bootstrap (≥1000 iteraciones): distribución de ΔAIC y estabilidad de coeficientes  
4. Permutación LSB/HSB: diferencia de medias dentro de bins de masa

**Criterio de decisión**  
**Evidencia fuerte** si:  
- coherencia de signo  
- p < 0.05 en δ  
- ΔAIC(δ) > ΔAIC(proxy) + 2 en mayoría de bootstrap  
- β_δ más estable y alejado de 0  

**Evidencia moderada** si:  
- coherencia de signo  
- mejora en AIC sin superar umbral +2  
- intervalos parcialmente solapados  

**No validada** si: signo incoherente o inestable

**Valor añadido**  
- Convierte la limitación (falta de δ en SPARC) en un test de robustez  
- Proporciona validación cruzada independiente  
- Introduce calibración futura de proxies  
- Refuerza la consistencia interna del framework