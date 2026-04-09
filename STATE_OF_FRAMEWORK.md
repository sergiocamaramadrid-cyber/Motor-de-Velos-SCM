# Informe de Estado del Framework SCM-Motor de Velos

**Fecha:** 9 de abril de 2026

---

## 1. Resumen ejecutivo

El Framework SCM-Motor de Velos se encuentra actualmente en un estado estable, modularizado y reproducible a nivel software. La base Python ha sido auditada y no presenta fallos bloqueantes ni duplicaciones involuntarias. La suite completa de pruebas pasa íntegramente con **574/574 tests aprobados**, lo que confirma la solidez operativa del framework y permite continuar el trabajo científico sin necesidad de una refactorización mayor.

La arquitectura actual ya no corresponde a un conjunto disperso de notebooks o scripts experimentales, sino a un framework estructurado por capas, con separación clara entre lógica de dominio, scripts de implementación, puntos de entrada canónicos y orquestadores de pipeline.

En consecuencia, el trabajo pendiente ya no es de saneamiento técnico del framework, sino de consolidación científica, documentación operativa y preparación de resultados para manuscrito.

---

## 2. Arquitectura actual

La estructura del proyecto queda organizada en tres niveles funcionales principales:

### 2.1. Módulos de dominio (`src/`)

Aquí reside la lógica de cálculo puro y análisis base del framework. Esta capa contiene el núcleo conceptual y computacional independiente de la ejecución operativa.

Ejemplos:

- `src/scm_analysis.py`
- `src/scm_models.py`
- `src/sensitivity.py`
- `src/lt/lt_dust_hinge_analysis.py`

### 2.2. Scripts de implementación (`scripts/`, capa 1)

Esta capa contiene las implementaciones principales de análisis aplicadas al pipeline SPARC/entorno:

- `sparc_slope_tail.py`
- `build_galaxy_catalog_env.py`
- `plot_sparc_slope_tail_hist.py`
- `plot_env_mass_scan.py`

Estas piezas representan la lógica operativa real de cálculo, fusión de catálogos y generación de figuras.

### 2.3. Entry-points canónicos (`scripts/`, capa 2)

Se han añadido puntos de entrada ligeros y normalizados que delegan en las implementaciones de la capa 1:

- `compute_slope_tail.py`
- `build_master_catalog.py`
- `mass_split_analysis.py`
- `env_mass_scan.py`

Esta decisión mejora la claridad de uso, permite una nomenclatura más limpia y facilita la ejecución reproducible desde pipeline o documentación.

### 2.4. Orquestadores (`scripts/`, capa 3)

Actualmente existen dos orquestadores:

- `run_full_pipeline.py` → **canónico**
- `run_pipeline.py` → **legado**

Ambos están testeados y son funcionales. La duplicación observada no es accidental: uno opera sobre entry-points y otro sobre implementaciones directas. No constituye un fallo estructural, aunque a medio plazo conviene dejar el estado "legacy" explícitamente documentado.

---

## 3. Estado técnico actual

### 3.1. Integridad del software

El framework presenta un estado técnico sólido:

| Métrica | Valor |
|---|---|
| Tests aprobados | **574 / 574** |
| Fallos | 0 |
| Fallos bloqueantes | 0 |
| Duplicaciones involuntarias | 0 |
| Regresiones conocidas | 0 |

Esto permite afirmar que la base software está saneada.

### 3.2. Duplicaciones detectadas

La única duplicación funcional aparente corresponde a la coexistencia entre:

- `run_pipeline.py`
- `run_full_pipeline.py`

Sin embargo, esta duplicación ha sido clasificada como **intencional y controlada**, no como deuda técnica crítica.

### 3.3. Cobertura y pruebas

La cobertura funcional está bien distribuida entre:

- módulos de análisis base
- scripts de SPARC
- construcción de catálogos
- escaneos ambientales
- puntos de entrada
- orquestadores

Quedan algunos scripts auxiliares sin cobertura directa, principalmente:

- `generate_env_figure.py`
- `download_sparc_data.py`

Su ausencia de test no compromete el núcleo del framework, pero constituye una mejora razonable para futuras iteraciones.

---

## 4. Estado de datos y reproducibilidad

Los datos reales de producción no están versionados dentro del repositorio, lo cual es correcto y esperable, dado que se trata de datasets externos o pesados. Entre ellos:

- `data/SPARC/rotmod/*.dat`
- `data/sparc_basic.csv`
- `data/env_proxy.csv`
- otros catálogos intermedios dependientes de análisis externos

Los tests usan datos sintéticos o fixtures de validación, por lo que la ausencia de los datasets reales en GitHub no afecta a la integridad de la suite.

Esto significa que el framework es **reproducible a nivel lógico y software**, aunque la reproducción completa del resultado científico requiere documentar claramente los archivos de entrada externos.

---

## 5. Estado científico-operativo

Desde el punto de vista operativo, el framework ya tiene cerrados los siguientes bloques:

- [x] cálculo de `slope_tail`
- [x] construcción del catálogo maestro con entorno
- [x] análisis de distribución para galaxias de alta masa
- [x] escaneo de correlación ambiental frente a umbral de masa
- [x] pipeline canónico de ejecución
- [x] validación por pruebas automatizadas

Por tanto, la base Python ya está preparada para sostener resultados científicos reproducibles y no necesita una nueva limpieza general antes de seguir avanzando.

---

## 6. Riesgos y deuda técnica residual

La deuda residual es menor y no impide continuar. Se concentra en:

### 6.1. Documentación operativa

Conviene fijar de forma explícita:

- qué pipeline se considera canónico
- qué datos externos requiere cada paso
- qué outputs son definitivos y cuáles auxiliares
- qué scripts quedan como legacy

### 6.2. Cobertura auxiliar

Sería útil añadir pruebas directas para:

- `generate_env_figure.py`
- `download_sparc_data.py`

### 6.3. Naming heredado

Algunos tests y nombres de archivo conservan nomenclatura histórica. No rompe nada, pero a largo plazo sería bueno armonizar nombres para reducir fricción documental.

---

## 7. Conclusión oficial

El Framework SCM-Motor de Velos está actualmente **limpio, estable y listo para continuar el trabajo científico**. La auditoría de la base Python no identifica fallos reales, duplicaciones involuntarias ni bloqueos de arquitectura. La suite completa pasa íntegramente y la estructura por capas ya está consolidada.

En consecuencia, el siguiente paso natural no es una nueva refactorización del framework, sino la consolidación de resultados, la documentación del flujo canónico y la preparación del material científico para manuscrito.

---

## 8. Recomendación de siguiente fase

Se recomienda abrir ahora una fase de trabajo centrada en cuatro objetivos:

1. fijar el pipeline canónico definitivo;
2. documentar los datasets de entrada requeridos;
3. consolidar los resultados científicos actuales en formato paper;
4. preparar un informe maestro del Framework para repositorio, Zenodo y manuscrito.

---

## 9. Frase breve de estado

> La base software del Framework SCM-Motor de Velos se encuentra estable, modularizada y reproducible, con **574/574 pruebas aprobadas** y sin fallos bloqueantes identificados. El trabajo pendiente ya no es de saneamiento técnico, sino de consolidación científica y documentación operativa.
