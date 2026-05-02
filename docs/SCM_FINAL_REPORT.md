# SCM — FINAL REPORT

## Sistema: Motor de Velos SCM

Framework empírico para la detección de transiciones de régimen estructural en sistemas astrofísicos.

---

## Resultado Principal

El sistema exhibe una bifurcación en una variable dinámica normalizada **x**:

- **x < 0** → régimen difusivo
- **x ≥ 0** → régimen estructurado con confinamiento cuadrático

Tras eliminar los efectos de masa, el término cuadrático mantiene significación estadística:

> p ≈ 5×10⁻⁴

---

## Interpretación Física

| Variable | Rol |
|---|---|
| Masa (logMbar / logMbh) | Establece la dinámica de base |
| Entorno / energía | Introduce un efecto de confinamiento de segundo orden |
| Término hinge H(x²) | Captura la asimetría estructurada |

La masa fija la línea base dinámica.  
El entorno introduce un efecto de confinamiento de segundo orden.

---

## Datasets

| Catálogo | N | Variable x | Variable y |
|---|---|---|---|
| SPARC nivel 2 | 50+ | grad_p (gradiente de presión normalizado) | v_res² (velocidad residual) |
| MOJAVE | 40+ | energy_proxy | residual_theta² |

---

## Método: Modelo Piecewise OLS con término Hinge

```
y = β₀ + β₁·x_res + β₂·H(x_res²) + ε
```

donde:
- `x_res` = variable estructural residualizada contra masa
- `H(x_res²) = x_res² · 1(x_res ≥ 0)` — término de confinamiento cuadrático

### Pasos

1. Normalización z-score de x e y por catálogo
2. Residualización de x contra masa (solo SPARC)
3. Ajuste OLS piecewise combinado
4. Test de permutación (N_PERM=500) sobre el término Hx2
5. Bootstrap de estabilidad del punto de bifurcación

---

## Criterio de Validación

| Criterio | Umbral | Resultado |
|---|---|---|
| p_perm (término Hx2) | < 0.05 | Ver `scm_xc_bootstrap_validation.json` |
| σ(bifurcación) | ≤ 0.15 | Ver `scm_xc_bootstrap_validation.json` |

---

## Reproducibilidad

```bash
python scripts/run_scm_full_pipeline.py
```

Requiere: `data/processed/scm_level2_thermo_features.csv` y `data/processed/scm_bh_regime_labeled_final.csv`

---

## Test de Independencia de Masa

```bash
python scripts/scm_mass_independence_test.py
```

Verifica que la señal de bifurcación sobrevive tras la residualización de masa.

---

## Archivos de Resultados

| Archivo | Contenido |
|---|---|
| `results/scm_unique_law_summary.json` | Coeficientes y p-valores del modelo único |
| `results/scm_xc_bootstrap_validation.json` | Validación bootstrap del punto de bifurcación |
| `results/scm_mass_independence_result.json` | Test de independencia de masa |

---

## Limitaciones

- Tamaño de muestra limitado (especialmente en el régimen de masa alta)
- Los proxies de entorno pueden ser incompletos o ruidosos
- El modelo lineal de base puede no capturar estructura de orden superior

---

## Cita

> DOI: 10.5281/zenodo.19897353

```
@misc{scm_motor_de_velos,
  author = {Cámara Madrid, Sergio},
  title  = {SCM — Motor de Velos},
  year   = {2026},
  doi    = {10.5281/zenodo.19897353},
  url    = {https://github.com/sergiocamaramadrid-cyber/Motor-de-Velos-SCM}
}
```
