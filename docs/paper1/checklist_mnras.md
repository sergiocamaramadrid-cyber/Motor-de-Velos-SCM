# Checklist MNRAS — SCM paper (envío final)

## Estructura

- [ ] `figures/` con PNG + PDF (6 archivos: fig01–fig03 × 2 formatos)
- [ ] `ms_scm.tex` compila sin warnings críticos
- [ ] Todas las referencias `\ref{}` y `\label{}` resueltas
- [ ] Numeración de figuras y tablas consistente con el texto

## Ciencia

- [ ] **Tabla OOS** (Tabla 4) incluida con p-value, n_improved, ΔRMSE_out
- [ ] **Discussion** sin claims fuertes sobre microphysics
- [ ] **Conclusions** fenomenológicas (F3 term, no dark matter claim)
- [ ] Abstract alineado con resultados numéricos bloqueados:
  - p = 1.19 × 10⁻¹⁰
  - 53/53 galaxias mejoradas
  - ΔRMSE_out ≈ −7.3

## Archivos a adjuntar en MNRAS submission system

- [ ] PDF final del manuscrito (`ms_scm.pdf`)
- [ ] Archivos de figuras (PDF preferido por MNRAS):
  - `figures/figure01_scatter.pdf`
  - `figures/figure02_delta_rmse_hist.pdf`
  - `figures/figure03_delta_rmse_scatter.pdf`
- [ ] (Opcional) CSV de resultados OOS como material suplementario

## Cover letter (puntos clave)

- [ ] Mencionar validación OOS estricta (70/30 galaxy-level split)
- [ ] Mencionar que el resultado es robusto a semilla (seeds 42–44)
- [ ] Enfatizar que el análisis es fenomenológico
- [ ] Indicar repo público con código reproducible

## Repo GitHub antes de enviar

- [ ] Branch `main` limpio (sin archivos de Carnac ni referencias cruzadas)
- [ ] `README.md` actualizado con instrucciones de reproducibilidad
- [ ] Tag `v1.0-submission` creado tras el envío
