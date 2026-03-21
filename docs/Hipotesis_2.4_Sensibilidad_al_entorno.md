Hipótesis derivada 2.4: Sensibilidad al entorno en función de la masa estelar

Enunciado
Si el gradiente de presión ambiental modula la turbulencia vertical en discos galácticos, las galaxias de baja masa —con pozos de potencial gravitatorio más someros— deberían ser más susceptibles a dicho gradiente. En estas galaxias, la presión externa puede perturbar con mayor facilidad la dinámica vertical del disco, amplificando el efecto sobre la dispersión vertical (σ_z) relativa a la rotación. Por tanto, el impacto del entorno sobre F₃ (o rec_slope) debe ser significativamente más intenso en el rango de baja masa estelar (log M_* < 9.0).

Observables

F₃ = σ_z / V_rot (observable primario)

rec_slope (observable secundario, cuando la relación radial sea más robusta)

log M_* (variable continua de estratificación)

Proxy ambiental: logΣ_HI_outer (principal) o δ (cuando esté disponible)

log SFR (control)


Predicción
Al estratificar la muestra en bins de masa estelar (bajo: log M_* < 9.0; medio: 9.0–9.8; alto: >9.8), el coeficiente de regresión del proxy ambiental (β₃ en el modelo completo) debe ser más negativo en el bin de baja masa que en los bins superiores. Esto implicaría que la presión externa contribuye de forma más pronunciada en galaxias de bajo potencial, un efecto diferencial detectable.

Estrategia estadística

1. Modelo con interacción
F₃ ∼ log M_* + log SFR + env + env × log M_*
(env = proxy ambiental estandarizado)
→ Si el término de interacción es significativo (p < 0.05) y negativo, indica que a menor masa el efecto ambiental se intensifica.


2. Análisis por bins

Dividir en terciles o cuartiles de masa estelar.

Ajustar el modelo completo (env + env² + env × log SFR) en cada bin.

Comparar β₃ mediante bootstrap (intervalos al 95%) o test de permutación cruzada.


Criterio de decisión
Evidencia a favor si:

El coeficiente de interacción env × log M_* es significativo (p < 0.05) con signo negativo.

O bien, en el bin de baja masa |β₃| es claramente mayor que en los bins altos, con intervalos al 95% que no se solapan (o p < 0.05 en permutación cruzada).


Valor añadido
Si se confirma, esta hipótesis fortalecería la relevancia del entorno como factor adicional en la regulación de la turbulencia vertical, especialmente en galaxias de bajo potencial. No implica descartar mecanismos internos (feedback, inestabilidades, etc.), sino que subraya una dependencia escalable con la masa estelar—un rasgo distintivo de modelos de presión externa y sus efectos ambientales. Si se validara, no solo se reforzaría la importancia del entorno en la dinámica de las galaxias de menor masa, sino que se abriría la posibilidad de extender la predicción: en entornos aún más densos, como grupos o cúmulos, el efecto diferencial podría ser incluso más marcado en galaxias satélite de baja masa.