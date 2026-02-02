
# Optimización de Sistema de Notas
## Hallazgos y Documentación Metodológica

Este documento resume la metodología, los modelos matemáticos y los hallazgos derivados de los experimentos de optimización de pesos de notas para minimizar la fracción de falsos positivos y negativos.

## Formulación del problema

En muchos cursos se utilizan sistemas de notas basados en promedios ponderados. Se definen un conjunto de actividades, cada una con un peso (porcentaje). Algunas actividades pueden ser "fáciles" y otras más avanzadas. Normalmente, las actividades más avanzadas tienen un mayor peso. También ocurre que en algunos cursos existen evaluaciones que son "definitorias". Así por ejemplo un curso puede tener un examen final que es definitorio y que tiene un peso mayor que las otras actividades. 

Sin embargo, este sistema puede no ser el más adecuado para evaluar el rendimiento de los estudiantes. Sucede a veces que estudiantes que han tenido un buen desempeño en la mayoría de las actividades, reprueban el curso debido a una mala evaluación en una actividad definitoria (llamaremos a este tipo de estudiantes falsos negativos). Por el contrario, puede suceder que estudiantes que han tenido un mal desempeño en la mayoría de las actividades, aprueben el curso debido a una buena evaluación en una actividad definitoria (llamaremos a este tipo de estudiantes falsos positivos). 

Por esta razón, se realizó una exploración para evaluar si era posible diseñar un sistema de evaluación, un modelo de cálculo de la nota final, que minimizara la fracción de falsos positivos y negativos. 

A estos sistemas de evaluación, que tienen en cuenta el desempeño del estudiante en todas las actividades más que simplemente el cálculo de un promedio ponderado, se los llama *sistemas de evaluación compensatorios*. Existen algunos trabajos en la literatura sobre el tema, por ejemplo: 

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4640898 
- https://www.researchgate.net/publication/308626958_Resitting_or_compensating_a_failed_examination_does_it_affect_subsequent_results

Para una investigación más detallada sobre el tema, se puede consultar https://chatgpt.com/s/t_697e4bba377c81919ae10660a757e589. 

## Modelo Matemático

La evaluación de un curso implica obtener una nota final a partir de las notas obtenidas en diferentes actividades. Definamos primero las siguientes variables: 

*   $X_i$: Nota del estudiante en el item $i$.
*   $W_i$: Peso del item $i$. También llamado ponderación o porcentaje. 
*   $U_i$: Umbral de rendimiento para el item $i$. El umbral es la nota mínima que debe obtener un estudiante en un item para que se considere que ha aprobado el item. Por defecto, el umbral es 3.0. Sin embargo, se puede configurar un umbral diferente para cada item. Por ejemplo, en un curso puede ser que el examen final sea definitorio y que tenga un peso mayor que las otras actividades. En este caso, el umbral para el examen final podría ser 3.0, mientras que para las otras actividades podría ser 2.5.

### Promedio ponderado

El modelo más simple, que se utiliza en la mayoría de los cursos universitarios, es el de **promedio ponderado**. En este modelo la nota final es simplemente el promedio ponderado de las notas obtenidas en cada actividad. Se calcula como:

$$ A = \sum_{i} X_i \cdot W_i $$

Aunque es un modelo que ha sido usado ampliamente en la historia y la mayoría de estudiantes y docentes están muy familiarizados con él. Sin embargo adolece de algunos defectos. Para empezar la elección de los pesos es prácticamente arbitraria. No se usan criterios rigurosos para escogerlos, más allá de asignar pesos mayores a las actividades que se consideran más importantes o más difíciles.

Como resultado de esta arbitrariedad, los promedios ponderados no tienen en cuenta la interrelación entre los ítems de evaluación, ni es capaz de reproducir el juicio que un docente puede hacer de forma más cualitativa al evaluar el conjunto de las notas obtenidas por un estudiante. 

Tomemos un ejemplo. Supongamos que un curso tiene dos actividades de evaluación. La primera actividad es un trabajo escrito y la segunda es un examen final. Supongamos que el trabajo escrito tiene un peso de 0.3 y el examen final tiene un peso de 0.7. Supongamos también que un estudiante obtiene una nota de 5.0 en el trabajo escrito y una nota de 2.0 en el examen final. En este caso, la nota final del estudiante sería:

$$ A = 5.0 \cdot 0.3 + 2.0 \cdot 0.7 = 1.5 + 1.4 = 2.9 $$

En este caso, el estudiante reprueba el curso. Sin embargo, un docente podría considerar que la nota final del estudiante debe ser aprobatoria, ya que ha obtenido una buena nota en el trabajo escrito y una mala nota en el examen final. En este caso, el promedio ponderado no es capaz de reproducir el juicio del docente. Decimos que se produce un falso negativo.

Por otro lado, consideremos un caso de **falso positivo**. Supongamos un curso con tres actividades: dos tareas cortas y un examen final. Las tareas tienen un peso del 15% cada una ($W_1=0.15, W_2=0.15$) y el examen final un peso del 70% ($W_3=0.70$). Un estudiante obtiene las siguientes notas:

*   Tarea 1: 2.0
*   Tarea 2: 2.0
*   Examen Final: 3.5

El cálculo del promedio ponderado sería:

$$ A = 2.0 \cdot 0.15 + 2.0 \cdot 0.15 + 3.5 \cdot 0.70 = 0.3 + 0.3 + 2.45 = 3.05 $$

En este escenario, el estudiante aprueba el curso. Sin embargo, el docente puede considerar que el desempeño ha sido deficiente durante la mayor parte del semestre y que una nota de 3.5 en el examen final no compensa el desconocimiento mostrado en las tareas. El sistema de promedio ponderado permite que el estudiante "se salve" al final, generando lo que denominamos un falso positivo.

### Modelo compensatorio con umbrales   

Para corregir el defecto de un promedio ponderado se puede por ejemplo introducir un factor de penalización que tenga en cuenta el desempeño del estudiante en cada actividad. En este modelo la nota final se calcula como:

$$N_{local} = \sum_{i=1}^n W_i X_i U_i $$

donde $W_i$ es el peso de la actividad $i$, $X_i$ es la nota del estudiante en la actividad $i$ y $U_i$ se conoce como una función de utilidad que se calcula con base en la nota obtenida en las demás actividades, así:

$$ U_i = \prod_{j \neq i} g(x_j) \quad \text{donde} \quad g(x) = \min\left(1, \frac{x_j}{t_j}\right) $$

Como un ejemplo, tomemos el caso del falso positivo anterior. En este caso, las notas son $X_1=2.0, X_2=2.0, X_3=3.5$ y los pesos son $W_1=0.15, W_2=0.15, W_3=0.70$. Los umbrales son $t_1=2.5, t_2=2.5, t_3=2.5$. En este caso, la nota final sería:

$$
\begin{aligned}
N_{local} &= 0.15 \cdot 2.0 \cdot \min\left(1, \frac{2.0}{2.5}\right) + 0.15 \cdot 2.0 \cdot \min\left(1, \frac{2.0}{2.5}\right) + 0.70 \cdot 3.5 \cdot \min\left(1, \frac{3.5}{2.5}\right) \\
&= 0.15 \cdot 2.0 \cdot 0.8 + 0.15 \cdot 2.0 \cdot 0.8 + 0.70 \cdot 3.5 \cdot 1 \\
&= 0.24 + 0.24 + 2.45 \\
&= 2.93
\end{aligned}
$$

nótese que la nota final es 2.93, que es menor que 3.05. Esto se debe a que el modelo compensatorio tiene en cuenta el desempeño del estudiante en cada actividad y penaliza el hecho de que haya obtenido una mala nota en el examen final. 

Uno de los inconvenientes de este modelo es que puede ser muy penalizante para los estudiantes que obtienen una nota baja en una actividad, incluso si obtienen una nota alta en las demás. Por ejemplo, si un estudiante obtiene una nota de 2.0 en una actividad, su nota final se verá reducida significativamente, incluso si obtiene una nota de 5.0 en las demás actividades. 

### Modelo compensatorio con interpolación

Para corregir estas distorsiones, se propone un modelo más robusto que utiliza los siguientes componentes:

**Factor de Proximidad ($G_{ij}$)**:
Mide qué tan cerca está una nota $X_j$ de su umbral $U_j$. Si $X_j < U_j$, el factor reduce la contribución de otros ítems.

**Factor de Utilidad ($F_{local}$)** o Factor de Penalización:
Modela la interdependencia. Si un estudiante falla en temas avanzados, su nota se "castiga".

$$ F_{local} = \min \left( 1, \frac{\sum_{i} X_i W_i \prod_{j \neq i} G_{ij}}{A} \right) $$

Donde $G_{ij}$ es un factor de reducción derivado de si la nota $X_j$ cumple o no con su umbral $U_j$.

**Nota Base ($N_{raw}$)**:
Interpolación entre el piso (mínima nota obtenida en items regulares, $m$) y el promedio clásico.
    
$$ 
N_{raw} = m + (A - m) \cdot F_{local} 
$$

**Nota Final Definitive ($N_{final}$)**:
Se aplican reglas de seguridad para no perjudicar injustamente ni regalar nota:
*   Si $A < 2.95$ (Reprobado por promedio): $N_{final} = A$ (Se mantiene el promedio).
*   Si $A \ge 2.95$ (Aprobado por promedio): $N_{final} = \min(N_{raw}, A)$ (Penalización activa: la nota final nunca puede ser mayor al promedio simple).

Tomemos un nuevo ejemplo en el que las notas son $X_1=1.0, X_2=2.0, X_3=3.5$ y los pesos son $W_1=0.15, W_2=0.15, W_3=0.70$. Los umbrales son $t_1=2.5, t_2=2.5, t_3=2.5$. En este caso, la nota final sería:

**Promedio Clásico ($A$):**
$$ A = 1.0 \cdot 0.15 + 2.0 \cdot 0.15 + 3.5 \cdot 0.70 = 2.90 $$

**Promedio con umbrales ($N_{local}$):**
$$
\begin{aligned}
N_{local} &= 1.0 \cdot 0.15 \cdot \min\left(1, \frac{1.0}{2.5}\right) + 2.0 \cdot 0.15 \cdot \min\left(1, \frac{2.0}{2.5}\right) + 3.5 \cdot 0.70 \cdot \min\left(1, \frac{3.5}{2.5}\right) \\
&= 1.0 \cdot 0.15 \cdot 0.4 + 2.0 \cdot 0.15 \cdot 0.8 + 3.5 \cdot 0.70 \cdot 1.0 \\
&= 0.06 + 0.24 + 2.45 \\
&= 2.75
\end{aligned}
$$

la nota $X_1 =1$ sanciona fuertemente el promedio ponderado.

**Factor de Utilidad ($F_{local}$):**

$$
\begin{aligned}
F_{local} &= \text{clip}\left( \frac{1.0 \cdot 0.15 \cdot 0.4 + 2.0 \cdot 0.15 \cdot 0.8 + 3.5 \cdot 0.70 \cdot 1.0}{2.90}, 0, 1 \right) \\
&= \text{clip}\left( \frac{0.06 + 0.24 + 2.45}{2.90}, 0, 1 \right) \\
&= \text{clip}\left( \frac{2.75}{2.90}, 0, 1 \right) \\
&= 0.95
\end{aligned}
$$

**Nota Base ($N_{raw}$):**
$$ N_{raw} = 1.0 + (2.90 - 1.0) \cdot 0.95 = 1.0 + 1.90 \cdot 0.95 = 1.0 + 1.805 = 2.805 $$

**Nota Final Definitive ($N_{final}$):**
$$ N_{final} = \min(2.805, 2.90) = 2.805 $$

Todo es muy prometedor, pero ¿en realidad reduce esto los falsos positivos y falsos negativos?

## Experimentos numéricos

Para poner a prueba la hipótesis de que el modelo compensatorio con interpolación reduce los falsos positivos y falsos negativos, se realizaron los siguientes experimentos.

### Generación de notas aleatorias

Después de definir los items de evaluación de un curso hipotético, se generaron 1000 estudiantes con notas aleatorias. Las notas se generaron siguiendo tres tipos de distribución:

1.  **Distribución Bimodal (Items "Clave"):** Representa actividades críticas que tienden a polarizar el desempeño del grupo. Se modela con dos centros (1.2 y 4.0) y un ratio de 0.4, simulando un escenario donde una parte significativa de los estudiantes no logra comprender los conceptos base mientras que el resto alcanza un desempeño satisfactorio.
2.  **Distribución con Sesgo Negativo (Items "Fáciles"):** Simula actividades de baja complejidad o talleres de refuerzo donde la gran mayoría de los estudiantes obtienen notas sobresalientes, situando la moda en 4.5.
3.  **Distribución de Examen:** Modela evaluaciones sumativas de mayor exigencia, con una media de 2.8 y una desviación estándar de 1.1, lo que refleja una mayor dispersión y una dificultad que sitúa el promedio ligeramente por debajo del umbral de aprobación.
4.  **Distribución Normal (Items "Avanzados" o "Normales"):** Sigue una campana de Gauss tradicional con media 3.0 y desviación 1.0, representando el comportamiento estándar de una población estudiantil en condiciones de evaluación equilibradas.

### Evaluación por un oráculo numérico

Una vez generadas las notas, se evaluó a cada estudiante utilizando un oráculo numérico que simula el criterio experto. El oráculo numérico es una función que toma como entrada las notas de los estudiantes y devuelve una nota final que se utiliza como referencia para evaluar el desempeño del modelo matemático.

**Variables Clave:**
*   $D$: Nota del Item Definitorio (Ej. Examen Final/Proyecto 2).
*   $Adv$: Conjunto de notas de items "avanzados".
*   $Norm$: Conjunto de todas las notas.

**Reglas Heurísticas:**
1.  Si $D < 2.0 \rightarrow$ **Reprueba** (Fallo crítico).
2.  Si $2.0 \le D < 2.5 \rightarrow$ Aprueba SOLO SI **todos** los items avanzados $> 2.95$.
3.  Si $2.5 \le D < 3.0 \rightarrow$ Aprueba SOLO SI **al menos 2** items avanzados $> 2.95$.
4.  Si $3.0 \le D < 3.5 \rightarrow$ Reprueba SI **al menos 2** items avanzados $< 2.95$.
5.  Si $3.5 \le D < 4.0 \rightarrow$ Reprueba SI **todos** los items normales están perdidos.
6.  Si $D \ge 4.0 \rightarrow$ Reprueba SI **todos** los items avanzados están perdidos.

### Determinación de la tasa de falsos positivos y falsos negativos

Una vez evaluado cada estudiante por el oráculo numérico, se determinó la tasa de falsos positivos y falsos negativos. Para ello se calculó la nota final del estudiante utilizando cada uno de los esquemas de evaluación descritos antes y se comparó el resultado (aprobación o no aprobación de acuerdo a la nota definitiva) con el juicio del oráculo numérico. 

Definimos la tasa de falsos positivos (FPR) como el porcentaje de estudiantes que el modelo matemático aprueba pero el oráculo numérico reprueba. Definimos la tasa de falsos negativos (FNR) como el porcentaje de estudiantes que el modelo matemático reprueba pero el oráculo numérico aprueba.

Matemáticamente:

$$ 
\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}
$$

$$
\text{FNR} = \frac{\text{FN}}{\text{FN} + \text{TP}}
$$

### Resultados

Se desarrollo un notebook especial para poner a calcular la FPR y FNR. Los detalles están en el notebook `ejemplo_tasa_falsos_positivos_negativos`. 

Para un curso sencillo con items y pesos definido como se muestra en la tabla (ver `config_meccel.py`):

```
========================================
PESOS FINALES DE EVALUACIÓN
========================================
ITEM                      PESO      
-----------------------------------
Quices                    0.2000
Taller                    0.3000
Proyecto1                 0.1660
Proyecto3                 0.1660
Proyecto2                 0.1680
-----------------------------------
TOTAL SUMA                1.0000
========================================
```

Se obtuvieron los siguientes resultados:

```
==================== Promedio_Clasico ====================
FPR (Falsos Aprobados): 49.36%
FNR (Falsos Reprobados): 18.54%
Accuracy: 64.60%
TP: 369 | TN: 277 | FP: 270 | FN: 84

==================== Nota_Umbral ====================
FPR (Falsos Aprobados): 3.47%
FNR (Falsos Reprobados): 73.73%
Accuracy: 64.70%
TP: 119 | TN: 528 | FP: 19 | FN: 334

==================== Nota_Final_Avanzada ====================
FPR (Falsos Aprobados): 22.85%
FNR (Falsos Reprobados): 59.60%
Accuracy: 60.50%
TP: 183 | TN: 422 | FP: 125 | FN: 270
```

Como es de esperarse la nota con umbral, tanto la simple como aquella que usa interpolación, presenta un número menor de falsos positivos, es decir, la probabilidad de ganar el curso a pesar de que un docente experimentado lo repruebe es baja. Sin embargo, la tasa de falsos negativos es alta, lo que indica que la probabilidad de reprobar el curso a pesar de que un docente experimentado lo apruebe es alta. Esto último es grave.

### Optimización de pesos

Quizás la razón por la cuál se obtienen estos resultados es que los pesos no están bien distribuidos. Para poner a prueba esta hipótesis, se realizó un experimento de optimización de pesos. Los detalles están en el notebook `ejemplo_optimiza_pesos.ipynb`.

Para obtimizar los pesos, primero se identifico los ítems de la evaluación susceptibles de variarse. En este caso, los pesos de los quices, el taller y el proyecto 1 (que es el mismo peso del proyecto 3), son las variables libres. El peso del proyecto 2, que es la evaluación definitoria, se determina con la condición de que la suma de todos los pesos sea igual a 1. Es decir, $W_{proyecto2} = 1 - W_{quices} - W_{taller} - W_{proyecto1}$. Naturalmente, se fija una condición de un valor máximo para el peso de esté item de modo que no pueda adoptar valores artificalmente grandes (p.e. por encima del 40%).

La función objetivo es **Discontinua (Escalonada)**.  Al cambiar un peso $W_i$ ligeramente, la nota final $N_{final}$ varía suavemente, pero la decisión binaria (Aprobar/Reprobar) cambia abruptamente solo cuando cruza 3.0. Esto hace que el costo (recuento de errores) sea constante en vecindarios locales, produciendo gradientes iguales a cero.

Hay que seleccionar adecuadamente el minimizador en este caso:

*   **Métodos de Gradiente (SLSQP)**: **Fallan**. Al detectar gradiente cero, terminan prematuramente sin mejorar los pesos.
*   **Métodos Libres de Gradiente (Powell, Differential Evolution)**: **Funcionan**.
    *   **Powell** fue superior en velocidad y capacidad para encontrar mínimos globales en este contexto específico.
    *   **Differential Evolution** (con restricciones lineales duras) tendió a atascarse en óptimos locales de menor calidad para ciertas configuraciones complejas (`astrobio`).

El item definitorio (ej. Proyecto 3) tiene un límite de peso (ej. $W_{def} \le 0.30$).

*   **Estrategia Ganadora**: Uso de **Penalización** en la función objetivo.
$$ \text{Si } W_{def} > W_{max} \rightarrow Costo += (W_{def} - W_{max}) \cdot 100 $$
    Esta aproximación "suave" permitió a Powell explorar cerca de los límites y converger mejor que imponer límites rígidos (`Constraints`).

### Resultados

Abajo se muestran los resultados de la optimización:

```
COMPARACIÓN PRE VS POST OPTIMIZACIÓN

--- Configuración Inicial (Sugerida) ---
Costo (FPR+FNR): 0.6207
FPR: 48.18%
FNR: 13.89%
Accuracy: 67.25%

========================================
PESOS FINALES DE EVALUACIÓN
========================================
ITEM                      PESO      
-----------------------------------
Quices                    0.2000
Taller                    0.3000
Proyecto1                 0.1660
Proyecto2                 0.1660
Proyecto3                 0.1680
-----------------------------------
TOTAL SUMA                1.0000
========================================

--- Configuración Optimizada ---
Costo (FPR+FNR): 0.5025
FPR: 31.36%
FNR: 18.89%
Accuracy: 74.25%

========================================
PESOS FINALES DE EVALUACIÓN
========================================
ITEM                      PESO      
-----------------------------------
Quices                    0.1935
Taller                    0.1687
Proyecto1                 0.1783
Proyecto2                 0.1783
Proyecto3                 0.2810
-----------------------------------
TOTAL SUMA                1.0000
========================================
```

Como puede verse, la optimización de pesos mejoró la tasa de falsos positivos y la precisión, pero empeoró un poco la tasa de falsos negativos. Sin embargo los pesos resultantes son más razonables desde el punto de vista de un docente. Por ejemplo, el peso del proyecto 3 es ahora del 28%, lo cual es un valor razonable para un examen final.

El resultado de la optimización puede cambiar considerablemente si se modifican propiedades de los ítems de evaluación. En el ejemplo anterior, el taller es un ítem "avanzado", lo que significa que se espera que los estudiantes tengan un buen desempeño en él. Si se cambia el tipo de ítem a "facil", los resultados de la optimización cambiarán:

```
COMPARACIÓN PRE VS POST OPTIMIZACIÓN

--- Configuración Inicial (Sugerida) ---
Costo (FPR+FNR): 0.5620
FPR: 26.43%
FNR: 29.77%
Accuracy: 72.10%

========================================
PESOS FINALES DE EVALUACIÓN
========================================
ITEM                      PESO      
-----------------------------------
Quices                    0.2000
Taller                    0.3000
Proyecto1                 0.1660
Proyecto2                 0.1660
Proyecto3                 0.1680
-----------------------------------
TOTAL SUMA                1.0000
========================================

--- Configuración Optimizada ---
Costo (FPR+FNR): 0.4181
FPR: 19.20%
FNR: 22.61%
Accuracy: 79.30%

========================================
PESOS FINALES DE EVALUACIÓN
========================================
ITEM                      PESO      
-----------------------------------
Quices                    0.2467
Taller                    0.1446
Proyecto1                 0.1547
Proyecto2                 0.1547
Proyecto3                 0.2993
-----------------------------------
TOTAL SUMA                1.0000
========================================
```

Primero, el FPR con los pesos originales es de por si pequeña, 29.77%. Con lo pesos optimizados, la tasa disminuye un poco hasta alcanzar un valor de 22.61%.

Con esto se nota que no solo los pesos pueden determinar la tasa de falsos positivos y negativos, sino también la percepción que tiene el docente (que en la simulación es el oráculo) de si un ítem es "facil", "avanzado" o "clave". 

### Nota de concepto

Una posible estrategia para mejorar la manera en la que se tiene en cuenta de forma global el desempeño, evaluando simultáneamente las notas de los ítems parciales del curso, es introducir un ítem de evaluación especial al que hemos llamado "nota de concepto". 

Para un experimento particular, la nota de concepto se calcula de la siguiente manera:

Reglas:
- 0.0: Todos los items avanzados son 0.0.
- 2.0: Todos los items avanzados son < umbral (perdidos).
- 2.5: Todos avanzados perdidos PERO todos fáciles ganados (>= umbral).
- 3.0: Al menos un item avanzado ganado.
- 3.5: Al menos un item avanzado ganado Y todos fáciles ganados.
- 4.0: Al menos dos items avanzados ganados.
- 4.5: Al menos dos items avanzados ganados Y todos fáciles ganados.
- 5.0: Todos los ítems (avanzados + faciles) ganados.

Como ven es una nota en niveles discretos que intenta capturar la idea de "nivel de dominio" del estudiante sobre el curso.

En un experimento particular (el mismo caso estudiado arriba con el item de Taller clasificado como fácil), la optimización produjo este resultado:

```
COMPARACIÓN PRE VS POST OPTIMIZACIÓN

--- Configuración Inicial (Sugerida) ---
Costo (FPR+FNR): 0.6402
FPR: 51.91%
FNR: 12.11%
Accuracy: 66.00%

========================================
PESOS FINALES DE EVALUACIÓN
========================================
ITEM                      PESO      
-----------------------------------
Quices                    0.1818
Taller                    0.2727
Proyecto1                 0.1509
Proyecto2                 0.1509
Proyecto3                 0.1527
Nota_Concepto             0.0909
-----------------------------------
TOTAL SUMA                1.0000
========================================

--- Configuración Optimizada ---
Costo (FPR+FNR): 0.4724
FPR: 36.91%
FNR: 10.33%
Accuracy: 75.05%

========================================
PESOS FINALES DE EVALUACIÓN
========================================
ITEM                      PESO      
-----------------------------------
Quices                    0.1100
Taller                    0.1863
Proyecto1                 0.1564
Proyecto2                 0.1564
Proyecto3                 0.2816
Nota_Concepto             0.1094
-----------------------------------
TOTAL SUMA                1.0000
========================================
```

Si se compara con es caso respectivo, la nota de concepto no mejoro significativamente el cambio en las FPR y FNR después de la optimización de los pesos. Es decir, poner una nota de concepto no parece ser una estrategia que mejore significativamente la manera en la que la nota final se calcula.

## Conclusiones preliminares

Aquí algunas conclusiones preliminares:

- Se confirma que el cálculo de la nota de un curso usando un promedio ponderado puede producir falsos positivos con probabilidades que oscilan (en nuestros experimentos), dependiendo de la distribución de las notas y los pesos asignados a cada item, entre 20% y 50%.

- Se confirma también que este esquema clásico puede también producir falsos negativos. Es decir, lo que casi nadie espera, que estudiantes que el docente considera que aprobaron el curso, en realidad no lo hicieron. En nuestros experimentos, la tasa de falsos negativos oscila entre 10% y 30%.

- Un sistema de calificación con umbrales, mejora la manera en la que la nota final se calcula, ya que permite penalizar a los estudiantes que no alcanzaron un desempeño mínimo en algunos de los items de evaluación. Sin embargo, aunque estos sistemas presentan tasas increíblemente bajas de falsos positivos (entre 1% y 5%), producen una cantidad asombrosamente alta de falsos negativos (entre 60% y 80%). Al punto que no es recomendada usarla.

- Optimizar los pesos de las notas para reducir la tasa de falsos positivos y negativos es posible y tiene el efecto deseado. Esto implica que nuestra hipótesis inicial, a saber, que la distribución de los pesos de las notas de un curso tiene un impacto en la probabilidad de que personas que aprobaron el curso no lo hagan (o viceversa), es correcta. 

- El procedimiento de optimización es altamente sensible a la elección del nivel de las evaluaciones. Por ejemplo, si se cambia el nivel de un item de "facil" a "avanzado", la tasa de falsos positivos y negativos puede cambiar significativamente. Esto se debe a la forma en la que se evalúan en nuestros modelos si una persona debe aprobar o no el curso usando lo que llamamos un "oráculo". Este oráculo utiliza un conjunto de condiciones para definir si una persona aprobó o no el curso, y estas condiciones son sensibles a si un ítem se considera avanzado o fácil. 

- Agregar una nota de concepto no parece ser una estrategia que mejore significativamente la manera en la que la nota final se calcula. En el experimento particular que estudiamos, la nota de concepto no mejoró significativamente el cambio en las FPR y FNR después de la optimización de los pesos. 

- En un experimento que realizamos con un conjunto de notas reales de un curso, obtuvimos que la optimización de los pesos no tienen un impacto significativo en la tasa de falsos positivos y negativos. 

## Perspectivas

Los resultados de estos experimentos dependen fuertemente de la manera como se definen los criterios para aprobar o no el curso. Es decir, dependen fuertemente de la manera como se define el "oráculo". Sería interesante realizar experimentos con datos evaluados por humanos con criterios menos determinísticos y más flexibles para ver si los resultados cambian significativamente. El problema de esta aproximación es que sería necesario un volumen significativo de datos por cada curso para realizar la minimización de forma estadísticamente relevante.

Para realizar este experimento se ha diseñado un notebook `ejemplo_genera_notas_para_humanos.ipynb` que permite generar archivos en Excel con un número dado de notas simuladas y para que un humano con ojo entrenado evalúe, con base en las notas obtenidas si un estudiante debería o no aprobar el curso.

Otra avenida de exploración sería la de probar algoritmos más sofisticados para la optimización de los pesos o para el oraculo. Por ejemplo se podrían utilizar árboles de decisión o redes neuronales para modelar el oráculo. En todos los casos, sin embargo la debilidad más notable es la ausencia de datos de entrenamiento apropiado.

## Recomendaciones finales

Mientras se resuelven muchos de los retos que hemos esbozado arriba, nuestra experiencia nos muestra que hay varias acciones sencillas que pueden tomarse para mejorar la manera en la que se calcula la nota final de un curso:

- Evaluar con herramientas numéricas la tasa de falsos positivos y negativos que una determinada configuración de evaluación produce. Esto permite identificar configuraciones que producen tasas de falsos positivos y negativos inaceptables.

- Evitar el uso de sistemas de calificación con umbrales. Aunque estos sistemas presentan tasas increíblemente bajas de falsos positivos (entre 1% y 5%), producen una cantidad asombrosamente alta de falsos negativos (entre 60% y 80%). Al punto que no es recomendada usarla.

- Escoger con juicio el grado de dificultad de cada una de las evaluaciones de nuestros cursos. No se trata simplemente de subir o bajar el promedio, sino el de comprometerse seriamente con hacer evaluaciones con grados de dificultad apropiados para cada nivel.

- Siempre que sea posible, tratar de optimizar los pesos de las evaluaciones para reducir la tasa de falsos positivos y negativos, asís sea poco.


**[Jorge I. Zuluaga](mailto:[jorge.zuluaga@udea.edu.co])**

2 de Febrero de 2026