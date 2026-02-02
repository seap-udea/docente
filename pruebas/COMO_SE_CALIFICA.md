# ¿Cómo se Califica? El Sistema de "Promedios Compensatorios Híbridos"

Este curso utiliza un sistema de calificación diseñado para asegurar no solo un buen promedio general, sino una **competencia mínima** en cada área evaluada. Técnicamente, este enfoque se conoce como **Enfoque Híbrido Compensatorio-Conjuntivo**.

A continuación explicamos cómo funciona, la fórmula matemática y ejemplos prácticos.

---

## 1. La Filosofía: Compensación vs. Conjunción

*   **Sistema Tradicional (Compensatorio):** Un promedio simple. Una nota muy alta (5.0) en la Tarea 1 compensa completamente una nota muy baja (1.0) en el Quiz 1.
*   **Sistema Estricto (Conjuntivo):** Debes ganar **todas** las materias. Si pierdes una, pierdes el curso.
*   **Nuestro Sistema (Híbrido):** Es un punto medio.
    *   Si tus notas están por encima de un **Umbral de Calidad (2.5)**, el sistema actúa como un promedio normal.
    *   Si tienes notas por debajo del umbral (< 2.5), el sistema te "cobra" un impuesto de penalización en los pesos de las otras notas. Entre más notas tengas por debajo del umbral, menor será el valor de tus notas buenas.

> **Regla de Oro:** ¡Evita a toda costa sacar menos de 2.5 en cualquier actividad! Una nota de 2.0 hace mucho más daño que una nota de 3.0 ayuda.

---

## 2. La Fórmula Matemática

La nota final no es solo la suma ponderada de tus notas. Se ajusta por unos **Factores de Utilidad ($U$)** que dependen del desempeño en las otras áreas.

$$ NotaFinal = \sum_{i} \left( w_i \cdot x_i \cdot U_i(\vec{x}) \right) $$

Donde:
*   $x_i$: Tu nota en la actividad $i$ (ej. Quiz 1).
*   $w_i$: El peso porcentual de esa actividad (ej. 20%).
*   $U_i(\vec{x})$: El **Factor de Penalización**.

### ¿Cómo se calcula $U_i$?
Para cada nota, miramos **todas las demás notas**. Si las demás notas cumplen el umbral ($U_{val} = 2.5$), el factor es 1.0 (no hay castigo). Si alguna falla, el factor baja.

$$ U_i = \prod_{j \neq i} \min\left(1, \frac{x_j}{t_j}\right) $$

**Interpretación:** Tu nota del Examen Final "vale completo" solo si demostraste competencia mínima en los Quices y Tareas. Si perdiste los quices, tu Examen Final "vale menos" en el cálculo.

---

## 3. Ejemplos Prácticos

Asumamos, para simplificar, 3 notas con los siguientes pesos:
*   **Quiz**: 30%
*   **Tarea**: 30%
*   **Examen**: 40%

**Nota:** Si el "Promedio Natural" (sin penalización) es menor a 3.0, la nota queda tal cual (o se aplica una regla de reprobación directa). La penalización afecta a quienes, por promedio simple, pasarían.

### Caso 1: El Ganador con Penalización (Zona de Riesgo)
*Estudiante que pasa, pero su nota baja considerablemente por un descache.*

**Notas:**
*   Quiz: **2.0** (¡Peligro! Debajo de 2.5)
*   Tarea: 4.5
*   Examen: 4.0

**Cálculo Tradicional (Promedio Simple):**
$$ (2.0 \times 0.3) + (4.5 \times 0.3) + (4.0 \times 0.4) = 0.6 + 1.35 + 1.6 = \textbf{3.55} $$
*En un sistema normal, este estudiante estaría tranquilo.*

**Cálculo Híbrido (Umbral 2.5):**
El Quiz de 2.0 afecta el peso de la Tarea y el Examen.
*   Factor de Penalización ($U_{quiz}$) = $\frac{2.0}{2.5} = 0.8$
*   La Tarea ya no vale el 100% de su peso, sino que se multiplica por 0.8.
*   El Examen también se multiplica por 0.8.

$$ NotaFinal \approx (2.0 \times 0.3) + (4.5 \times 0.3 \times \textbf{0.8}) + (4.0 \times 0.4 \times \textbf{0.8}) $$
$$ = 0.6 + 1.08 + 1.28 = \textbf{2.96} $$

**Resultado:** El estudiante pasa de un cómodo **3.55** a un **2.96**.
*Nota: Dependiendo del redondeo, podría salvarse raspando o perder. En este ejemplo estricto, ¡PERDERÍA por un solo quiz malo!*

*(Para el ejemplo solicitado de "baja un poco pero gana", imaginemos que el Quiz fue 2.4 en lugar de 2.0. La penalización sería menor y sacaría ~3.4 en vez de 3.55).*

### Caso 2: Perder Estando en el Rango (El Efecto Dominó)
*Estudiante con buen promedio teórico pero con debilidades críticas.*

**Notas:**
*   Quiz: **2.0**
*   Tarea: **2.0**
*   Examen: **4.8** (¡Excelente examen!)

**Cálculo Tradicional:**
$$ (2.0 \times 0.3) + (2.0 \times 0.3) + (4.8 \times 0.4) = 0.6 + 0.6 + 1.92 = \textbf{3.12} $$
*Promedio > 3.0. Debería ganar.*

**Cálculo Híbrido:**
Aquí hay **dos** notas malas. Se castigan mutuamente y castigan al examen.
*   El Examen recibe castigo del Quiz (0.8) Y de la Tarea (0.8).
*   $U_{examen} = 0.8 \times 0.8 = 0.64$.

$$ AporteExamen = 4.8 \times 0.4 \times \textbf{0.64} = 1.23 $$
(En lugar del 1.92 original).

$$ NotaFinal \approx \text{Aportes Bajos} + 1.23 \approx \textbf{2.5} $$

**Resultado:** A pesar de demostrar excelencia en el final y tener un promedio simple de 3.12, el estudiante **pierde la materia** con 2.5 porque falló sistemáticamente en los componentes previos.

---

## Resumen
1.  **Mantén todo sobre 2.5**: Es la zona segura.
2.  **Una nota mala contagia a las buenas**: Un 1.0 en un quiz le quita valor a tu 5.0 en el examen.
3.  **Promedio > 3.0 no garantiza nada**: Si ese promedio se logró "arrastrando" notas muy malas, el sistema te bajará la definitiva.
