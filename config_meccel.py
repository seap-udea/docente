"""
Módulo config
=============

Este módulo contiene la configuración de la evaluación para el curso.
Define los items de evaluación, sus pesos, correlaciones y tipos de distribución.

Claves de Configuración
-----------------------

A continuación se describen las claves utilizadas en la definición de items:

*   **nombre** (*str*): Identificador único del item (ej. "Quices_Mod1", "Parcial_1").
*   **tipo** (*str*): Determina la distribución estadística de las notas generadas.
    *   ``facil``: Distribución con sesgo negativo; la mayoría obtiene notas altas.
    *   ``avanzado``: Distribución normal centrada en 3.0. Item de dificultad media/alta.
    *   ``clave``: Distribución bimodal. Separa grupos de alto y bajo rendimiento.
    *   ``examen``: Distribución normal para exámenes, típicamente con media algo inferior a 3.0.
*   **peso** (*float*, opcional): Peso fijo asignado al item (ej. 0.20). Si se omite, se calcula automáticamente.
*   **peso_sugerido** (*float*, opcional): Peso sugerido para la optimización si no se fija un peso estricto.
*   **peso_min** (*float*, opcional): Límite inferior para el peso durante la optimización.
*   **peso_max** (*float*, opcional): Límite superior para el peso durante la optimización.
*   **correlacionado_con** (*str*, opcional): Nombre de otro item con el que este item guarda relación.
*   **factor_correlacion** (*float*, opcional): Multiplicador que ajusta la relación con el item padre.
*   **debilidad** (*float*, opcional): La nota debajo de la cual se considera que la nota es muy baja.
*   **umbral** (*float*, opcional): En esquema de calificación con umbrales, cuando la nota está debajo de este valor las otras notas sufren una disminución.
*   **item_definitorio** (*dict*): Configuración especial para el examen final o evaluación de cierre.
    *   **muerte_subita** (*float*): Nota mínima requerida en este item para aprobar (concepto ilustrativo).
    *   **peso_maximo** (*float*): Restricción superior para el peso del examen final.

Ejemplo de Configuración
------------------------

La configuración se define como un diccionario con dos claves principales:
'items_normales' y 'item_definitorio'.
"""

config_evaluacion = {
    "items_normales": [
        {
            "nombre": "Quices", 
            "debilidad": 3.5,
            "peso_min": 0.10, "peso_max": 0.30, "peso_sugerido": 0.20, 
            "tipo": "avanzado",
            "umbral": 3.0 
        },
        {
            "nombre": "Taller", 
            "debilidad": 3.5,
            "peso_min": 0.10, "peso_max": 0.30, "peso_sugerido": 0.30, 
            "tipo": "facil",
            "umbral": 3.0 
        },
        {
            "nombre": "Proyecto1", 
            "debilidad": 2.5,
            "peso_min": 0.05, "peso_max": 0.20, "peso_sugerido": 0.166, 
            "tipo": "clave",
            "umbral": 3.0 
        },
        {
            "nombre": "Proyecto2", 
            "debilidad": 2.5,
            "correlacionado_con": "Proyecto1", "factor_correlacion": 1.0,
            "tipo": "avanzado",
            "umbral": 3.0 
        },
        
    ],

    "item_definitorio": {
        "nombre": "Proyecto3",
        "muerte_subita": 2.50, # muerte_subita
        "peso_maximo": 0.30,   # peso_maximo
        "tipo": "examen",
        "umbral": 2.5
    },
    
    "nota_concepto": {
        "peso_minimo": 0.05,
        "peso_maximo": 0.15,
        "peso_sugerido": 0.10,
        "umbral": 1e-20 # Para que no sancione por umbral
    }
}
