"""
Módulo notas
============

Este módulo proporciona herramientas para la generación de calificaciones sintéticas
y la optimización de pesos para cursos académicos.

Permite configurar items de evaluación (quices, tareas, exámenes) con correlaciones,
umbrales y tipos de distribución de notas, para luego generar datos simulados.

Para detalles sobre la configuración, consulte el módulo `config`.
"""

import numpy as np
import pandas as pd

def autoconfigura_items(config):
    """
    Ajusta los items correlacionados en la configuración dada.
    
    Esta función itera sobre los items y si encuentra dependencias (correlacionado_con),
    ajusta los parámetros del item hijo (peso_min, peso_max, peso_sugerido) basándose en el padre.
    
    Args:
        config (dict): Diccionario de configuración con 'items_normales'.

    Returns:
        dict: La configuración actualizada (in-place).

    Ejemplo:
        >>> import config
        >>> conf = autoconfigura_items(config.config_evaluacion)
        Autoconfigurando items correlacionados...
         > Quiz_1: Ajustado vs Tarea_1 (x0.5)
    """
    print("Autoconfigurando items correlacionados...")
    reg_items = config['items_normales']
    item_map = {item['nombre']: item for item in reg_items}

    for item in reg_items:
        if 'correlacionado_con' in item:
            parent_name = item['correlacionado_con']
            factor = item['factor_correlacion']
            if parent_name in item_map:
                parent = item_map[parent_name]
                item['peso_min'] = parent.get('peso_min', 0.0) * factor
                item['peso_max'] = parent.get('peso_max', 1.0) * factor
                item['peso_sugerido'] = parent.get('peso_sugerido', 0.0) * factor
                print(f" > {item['nombre']}: Ajustado vs {parent_name} (x{factor})")
            else:
                print(f" [!] Error: Padre '{parent_name}' no encontrado para '{item['nombre']}'")
    
    # Calcular y almacenar pesos finales en la configuración
    try:
        w_reg, w_def, w_nc = calcular_pesos(config)
        
        for item, w in zip(config['items_normales'], w_reg):
            item['peso_final'] = w
            
        config['item_definitorio']['peso_final'] = w_def
        
        if 'nota_concepto' in config:
            nc = config['nota_concepto']
            nc['peso_final'] = w_nc
            
    except NameError:
        # En caso de que calcular_pesos no esté definido aún (circular/orden)
        pass
        
    return config

def muestra_pesos(config):
    """
    Muestra los pesos finales definidos en la configuración.
    
    Args:
        config (dict): Configuración de evaluación ya procesada.
    """
    print("\n" + "="*40)
    print("PESOS FINALES DE EVALUACIÓN")
    print("="*40)
    
    total = 0.0
    
    # Items Normales
    print(f"{'ITEM':<25} {'PESO':<10}")
    print("-" * 35)
    
    for item in config['items_normales']:
        w = item.get('peso_final', 0.0)
        print(f"{item['nombre']:<25} {w:.4f}")
        total += w
        
    # Item Definitorio
    def_item = config['item_definitorio']
    w_def = def_item.get('peso_final', 0.0)
    print(f"{def_item['nombre']:<25} {w_def:.4f}")
    total += w_def
    
    # Nota Concepto
    if 'nota_concepto' in config:
        nc = config['nota_concepto']
        w_nc = nc.get('peso_final', 0.0)
        print(f"{'Nota_Concepto':<25} {w_nc:.4f}")
        # Añadir al total solo para visualizar impacto si fuera parte de la suma
        # Ojo: si no es parte de la suma de examen, mostrarlo separado?
        # Lo sumamos para ver si todo suma 1.0 (o +0.1 etc)
        total += w_nc
        
    print("-" * 35)
    print(f"{'TOTAL SUMA':<25} {total:.4f}")
    print("="*40)

# ==========================================
# RUTINAS DE AJUSTES DE PESOS
# ==========================================

def obtener_indices_independientes(config):
    """
    Identifica los índices de los items que no tienen correlaciones explícitas
    y que tienen definidos los parámetros de optimización (peso_min, peso_max, peso_sugerido).

    Args:
        config (dict): Configuración de evaluación.

    Returns:
        list: Lista de índices enteros de los items a optimizar.
    """
    indices = []
    items = config['items_normales']
    required_keys = ['peso_min', 'peso_max', 'peso_sugerido']
    
    for i, item in enumerate(items):
        # Independiente y con todas las claves de optimización
        if 'correlacionado_con' not in item:
            if all(k in item for k in required_keys):
                indices.append(i)
    return indices

def reconstruir_pesos_completos(x_independent, config):
    """
    Reconstruye el vector completo de pesos a partir de los pesos independientes.

    Propaga los pesos a los items correlacionados.

    Args:
        x_independent (array-like): Array con los pesos de los items independientes.
        config (dict): Configuración de evaluación.

    Returns:
        np.array: Array con los pesos completos para todos los items normales.
        
    Ejemplo:
        >>> x_indep = [0.15, 0.15, 0.15]
        >>> w_full = reconstruir_pesos_completos(x_indep, config)
    """
    items = config['items_normales']
    indep_indices = obtener_indices_independientes(config)
    
    weight_map = {}
    
    # 1. Llenar Independientes
    current_indep_idx = 0
    full_weights = np.zeros(len(items))
    
    for i in indep_indices:
        val = x_independent[current_indep_idx]
        item_name = items[i]['nombre']
        weight_map[item_name] = val
        full_weights[i] = val
        current_indep_idx += 1
        
    # 1.1. Llenar Independientes NO optimizados (Fixed)
    # Si un item es independiente pero no está en indices_indep, tomamos su peso fijo o sugerido
    for i, item in enumerate(items):
        if i not in indep_indices and 'correlacionado_con' not in item:
            # Usar 'peso' fijo si existe, sino 'peso_sugerido'
            w_val = item.get('peso', item.get('peso_sugerido', 0.0))
            full_weights[i] = w_val
            weight_map[item['nombre']] = w_val
        
    # Bucle de 3 pasadas para dependencias
    for _ in range(3):
        for i, item in enumerate(items):
            if full_weights[i] == 0.0 and 'correlacionado_con' in item:
                parent = item['correlacionado_con']
                if parent in weight_map:
                    w_val = weight_map[parent] * item['factor_correlacion']
                    full_weights[i] = w_val
                    weight_map[item['nombre']] = w_val
                    
    return full_weights

def calcular_pesos(config_in):
    """
    Calcula los pesos finales sugeridos basados en la configuración.

    Utiliza los valores 'sugerido' o 'peso' definidos en la configuración y propaga
    las correlaciones para obtener el vector final de pesos normales y el peso del examen.

    Args:
        config_in (dict): Configuración de evaluación.

    Returns:
        tuple: (final_reg_weights, ft_w)
            - final_reg_weights (np.array): Pesos de items normales.
            - ft_w (float): Peso del item definitorio (examen).

    Ejemplo:
        >>> reg_w, ex_w = calcular_pesos(config)
        >>> print(ex_w)
        0.28
    """
    reg_items = config_in['items_normales']
    def_item = config_in['item_definitorio']
    weight_map = {}
    
    # Inicio
    for item in reg_items:
        if 'peso' in item:
            weight_map[item['nombre']] = item['peso']
        elif 'correlacionado_con' not in item:
            weight_map[item['nombre']] = item.get('peso_sugerido', 0.0)
            
    # Propagacion (Loop seguro)
    # final_reg_weights = [] # No usado en el bucle
    for _ in range(3): 
        for item in reg_items:
            w = 0.0
            if 'peso' in item:
                 w = item['peso']
            elif 'correlacionado_con' in item:
                 parent = item['correlacionado_con']
                 if parent in weight_map:
                     w = weight_map[parent] * item['factor_correlacion']
                 else:
                     w = item.get('peso_sugerido', 0.0)
            else:
                 w = item.get('peso_sugerido', 0.0)
            weight_map[item['nombre']] = w
            
    # Extraccion Final
    final_reg_weights = [weight_map[item['nombre']] for item in reg_items]
    final_reg_weights = np.array(final_reg_weights)
    
    # Residual
    if 'peso' in def_item:
        ft_w = def_item['peso']
    else:
        ft_w = 1.0 - np.sum(final_reg_weights)
        if ft_w < 0: ft_w = 0.0
        
    # --- AJUSTE DE ESCALADO POR NOTA CONCEPTO ---
    nc_w = 0.0
    if 'nota_concepto' in config_in:
        nc = config_in['nota_concepto']
        nc_w = nc.get('peso', nc.get('peso_sugerido', 0.0))
        
        # Factor de normalización
        # Suma actual = 1.0 (Items+Def) + nc_w
        # Queremos que la suma total sea 1.0
        current_sum = np.sum(final_reg_weights) + ft_w + nc_w
        if current_sum > 0:
             factor = 1.0 / current_sum
             final_reg_weights *= factor
             ft_w *= factor
             nc_w *= factor
        
    return final_reg_weights, ft_w, nc_w

# ==========================================
# CÁLCULO DE NOTAS CON UMBRALES
# ==========================================

def _calcula_factor_umbral(X, U):
    """
    Calcula el factor de corte (g) dado un umbral U.
    Implementación interna correspondiente a g_threshold_vectorized.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = np.clip(X / U, 0, 1)
    return factor

def _aplicar_ayuda_redondeo(notas_array):
    """
    Aplica la regla de negocio: 'Todas las notas definitivas que queden 
    entre 2.90 y 3.00 redondearlas siempre a 3.0'.
    
    Se aplica después del redondeo a 2 decimales.
    
    Args:
        notas_array (np.array): Array de notas o escalar.
        
    Returns:
        np.array: Notas ajustadas.
    """
    # Asegurar redondeo previo a 2 decimales para consistencia en la comparación
    notas_redondeadas = np.round(notas_array, 2)
    
    # Aplicar regla: 2.90 <= x < 3.00 -> 3.00
    # Usamos (>= 2.90) & (< 3.00) sobre los valores ya redondeados
    mask = (notas_redondeadas >= 2.90) & (notas_redondeadas < 3.00)
    
    # Retornamos una copia modificada
    return np.where(mask, 3.0, notas_redondeadas)

def calcula_promedio_con_umbrales_avanzado(df, config, U=None, threshold_fail=2.95):
    """
    Calcula la nota final interpolada considerando umbrales de rendimiento.

    Esta rutina penaliza el promedio clásico si el desempeño en componentes individuales
    cae por debajo de un umbral, modelando dependencias entre conocimientos.
    
    Es una generalización de calculate_grade_strict_general.

    Args:
        df (pd.DataFrame): DataFrame con las notas de los estudiantes.
        config (dict): Configuración de evaluación.
        U (float/array, optional): Umbral de referencia de utilidad. 
                                   Si es None, se usan los umbrales individuales de la configuración.
        threshold_fail (float): Nota mínima de aprobación para aplicar lógica de no-perjuicio (default: 2.95).

    Returns:
        pd.DataFrame: DataFrame extendido con columnas:
            - Promedio_Clasico: Nota promedio ponderada simple.
            - Nota_Final: Nota ajustada con umbrales.
            - Factor_Local: Factor de penalización aplicado.
            - Piso: Mínimo de los items normales.
    """
    # 1. Obtener Pesos (desempaquetado actualizado)
    reg_weights, ft_weight, nc_weight = calcular_pesos(config)
    
    # 2. Identificar Columnas
    cols_reg = [item['nombre'] for item in config['items_normales']]
    col_def = config['item_definitorio']['nombre']
    
    all_cols = cols_reg + [col_def]
    weights_list = [reg_weights, [ft_weight]]
    
    # Agregar Nota Concepto si existe
    if 'nota_concepto' in config:
        all_cols.append('nota_concepto')
        weights_list.append([nc_weight])
        
    weights_all = np.concatenate(weights_list)
    
    # Validar columnas
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el DataFrame: {missing}")
        
    # 3. Matriz de Notas y Pesos
    X_comps = df[all_cols].values
    W_global = weights_all
    
    eps = 1e-9
    
    # 4. Promedio Clásico (A)
    # A = sum(X_i * W_i)
    A = np.dot(X_comps, W_global)
    
    # 5. Piso (m)
    # Se toma el mínimo de los items normales (excluyendo examen final)
    # Esto sigue la lógica original de 'min(M1, M2, M3)'
    X_reg = df[cols_reg].values
    m_floor = np.min(X_reg, axis=1)
    
    # 6. Factores g(x)
    # Definir U: Si es None, usar los umbrales de config, sino usar el valor pasado
    if U is None:
        thresholds = [item.get('umbral', 2.5) for item in config['items_normales']]
        thresholds.append(config['item_definitorio'].get('umbral', 2.5))
        if 'nota_concepto' in config:
             thresholds.append(config['nota_concepto'].get('umbral', 1e-20))
        U_act = np.array(thresholds)
    else:
        U_act = U

    G = _calcula_factor_umbral(X_comps, U_act)
    
    # 7. Pesos Locales U_i (Interdependencia)
    # La utilidad del item i depende del rendimiento en los otros items
    n_items = X_comps.shape[1]
    U_factors = np.zeros_like(G)
    
    for i in range(n_items):
        # Indices de 'otros' items
        others_idx = [j for j in range(n_items) if j != i]
        U_factors[:, i] = np.prod(G[:, others_idx], axis=1)
        
    # 8. Suma Local Penalizada S_local
    # S_local = sum(X_i * W_i * U_i)
    S_local = np.sum(X_comps * W_global * U_factors, axis=1)
    
    # 9. Factor Local F_local
    # Ratio entre suma penalizada y promedio ideal
    F_local = np.zeros_like(A)
    mask_A = A > eps
    F_local[mask_A] = np.clip(S_local[mask_A] / A[mask_A], 0, 1)
    F_local[~mask_A] = 1.0
    
    # 10. Nota Final Interpolada (Raw)
    # N = m + (A - m) * F
    N_raw = m_floor + (A - m_floor) * F_local
    N_raw = np.maximum(N_raw, m_floor)
    
    # 11. Ajustes Finales (Políticas de seguridad)
    # Si A < 3.0 (reprobado), mantuvo el promedio (no castigar más).
    # Si A >= 3.0 (aprobado), N = min(N_raw, A) (nunca bonificar).
    N_final = np.where(
        A < threshold_fail,
        A,
        np.minimum(N_raw, A)
    )
    
    # Retornar DataFrame con resultados
    res = df.copy()
    # Aplicamos la ayuda de redondeo también al promedio clásico si se considera nota definitiva
    res['Promedio_Clasico'] = _aplicar_ayuda_redondeo(A)
    
    res['Piso'] = m_floor.round(2)
    res['Factor_Utilidad'] = F_local.round(2) # Renombrado de Factor_Local para claridad
    
    # Aplicamos la ayuda de redondeo a la Nota Final
    res['Nota_Final'] = _aplicar_ayuda_redondeo(N_final)
    
    return res

def calcula_promedio_con_umbrales_simple(df, config, threshold_fail=3.0, factor_penalizacion=0.8):
    """
    Calcula la nota final usando una lógica de umbrales estricta pero simplificada.
    
    Esta función penaliza la nota si el rendimiento en items individuales es bajo
    (inferior al umbral configurado para cada item).
    
    Args:
        df (pd.DataFrame): DataFrame con las notas de los estudiantes.
        config (dict): Configuración de evaluación.
        threshold_fail (float): Nota por debajo de la cual se considera reprobado el promedio simple (default 3.0).
        factor_penalizacion (float): Factor de reducción si se reprueba el promedio simple (default 0.8).
        
    Returns:
        np.array: Array con las notas finales ajustadas.
        
    Ejemplo:
        >>> final_grades = calcula_promedio_con_umbrales_simple(df, config)
    """
    # 1. Obtener Pesos
    reg_weights, ft_weight, nc_weight = calcular_pesos(config)
    
    reg_items = config['items_normales']
    def_item = config['item_definitorio']
    
    # Preparar listas base
    weights_list = [reg_weights, [ft_weight]]
    reg_names = [item['nombre'] for item in reg_items]
    def_name = def_item['nombre']
    all_names = reg_names + [def_name]
    thresholds = [item.get('umbral', 2.5) for item in reg_items]
    thresholds.append(def_item.get('umbral', 2.5))
    
    # Agregar Nota Concepto si existe
    if 'nota_concepto' in config:
        weights_list.append([nc_weight])
        all_names.append('nota_concepto')
        thresholds.append(config['nota_concepto'].get('umbral', 1e-20))
    
    # Concatenar pesos
    all_weights = np.concatenate(weights_list)
    
    # Matriz de notas
    grades_matrix = df[all_names].values
    
    # Extraer vector de umbrales
    # thresholds calculado arriba
    T_vector = np.array(thresholds)
    
    # Factores U (Utilidad)
    # ratios = clam(grade / threshold, 0, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
         ratios = np.clip(grades_matrix / T_vector, 0, 1)
    
    n_items = len(all_names)
    u_factors = np.zeros_like(grades_matrix)
    
    # Para cada item i, su factor U depende del rendimiento en los OTROS items
    for i in range(n_items):
        mask = np.ones(n_items, dtype=bool)
        mask[i] = False
        u_factors[:, i] = np.prod(ratios[:, mask], axis=1)
        
    # Nota estricta
    final_strict = np.sum(grades_matrix * all_weights * u_factors, axis=1)
    
    # Nota simple (promedio ponderado)
    final_raw = np.sum(grades_matrix * all_weights, axis=1)
    
    # Lógica de fallo
    mask_fail = final_raw < threshold_fail
    final_grades = np.where(mask_fail, factor_penalizacion * final_raw, final_strict)
    
    # Aplicar ayuda de redondeo (2.90 -> 3.00)
    return _aplicar_ayuda_redondeo(final_grades)

def mostrar_detalle_estudiante(idx, df, config, U=None):
    """
    Muestra el cálculo detallado paso a paso para un estudiante específico,
    comparando el método simple y el avanzado.
    """
    row = df.loc[idx]
    
    # 1. Pesos y Configuración
    reg_weights, ft_weight, nc_weight = calcular_pesos(config)
    reg_items = config['items_normales']
    def_item = config['item_definitorio']
    
    weights_list = [reg_weights, [ft_weight]]
    reg_names = [item['nombre'] for item in reg_items]
    def_name = def_item['nombre']
    all_names = reg_names + [def_name]
    thresholds = [item.get('umbral', 2.5) for item in reg_items]
    thresholds.append(def_item.get('umbral', 2.5))

    if 'nota_concepto' in config:
        weights_list.append([nc_weight])
        all_names.append('nota_concepto')
        thresholds.append(config['nota_concepto'].get('umbral', 1e-20))
    
    all_weights = np.concatenate(weights_list)
    
    # Valores del estudiante
    grades = row[all_names].values
    
    print(f"\n{'='*60}")
    print(f"ESTUDIANTE ID: {idx}")
    print(f"{'='*60}")
    
    print("\n1. NOTAS Y PESOS:")
    for name, w, grade in zip(all_names, all_weights, grades):
        print(f"   {name:<15} : {grade:5.2f} (Peso: {w:.2%})")
        
    # Promedio Clásico
    mean_classic = np.sum(grades * all_weights)
    print(f"\n   PROMEDIO CLÁSICO : {mean_classic:.4f}")

    # --- MÉTODO SIMPLE ---
    print(f"\n{'='*20} MÉTODO SIMPLE {'='*20}")
    # Thresholds ya calculados arriba para metodo simple
    T_vector = np.array(thresholds)
    
    print("\n2. FACTORES DE UMBRAL (Simple Strict):")
    # ratios = clam(grade / threshold, 0, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
         ratios = np.clip(grades / T_vector, 0, 1)
         
    for n, r, t in zip(all_names, ratios, T_vector):
        print(f"   Ratio {n:<12} (Umbral {t}): {r:.2f}")
        
    n_items = len(all_names)
    u_factors_simple = np.zeros_like(grades)
    
    print("\n3. FACTORES DE UTILIDAD (U_i):")
    for i in range(n_items):
        # El factor U_i es el producto de los ratios de los OTROS items
        others_indices = [j for j in range(n_items) if j != i]
        
        details = []
        prod = 1.0
        for j in others_indices:
            val = ratios[j]
            prod *= val
            details.append(f"{all_names[j][:4]}({val:.2f})")
            
        u_factors_simple[i] = prod
        details_str = " * ".join(details)
        print(f"   U_{all_names[i]:<12} = {prod:.4f}  [= {details_str}]")
        
    strict_val = np.sum(grades * all_weights * u_factors_simple)
    print(f"\n   Nota Estricta (Sum Ponderada * U): {strict_val:.4f}")
    
    print("\n4. CÁLCULO FINAL (SIMPLE):")
    if mean_classic < 3.0:
        print(f"   [REPROBADO] Promedio ({mean_classic:.2f}) < 3.0")
        print(f"   -> Nota Final = 0.8 * {mean_classic:.4f} = {0.8 * mean_classic:.4f}")
    else:
        print(f"   [APROBADO] Promedio >= 3.0")
        print(f"   -> Nota Final = Nota Estricta ({strict_val:.4f})")

    # --- MÉTODO AVANZADO ---
    print(f"\n{'='*20} MÉTODO AVANZADO {'='*20}")
    
    # U Default Logic
    if U is None:
        U_act = T_vector # Same thresholds as simple
        print(f"   (Usando umbrales individuales como U)")
    else:
        U_act = U
        print(f"   (Usando U global = {U})")

    # G Factors
    G = _calcula_factor_umbral(grades.reshape(1, -1), U_act).flatten()
    
    # Floor
    X_reg = row[reg_names].values
    m_floor = np.min(X_reg)
    print(f"   Piso (mínimo items regulares): {m_floor:.2f}")

    U_factors_adv = np.zeros_like(grades)
    print("\n5. PESOS LOCALES INTERDEPENDIENTES (U_i):")
    for i in range(n_items):
        others_indices = [j for j in range(n_items) if j != i]
        
        details = []
        prod = 1.0
        for j in others_indices:
            val = G[j]
            prod *= val
            details.append(f"{all_names[j][:4]}({val:.2f})")
            
        U_factors_adv[i] = prod
        details_str = " * ".join(details)
        print(f"   U_{all_names[i]:<12} = {prod:.4f}  [= {details_str}]")
        
    S_local = np.sum(grades * all_weights * U_factors_adv)
    print(f"   Suma Local Penalizada (S_local): {S_local:.4f}")
    
    eps = 1e-9
    if mean_classic > eps:
        F_local = np.clip(S_local / mean_classic, 0, 1)
    else:
        F_local = 1.0
    print(f"   Factor Local (F_local = S_local/A): {F_local:.4f}")
    
    # Interpolation
    N_raw = m_floor + (mean_classic - m_floor) * F_local
    N_raw = max(N_raw, m_floor)
    print(f"   Nota Interpolada (Raw): {N_raw:.4f}")
    
    # Final Adjustments
    threshold_fail = 2.95
    if mean_classic < threshold_fail:
        final_adv = mean_classic
        print(f"   [BAJO UMBRAL FAIL] {mean_classic:.2f} < {threshold_fail}. Nota = Promedio.")
    else:
        final_adv = min(N_raw, mean_classic)
        print(f"   [NORMAL] Nota Final = min({N_raw:.4f}, {mean_classic:.4f}) = {final_adv:.4f}")
        
    # Verificar Ayuda de Redondeo
    final_adv_rounded = round(final_adv, 2)
    if 2.90 <= final_adv_rounded < 3.00:
        print(f"   [AYUDA] Nota {final_adv_rounded} está entre 2.90 y 3.00. Se ajusta a 3.00.")
        final_adv = 3.0


# ==========================================
# RUTINAS DE OPTIMIZACIÓN
# ==========================================

def preparar_variables_optimizacion(config):
    """
    Prepara los vectores y metadatos necesarios para la optimización de pesos.
    
    Identifica todos los items independientes (normales y nota_concepto) que definen
    rasgos de optimización (peso_min/max, sugerido).
    
    Args:
        config (dict): Configuración de evaluación.
        
    Returns:
        tuple: (x0, bounds, mapping_info)
            - x0 (np.array): Valor inicial (peso_sugerido) de las variables.
            - bounds (list): Lista de tuplas (min, max) para cada variable.
            - mapping_info (list): Lista de diccionarios con metadatos para mapear x[i] al config.
    """
    x0 = []
    bounds = []
    mapping_info = []
    
    # 1. Items Normales Independientes
    items = config['items_normales']
    required_keys = ['peso_min', 'peso_max', 'peso_sugerido']
    
    for i, item in enumerate(items):
        if 'correlacionado_con' not in item:
            if all(k in item for k in required_keys):
                x0.append(item['peso_sugerido'])
                bounds.append((item['peso_min'], item['peso_max']))
                mapping_info.append({'type': 'normal', 'index': i, 'name': item['nombre']})
                
    # 2. Nota Concepto (si existe)
    if 'nota_concepto' in config:
        nc = config['nota_concepto']
        # Claves pueden ser peso_minimo o peso_min. Estandarizamos chequeo
        p_min = nc.get('peso_minimo', nc.get('peso_min'))
        p_max = nc.get('peso_maximo', nc.get('peso_max'))
        p_sug = nc.get('peso_sugerido')
        
        if p_min is not None and p_max is not None and p_sug is not None:
             x0.append(p_sug)
             bounds.append((p_min, p_max))
             mapping_info.append({'type': 'concept', 'name': 'nota_concepto'})
             
    return np.array(x0), bounds, mapping_info

def actualizar_pesos_desde_vector(x, config_in, mapping_info):
    """
    Actualiza la configuración con los pesos del vector de optimización x.
    
    Args:
        x (np.array): Vector de pesos optimizados.
        config_in (dict): Configuración a actualizar (se modifica copia o in-place?). 
                          Recomendado pasar copia si se usa en loop.
        mapping_info (list): Metadatos retornados por preparar_variables_optimizacion.
        
    Returns:
        dict: Configuración actualizada.
    """
    config = config_in # Modificación in-place del objeto pasado (debe ser copia si se requiere)
    items = config['items_normales']
    
    # 1. Aplicar valores de x a la config
    for val, info in zip(x, mapping_info):
        if info['type'] == 'normal':
            idx = info['index']
            # Actualizamos 'peso' para que calcular_pesos lo tome como fijo
            items[idx]['peso'] = val 
        elif info['type'] == 'concept':
             if 'nota_concepto' in config:
                 config['nota_concepto']['peso'] = val
                 
    # 2. Recalcular pesos completos (propagación y normalización)
    # Esto actualiza peso_final internamente si llamamos autoconfigura, 
    # pero aquí quizás solo queremos asegurar que calcular_pesos lo tome.
    # El flujo de optimización suele usar calcular_pesos después.
    
    # Si queremos que la config refleje el estado completo:
    # (Opcional, pero útil para coherencia)
    try:
         w_reg, w_def, w_nc = calcular_pesos(config)
         for item, w in zip(items, w_reg):
             item['peso_final'] = w
         config['item_definitorio']['peso_final'] = w_def
         if 'nota_concepto' in config:
             config['nota_concepto']['peso_final'] = w_nc
    except:
        pass
        
    return config


# ==========================================
# ORÁCULO Y MÉTRICAS
# ==========================================

def calcula_decision_oraculo(row, config):
    """
    Determina si un estudiante aprueba (1) o reprueba (0) basándose en reglas heurísticas (Oráculo).
    
    Reglas:
    1. Definitorio < 2.0: Reprueba (0).
    2. 2.0 <= Definitorio < 2.5: Aprueba (1) SI todos los avanzados > 2.95.
    3. 2.5 <= Definitorio < 3.0: Aprueba (1) SI al menos 2 avanzados > 2.95.
    4. 3.0 <= Definitorio < 3.5: Reprueba (0) SI al menos 2 avanzados < 2.95.
    5. 3.5 <= Definitorio < 4.0: Reprueba (0) SI todos los normales (filtro) están perdidos (< 2.95).
    6. 4.0 <= Definitorio < 5.0: Reprueba (0) SI todos los avanzados están perdidos (< 2.95).
    7. En otro caso: Aprueba (1).
    
    Args:
        row (pd.Series): Fila del DataFrame con las notas de un estudiante.
        config (dict): Configuración de evaluación.
        
    Returns:
        int: 1 si aprueba, 0 si reprueba.
    """
    # Identificar items
    def_item_name = config['item_definitorio']['nombre']
    items_normales = config['items_normales']
    
    # Separar por tipos
    # Asumimos que 'items de filtro' se refiere a todos los items normales
    names_avanzados = [i['nombre'] for i in items_normales if i.get('tipo', 'avanzado') == 'avanzado']
    names_todos_normales = [i['nombre'] for i in items_normales]
    
    # Valores
    val_def = row[def_item_name]
    vals_adv = row[names_avanzados]
    vals_all_norm = row[names_todos_normales]
    
    # Umbral de aprobación estandár para chequeos internos
    PASS_THRESH = 2.95
    
    # 1. Def < 2.0 -> Perder
    if val_def < 2.0:
        return 0
    
    # 2. 2.0 <= Def < 2.5 -> Ganar SI todos avanzados > PASS
    elif 2.0 <= val_def < 2.5:
        if (vals_adv > PASS_THRESH).all():
            return 1
        return 0 # Si no cumple, pierde (asumo default behavior negativo en rangos bajos)
        
    # 3. 2.5 <= Def < 3.0 -> Ganar SI al menos 2 avanzados > PASS
    elif 2.5 <= val_def < 3.0:
        if (vals_adv > PASS_THRESH).sum() >= 2:
            return 1
        return 0

    # 4. 3.0 <= Def < 3.5 -> Perder SI al menos 2 avanzados < PASS
    elif 3.0 <= val_def < 3.5:
        if (vals_adv < PASS_THRESH).sum() >= 2:
            return 0
        return 1 # Default positivo en rango medio

    # 5. 3.5 <= Def < 4.0 -> Perder SI todos items normales perdidos
    elif 3.5 <= val_def < 4.0:
        if (vals_all_norm < PASS_THRESH).all():
            return 0
        return 1

    # 6. 4.0 <= Def < 5.0 -> Perder SI todos avanzados perdidos
    elif 4.0 <= val_def <= 5.0:
        if (vals_adv < PASS_THRESH).all():
            return 0
        return 1
        

    return 1

def genera_nota_concepto(row, config, pass_threshold=3.0):
    """
    Genera la nota de concepto basada en reglas heurísticas sobre el rendimiento general.
    
    Reglas:
    - 0.0: Todos los items avanzados son 0.0.
    - 2.0: Todos los items avanzados son < umbral (perdidos).
    - 2.5: Todos avanzados perdidos PERO todos fáciles ganados (>= umbral).
    - 3.0: Al menos un item avanzado ganado.
    - 3.5: Al menos un item avanzado ganado Y todos fáciles ganados.
    - 4.0: Al menos dos items avanzados ganados.
    - 4.5: Al menos dos items avanzados ganados Y todos fáciles ganados.
    - 5.0: Todos los ítems (avanzados + faciles) ganados.
    
    Args:
        row (pd.Series): Fila de notas de un estudiante.
        config (dict): Configuración de evaluación.
        pass_threshold (float): Nota mínima para considerar un item ganado (default 3.0).
        
    Returns:
        float: Nota de concepto generada.
    """
    items_normales = config['items_normales']
    
    # Identificar columnas
    names_avanzados = [i['nombre'] for i in items_normales if i.get('tipo', 'avanzado') == 'avanzado']
    names_faciles = [i['nombre'] for i in items_normales if i.get('tipo', 'avanzado') == 'facil']
    names_todos = [i['nombre'] for i in items_normales] # Todos items normales
    
    # Valores
    vals_adv = row[names_avanzados]
    vals_easy = row[names_faciles]
    vals_all = row[names_todos]
    
    # Pre-cálculos booleanos
    adv_zeros = (vals_adv == 0.0).all()
    adv_lost_all = (vals_adv < pass_threshold).all() 
    easy_won_all = (vals_easy >= pass_threshold).all()
    adv_won_count = (vals_adv >= pass_threshold).sum()
    all_won = (vals_all >= pass_threshold).all()
    
    # Evaluar Reglas (Orden Jerárquico Inverso comúnmente, o if-elif exclusivo)
    # Dado que las reglas son mutuamente excluyentes en su mayoría o escalonadas, 
    # evaluamos de mayor a menor nota para capturar la mejor categoría posible?
    # O seguimos el orden del enunciado estrictamente?
    # El enunciado da condiciones puntuales.
    # Vamos a evaluar condiciones de "más éxito" primero para dar la nota más alta posible?
    # Re-leyendo:
    # 5.0 si todos ganados.
    # 4.5 si 2 adv ganados y todos faciles ganados.
    # 4.0 si 2 adv ganados.
    # 3.5 si 1 adv ganado y todos faciles ganados.
    # 3.0 si 1 adv ganado.
    # 2.5 si adv perdidos y faciles ganados.
    # 2.0 si adv perdidos.
    # 0.0 si adv son 0.0.
    
    if all_won:
        return 5.0
    
    if adv_won_count >= 2:
        if easy_won_all:
            return 4.5
        return 4.0
        
    if adv_won_count >= 1:
        if easy_won_all:
            return 3.5
        return 3.0
        
    if adv_lost_all:
        # Check 0.0 condition (subset of lost) - Prioridad sobre el rescate de faciles
        if adv_zeros:
            return 0.0
            
        if easy_won_all:
            return 2.5
        
        return 2.0
        
    # Default fallback (si no cae en ninguna anterior, ej. no todos adv perdidos pero 0 ganados? 
    # Implica notas entre 0 y 3.0 pero no todos < 3.0? No, si count < 1, entonces todos < 3.0 (lost_all).
    # Entonces el caso 'adv_lost_all' cubre count=0.
    
    return 2.0 # Fallback por seguridad


def analisis_falsos_positivos_negativos(df, col_nota, config):
    """
    Calcula tasas de falsos positivos y negativos comparando una nota con el Oráculo.

    Args:
        df (pd.DataFrame): DataFrame con notas y (opcionalmente) columna 'Decision_Oraculo'.
        col_nota (str): Nombre de la columna con la nota a evaluar.
        config (dict): Configuración de evaluación.

    Returns:
        dict: Diccionario con métricas (FPR, FNR, Accuracy, etc.) y DataFrames de ejemplo.
    """
    # 1. Calcular Oráculo si no existe
    # Usamos una copia para no afectar el DF original si no se desea
    df_wk = df.copy()
    
    if 'Decision_Oraculo' not in df_wk.columns:
        df_wk['Decision_Oraculo'] = df_wk.apply(lambda row: calcula_decision_oraculo(row, config), axis=1)
    
    y_true = df_wk['Decision_Oraculo']
    y_pred = (df_wk[col_nota] >= 3.0).astype(int)
    
    # 2. Confusion Matrix
    TP = ((y_pred == 1) & (y_true == 1)).sum()
    TN = ((y_pred == 0) & (y_true == 0)).sum()
    FP = ((y_pred == 1) & (y_true == 0)).sum()
    FN = ((y_pred == 0) & (y_true == 1)).sum()
    
    total = len(df_wk)
    
    # 3. Metrics
    # FPR: De los que debían perder (0), cuántos ganaron (1)? -> FP / (FP + TN)
    div_fpr = (FP + TN)
    fpr = FP / div_fpr if div_fpr > 0 else 0.0
    
    # FNR: De los que debían ganar (1), cuántos perdieron (0)? -> FN / (FN + TP)
    div_fnr = (FN + TP)
    fnr = FN / div_fnr if div_fnr > 0 else 0.0
    
    accuracy = (TP + TN) / total if total > 0 else 0.0
    
    # 4. Examples
    fp_examples = df_wk[(y_pred == 1) & (y_true == 0)].copy()
    fn_examples = df_wk[(y_pred == 0) & (y_true == 1)].copy()
    
    return {
        'FPR': fpr,
        'FNR': fnr,
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'Accuracy': accuracy,
        'FP_Examples': fp_examples,
        'FN_Examples': fn_examples
    }


# ==========================================
# RUTINAS DE GENERACIÓN DE DATOS

# ==========================================
def generar_distribucion_bimodal(n, low_center=1.5, high_center=4.5, ratio=0.5, sigma=0.6):
    """
    Genera una distribución bimodal de notas.

    Args:
        n (int): Número de notas.
        low_center (float): Centro de la moda baja.
        high_center (float): Centro de la moda alta.
        ratio (float): Proporción de notas en la moda alta.
        sigma (float): Desviación estándar.

    Returns:
        np.array: Array de notas generadas [0, 5].
    """
    n_high = int(n * ratio)
    n_low = n - n_high
    high_grades = np.random.normal(high_center, sigma, n_high)
    low_grades = np.random.normal(low_center, sigma, n_low)
    combined = np.concatenate([high_grades, low_grades])
    np.random.shuffle(combined)
    return np.clip(combined, 0, 5)

def generar_distribucion_negative_skew(n, mode=4.5, sigma=1.0):
    """
    Genera una distribución con sesgo negativo (tendencia a notas altas).

    Args:
        n (int): Número de notas.
        mode (float): Moda de la distribución.
        sigma (float): No utilizado actualmente (revisar implementación).

    Returns:
        np.array: Array de notas.
    """
    raw = mode - np.random.exponential(scale=0.8, size=n)
    return np.clip(raw, 0, 5)

def generar_distribucion_normal(n, mean=3.0, sigma=1.0):
    """
    Genera una distribución normal truncada entre 0 y 5.

    Args:
        n (int): Número de notas.
        mean (float): Media.
        sigma (float): Desviación estándar.

    Returns:
        np.array: Array de notas.
    """
    return np.clip(np.random.normal(mean, sigma, n), 0, 5)

def generar_distribucion_exam(n, mean=2.8, sigma=1.1):
    """
    Genera notas para un examen con parámetros específicos.

    Args:
        n (int): Número de notas.
        mean (float): Media.
        sigma (float): Desviación estándar.

    Returns:
        np.array: Array de notas.
    """
    return np.clip(np.random.normal(mean, sigma, n), 0, 5)

def genera_datos(config, N=2000):
    """
    Genera un DataFrame de pandas con notas simuladas basadas en la configuración.

    Args:
        config (dict): Configuración de evaluación.
        N (int): Número de estudiantes a simular.

    Returns:
        pd.DataFrame: DataFrame con columnas por cada item y N filas.

    Ejemplo:
        >>> df = genera_datos(config, N=100)
        >>> df.head()
           Quiz_1  Tarea_1 ...
        0    4.2      3.8  ...
    """
    data = {}
    
    def generate_column_by_type(dtype):
        if dtype == 'clave':
            return generar_distribucion_bimodal(N, low_center=1.2, high_center=4.0, ratio=0.4)
        elif dtype == 'facil':
            return generar_distribucion_negative_skew(N, mode=4.5)
        elif dtype == 'examen':
            return generar_distribucion_exam(N, mean=2.8, sigma=1.1)
        else: # avanzado / normal
            return generar_distribucion_normal(N, mean=3.0, sigma=1.0)

    # Items Normales
    for item in config['items_normales']:
        dtype = item.get('tipo', 'avanzado')
        data[item['nombre']] = generate_column_by_type(dtype)
        
    # Item Definitorio
    def_item = config['item_definitorio']
    dtype = def_item.get('tipo', 'avanzado')
    data[def_item['nombre']] = generate_column_by_type(dtype)
    
    df = pd.DataFrame(data)
    
    # Generar Nota Concepto si existe en la configuración
    if 'nota_concepto' in config:
        # Se calcula fila por fila usando la lógica definida
        df['nota_concepto'] = df.apply(lambda row: genera_nota_concepto(row, config), axis=1)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df.round(2)

def mostrar_notas(df, config, n=None):
    """
    Retorna un DataFrame con las notas formateadas y cabeceras informativas (Peso %, Tipo).
    Útil para inspección visual de datos para humanos.
    
    Args:
        df (pd.DataFrame): DataFrame con las notas originales.
        config (dict): Configuración de evaluación.
        n (int, optional): Número de filas a mostrar (aleatorias). Si es None, muestra todo.
        
    Returns:
        pd.DataFrame: DataFrame formateado con nuevas cabeceras.
    """
    # 1. Calcular Pesos Actuales
    reg_weights, w_final, w_nc = calcular_pesos(config)
    
    # 2. Construir Mapping de Columnas y Cabeceras
    cols_ordered = []
    headers = []
    
    # Items Normales
    for i, item in enumerate(config['items_normales']):
        w_p = reg_weights[i] * 100.0
        t_orig = item.get('tipo', '?')
        # Formato: Nombre [15.0%, avanzado]
        header_str = f"{item['nombre']} [{w_p:.1f}%, {t_orig}]"
        
        cols_ordered.append(item['nombre'])
        headers.append(header_str)
        
    # Examen Final (Definitorio)
    def_item = config['item_definitorio']
    w_f_p = w_final * 100.0
    t_def = def_item.get('tipo', 'examen')
    header_def = f"{def_item['nombre']} [{w_f_p:.1f}%, {t_def}]"
    
    cols_ordered.append(def_item['nombre'])
    headers.append(header_def)
    
    # Nota Concepto (si existe en config)
    if 'nota_concepto' in config:
        nc = config['nota_concepto']
        w_nc_p = w_nc * 100.0
        header_nc = f"Nota_Concepto [{w_nc_p:.1f}%, concepto]"
        # Solo agregar si existe en el DF (a veces se genera, a veces no)
        if 'nota_concepto' in df.columns:
            cols_ordered.append('nota_concepto')
            headers.append(header_nc)
        elif 'Nota_Concepto' in df.columns:
            cols_ordered.append('Nota_Concepto')
            headers.append(header_nc)

    # Columnas Extra (Calculadas) si existen
    extras = ['Promedio_Clasico', 'Nota_Final', 'Decision_Oraculo']
    for col in extras:
        if col in df.columns:
            cols_ordered.append(col)
            headers.append(col)
            
    # 3. Filtrar y Renombrar
    # Verificar que las columnas existan
    valid_cols = [c for c in cols_ordered if c in df.columns]
    valid_headers = [h for c, h in zip(cols_ordered, headers) if c in df.columns]
    
    df_view = df[valid_cols].copy()
    
    if n is not None and len(df_view) > n:
        df_view = df_view.sample(n=n, random_state=42)
        
    df_view.columns = valid_headers
    
    return df_view.round(2)
