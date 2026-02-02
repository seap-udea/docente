def objective_function(x_independent, df, config):
    # 1. Reconstruct full weights
    weights_regular = reconstruct_full_weights(x_independent, config)
    
    # 2. Get Ground Truth
    y_true = get_oracle_decisions_general(weights_regular, df, config)
    
    # 3. Calculate Grades
    y_pred, _ = calculate_grade_strict_general(weights_regular, df, config)
    
    # 4. Error
    pass_th = 2.95
    err_pass = np.maximum(0, pass_th - y_pred[y_true==1])**2
    err_fail = np.maximum(0, y_pred[y_true==0] - pass_th)**2
    
    return np.sum(err_pass) + np.sum(err_fail)

config_evaluacion = {
    "regular_items": [
        # MÓDULO 1: Intro (Easy)
        {
            "name": "Quiz_1", 
            "weakness": 3.5,
            "correlated_with": "Tarea_1", "correlation_factor": 0.5,
            "type": "easy" 
        },
        {
            "name": "Tarea_1", 
            "weakness": 2.5,
            "min": 0.05, "max": 0.20, "suggested": 0.15, 
            "type": "advanced"
        },
        
        # MÓDULO 2: Filtro (Filter) - Correlacionado con M1
        {
            "name": "Quiz_2", 
            "correlated_with": "Quiz_1", "correlation_factor": 1.2,
            "weakness": 3.5,
            "type": "easy"
        },
        {
            "name": "Tarea_2", 
            "correlated_with": "Tarea_1", "correlation_factor": 1.2,
            "weakness": 2.5,
            "type": "filter"
        },
        
        # MÓDULO 3: Avanzado (Normal) - Correlacionado con M1 (Factor 1.0)
        {
            "name": "Quiz_3", 
            "correlated_with": "Quiz_1", "correlation_factor": 1.0,
            "weakness": 3.5,
            "type": "easy"
        },
        {
            "name": "Tarea_3", 
            "weakness": 2.5,
            "correlated_with": "Tarea_1", "correlation_factor": 1.0,
            "type": "advanced"
        },
    ],
    "definitory_item": {
        "name": "Examen Final",
        "sudden_death": 2.50,
        "max_weight": 0.40,
        "type": "exam"
    }
}

def objective_function(x_independent, df, config):
    weights_regular = reconstruct_full_weights(x_independent, config)
    y_true = get_oracle_decisions_general(weights_regular, df, config)
    y_pred, _ = calculate_grade_strict_general(weights_regular, df, config)
    
    pass_th = 2.95
    # False Negatives (Should Win, Failed)
    # Effectively minimizing Adjustment Error (Distance to 2.95)
    err_fn = np.maximum(0, pass_th - y_pred[y_true==1])**2
    
    # False Positives (Should Fail, Won)
    # Balanced Error (No extra penalty multiplier)
    err_fp = np.maximum(0, y_pred[y_true==0] - pass_th)**2

    # Incluir el FPR y FNR en la minimización
    grade_final, _ = calculate_grade_strict_general(weights_regular, df, config)

    # 1. FPR
    fp = ((y_true == 0) & (grade_final >= 2.95)).sum()
    tn = ((y_true == 0) & (grade_final < 2.95)).sum()
    real_negatives = fp + tn
    fpr = fp / (real_negatives + 1e-9)
    
    # 2. FNR
    fn = ((y_true == 1) & (grade_final < 2.95)).sum()
    tp = ((y_true == 1) & (grade_final >= 2.95)).sum()
    real_positives = fn + tp
    fnr = fn / (real_positives + 1e-9)
    
    # Total Balanced Loss
    return np.sum(err_fn) / len(y_true) + np.sum(err_fp) / len(y_true) + fpr +  fnr


# 4. ORÁCULO DINÁMICO
def get_oracle_decisions_general(weights_regular, df, config):
    reg_items = config['regular_items']
    def_item = config['definitory_item']
    
    w_sum_reg = np.sum(weights_regular)
    w_def = 1.0 - w_sum_reg
    
    reg_names = [item['name'] for item in reg_items]
    def_name = def_item['name']
    
    # Scores
    reg_scores = df[reg_names].values
    score_regular = np.sum(reg_scores * weights_regular, axis=1) / (w_sum_reg + 1e-9)
    score_def = df[def_name].values
    
    decisions = np.ones(len(df), dtype=int)
    
    # 1. Sudden Death
    decisions[score_def < def_item['sudden_death']] = 0
    
    # 2. Weakness
    weakness_thresholds = np.array([item['weakness'] for item in reg_items])
    agg_weakness_thresh = np.sum(weakness_thresholds * weights_regular) / (w_sum_reg + 1e-9)
    
    mask_weak = (score_def >= def_item['sudden_death']) &                 (score_def < 3.0) &                 (score_regular < agg_weakness_thresh)
    decisions[mask_weak] = 0
    
    return decisions

# B. Helper para Derivar Pesos

Vamos ahora a crear la rutina de oráculo en un nuevo notebook que se llama "ejemplo_oraculo.ipynb". 

En esta rutina el profesor va a definir los criterios que le llevan a definir si un estudiante debería recibir un aprobatorio o no dependiendo del conjunto de sus notas (no del promedio). Vamos a programar unas reglas básicas. Por ejemplo:

- Si el "item_definitorio" está por debajo de 2.0 definitivamente el estudiante no debe ganar (decision=1).

- Si el "item_definitorio" estar entre 2.0 y 2.5 revisa si los ítems avanzados están todos por encima de aprobatorio. En ese caso debe ganar (decision=1).

- Si el "item_definitorio" estar entre 2.5 y 3.0 revisa si al menos dos de los ítems avanzados están por encima de aprobatorio. En ese caso debe ganar (decision=1).

- Si el "item_definitorio" estar entre 3.0 y 3.5 pero al menos dos items avanzados fueron reprobados entonces debe perder (decision=0)

- Si el "item_definitorio" estar entre 3.5 y 4.0 pero todos los items de filtro están perdidos debe perder (decision=0)

- Si el "item_definitorio" estar entre 4.0 y 5.0 pero todos los items avanzados están perdidos debe perder (decision=0)

Utiliza la rutina de oraculo para crear un DataFrame con las notas generadas aleatoriamente, el promedio simple, el promedio con umbral simple, el promedio con umbral avanzado y lo que dice el oraculo. Este notebook será usado para las tareas de aprendizaje de máquina