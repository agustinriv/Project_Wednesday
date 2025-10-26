import optuna
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import os
import logging
from .config import (
    SEMILLA, MES_TRAIN, STUDY_NAME,
    GANANCIA_ACIERTO, COSTO_ESTIMULO, PARAMETROS_LGB
)
from .gain_function import ganancia_evaluator
from datetime import datetime

logger = logging.getLogger(__name__)

def objetivo_ganancia_cv(trial, df) -> float:
    """
    Función objetivo para Optuna con Cross Validation.
    Utiliza SEMILLA[0] desde configuración para reproducibilidad.
  
    Args:
        trial: Trial de Optuna
        df: DataFrame con datos
  
    Returns:
        float: Ganancia promedio del CV
    """
    # Hiperparámetros a optimizar (desde configuración YAML)
    params = {
        'objective': 'binary',
        'metric': 'custom',  # Usamos nuestra métrica personalizada
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'num_leaves': trial.suggest_int('num_leaves', PARAMETROS_LGB['num_leaves']["min"], PARAMETROS_LGB["num_leaves"]["max"]),
        'learning_rate': trial.suggest_float('learning_rate', PARAMETROS_LGB['learning_rate']["min"], PARAMETROS_LGB['learning_rate']["max"], log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', PARAMETROS_LGB['feature_fraction']["min"], PARAMETROS_LGB['feature_fraction']["max"]),
        'bagging_fraction': trial.suggest_float('bagging_fraction', PARAMETROS_LGB['bagging_fraction']["min"], PARAMETROS_LGB['bagging_fraction']["max"]),
        'max_bin': 31,
        'seed': SEMILLA[0],  # Desde configuración YAML
        'verbosity': -1
    }

    # Preparar datos para CV
    df_cv = df[df['foto_mes'].isin(MES_TRAIN)]

    # Features y target
    X_train = df_cv.drop(['clase_ternaria', 'target'], axis=1)
    y_train = df_cv['clase_ternaria']
  
    # Crear dataset de LightGBM
    dataset = lgb.Dataset(X_train, label=y_train)
  
    # Configurar CV con semilla desde configuración
    cv_results = lgb.cv(
        params,
        dataset,
        num_boost_round=1000,
        nfold=5,
        seed= SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        stratified=True,
        feval=ganancia_evaluator,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
  
    # Extraer ganancia promedio y max
    ganancias_cv = cv_results['valid ganancia-mean']
    ganancia_maxima = float(np.max(ganancias_cv))  

    # Mejor iteración
    best_iteration = len(ganancias_cv) - 1

    logger.debug(f"Trial {trial.number}: Ganancia CV = {ganancia_maxima:,.0f}")
    logger.debug(f"Trial {trial.number}: Mejor iteración = {best_iteration}")

    # Guardar iteración para análisis posterior
    guardar_iteracion_cv(trial, ganancia_maxima, ganancias_cv, best_iteration=best_iteration)

    return ganancia_maxima

def guardar_iteracion_cv(trial, ganancia_maxima, ganancias_cv, best_iteration=None, archivo_base=None):

    if archivo_base is None:
        archivo_base = STUDY_NAME

    archivo = f"resultados/{archivo_base}_iteraciones.json"
    os.makedirs("resultados", exist_ok=True)

    # Datado de iteración
    iteracion_data = {
        'trial_number': trial.number,
        'params': trial.params,
        'best_iteration': int(best_iteration) if best_iteration is not None else None,
        'value': float(ganancia_maxima),
        'datetime': datetime.now().isoformat(),
        'state': 'COMPLETE'
    }

    # Cargar datos existentes
    if os.path.exists(archivo):
        with open(archivo, 'r') as f:
            try:
                datos_existentes = json.load(f)
                if not isinstance(datos_existentes, list):
                    datos_existentes = []
            except json.JSONDecodeError:
                datos_existentes = []
    else:
        datos_existentes = []

    # Agregar nueva iteración a los datos existentes
    datos_existentes.append(iteracion_data)

    # Guardar nuevamente en el archivo
    with open(archivo, 'w') as f:
        json.dump(datos_existentes, f, indent=4)

    logger.info(f"Iteración CV {trial.number} guardada - Ganancia: {ganancia_maxima:,.0f}")

def optimizar_con_cv(df, n_trials=3) -> optuna.Study:
    """
    Ejecuta optimización bayesiana con Cross Validation.
  
    Args:
        df: DataFrame con datos
        n_trials: Número de trials a ejecutar
  
    Returns:
        optuna.Study: Estudio de Optuna con resultados de CV
    """
    study_name = f"{STUDY_NAME}"
  
    # Crear estudio
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA)
    )
  
    # Ejecutar optimización
    study.optimize(lambda trial: objetivo_ganancia_cv(trial, df), n_trials=n_trials)
  
    # Resultados
    logger.info(f"Mejor ganancia: {study.best_value:,.0f}")
    logger.info(f"Mejores parámetros: {study.best_params}")
    logger.info(f"Mejores parámetros: {study.best_params}")
  
    return study