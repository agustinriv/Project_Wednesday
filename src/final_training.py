import pandas as pd
import lightgbm as lgb
import numpy as np
import logging
import os
from datetime import datetime
from .config import FINAL_TRAIN, FINAL_PREDIC, SEMILLA
from .best_params import cargar_mejores_hiperparametros
from .gain_function import ganancia_lgb_binary

logger = logging.getLogger(__name__)

def preparar_datos_entrenamiento_final(df: pd.DataFrame) -> tuple:
    """
    Prepara los datos para el entrenamiento final usando todos los períodos de FINAL_TRAIN.
  
    Args:
        df: DataFrame con todos los datos
  
    Returns:
        tuple: (X_train, y_train, X_predict, clientes_predict)
    """
    logger.info(f"Preparando datos para entrenamiento final")
    logger.info(f"Períodos de entrenamiento: {FINAL_TRAIN}")
    logger.info(f"Período de predicción: {FINAL_PREDIC}")
  
    # Datos de entrenamiento: todos los períodos en FINAL_TRAIN
  
    # Datos de predicción: período FINAL_PREDIC 

    logger.info(f"Registros de entrenamiento: {len(df_train):,}")
    logger.info(f"Registros de predicción: {len(df_predict):,}")
  
    #Corroborar que no esten vacios los df

    # Preparar features y target para entrenamiento
  
    X_train 
    y_train 

    # Preparar features para predicción
    X_predict 
    clientes_predict 

    logger.info(f"Features utilizadas: {len(features_cols)}")
    logger.info(f"Distribución del target - 0: {(y_train == 0).sum():,}, 1: {(y_train == 1).sum():,}")
  
    return X_train, y_train, X_predict, clientes_predict

def entrenar_modelo_final(X_train: pd.DataFrame, y_train: pd.Series, mejores_params: dict) -> lgb.Booster:
    """
    Entrena el modelo final con los mejores hiperparámetros.
  
    Args:
        X_train: Features de entrenamiento
        y_train: Target de entrenamiento
        mejores_params: Mejores hiperparámetros de Optuna
  
    Returns:
        lgb.Booster: Modelo entrenado
    """
    logger.info("Iniciando entrenamiento del modelo final")
  
    # Configurar parámetros del modelo
    params = {
        'objective': 'binary',
        'metric': 'None',  # Usamos nuestra métrica personalizada
        'random_state': SEMILLA[0] if isinstance(SEMILLA, list) else SEMILLA,
        'verbose': -1,
        **mejores_params  # Agregar los mejores hiperparámetros
    }
  
    logger.info(f"Parámetros del modelo: {params}")
  
    # Crear dataset de LightGBM
  
    # Entrenar modelo con lgb.train()

    return modelo

def generar_predicciones_finales(
    modelo: lgb.Booster,
    data: pd.DataFrame,
    dtrain_final: lgb.Dataset,
    mes_pred: int,
    k_corte: int = 9500,
    id_col: str = 'numero_de_cliente',
    output_csv: str | None = None
) -> dict[str, pd.DataFrame]:
    """
    Genera predicciones finales para un mes objetivo, seleccionando los top-k clientes
    con mayor probabilidad según un modelo de LightGBM entrenado.

    Args:
        modelo: Modelo LightGBM entrenado.
        data: DataFrame original con todas las filas.
        dtrain_final: Dataset de LightGBM utilizado para obtener el orden de las features.
        mes_pred: Valor del campo 'foto_mes' a predecir.
        k_corte: Cantidad de clientes con mayor probabilidad a seleccionar.
        id_col: Nombre de la columna ID del cliente.
        output_csv: Si se indica, guarda el submit a CSV con ese nombre.

    Returns:
        dict con:
            - 'predicciones': DataFrame con columnas [id, foto_mes, prob_predicha, rank, pred_bin]
            - 'submit': DataFrame listo para enviar con columnas [id, Predicted]
    """
    logger.info(f"Generando predicciones para foto_mes={mes_pred} (top_k={k_corte})")

    # Seleccionar features y datos del período
    features = dtrain_final.feature_name
    pred_final = data[data['foto_mes'] == mes_pred].copy()
    X_pred = pred_final.loc[:, features]

    # Generar probabilidades
    probs = modelo.predict(X_pred, num_iteration=modelo.best_iteration)
    logger.info(f"Predicciones generadas: {len(probs):,} filas")

    # Ordenar de mayor a menor probabilidad
    order = np.argsort(probs)[::-1]
    idx_contactar = order[:k_corte]

    # Armar DataFrame base
    if id_col not in pred_final.columns:
        logger.warning(f"No se encontró la columna '{id_col}', se usará el índice como ID.")
        id_values = pred_final.index.values
        id_col = 'idx'
    else:
        id_values = pred_final[id_col].values

    salida = pd.DataFrame({
        id_col: id_values,
        'foto_mes': pred_final['foto_mes'].values,
        'prob_predicha': probs
    })

    # Calcular ranking
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    salida['rank'] = rank

    # Generar vector binario de predicción
    pred_bin = np.zeros(len(pred_final), dtype=int)
    pred_bin[idx_contactar] = 1
    salida['pred_bin'] = pred_bin

    # Crear DataFrame de submit
    submit = pd.DataFrame({
        id_col: id_values.astype(int),
        'Predicted': pred_bin.astype(int)
    })

    # Guardar CSV si corresponde
    if output_csv:
        submit.to_csv(output_csv, index=False)
        logger.info(f"Archivo CSV guardado en: {output_csv}")

    return {'predicciones': salida, 'submit': submit}