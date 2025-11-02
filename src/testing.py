import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from .config import *
from .config import (
    MES_TEST, MES_TRAIN,GANANCIA_ACIERTO, COSTO_ESTIMULO
)

logger = logging.getLogger(__name__)

def evaluar_en_test(df, mejores_params, best_iter=None) -> dict:
    """
    Evalúa el modelo con los mejores hiperparámetros en el conjunto de test.
    Solo calcula la ganancia, sin usar sklearn.
  
    Args:
        df: DataFrame con todos los datos
        mejores_params: Mejores hiperparámetros encontrados por Optuna
  
    Returns:
        dict: Resultados de la evaluación en test (ganancia + estadísticas básicas)
    """
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    logger.info(f"Período de test: {MES_TEST}")
  
    # Preparar datos de entrenamiento
    df_train_completo = df[df['foto_mes'].isin(MES_TRAIN)]
    df_test = df[df['foto_mes'].isin(MES_TEST)]

    X_train = df_train_completo.drop(columns=['clase_ternaria','clase_peso','clase_binaria2'])
    y_train = df_train_completo['clase_binaria2']
    w_train = df_train_completo['clase_peso']
    
    X_test = df_test.drop(['clase_ternaria', 'clase_peso','clase_binaria2'], axis=1)
    y_test_class = df_test['clase_ternaria']

    # Entrenar modelo con mejores parámetros
    
    params = mejores_params.copy()
    params.update({
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        'seed': SEMILLA[0],
        'verbosity': -1
    })
    
    train_data = lgb.Dataset(X_train,
                            label=y_train, weight=w_train)

    best_iter = int(best_iter)
  
    # Modelo y predicción
    model_test = lgb.train(params,
                train_data,
                num_boost_round=best_iter)
    
    y_pred_test = model_test.predict(X_test)

    # Ganancia y orden
    ganancia = np.where(y_test_class == 'BAJA+2', GANANCIA_ACIERTO, 0) - np.where(y_test_class != 'BAJA+2', COSTO_ESTIMULO, 0)
    order = np.argsort(y_pred_test)[::-1] #ordeno por probabilidad
    ganancia_ord = ganancia[order]
    ganancia_cum = np.cumsum(ganancia_ord)
  
    # Ventana de análisis
    piso_envios = 4000
    techo_envios = 20000

    # Ganancia máxima y corte
    ganancia_max = ganancia_cum.max()
    corte_optimo = np.where(ganancia_cum == ganancia_max)[0][0] 

    logger.info(f"Ganancia máxima: {ganancia_max:,.0f} en corte {corte_optimo}")

    # Guardar gráfico
    os.makedirs("graficos_test", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_png = f"graficos_test/{STUDY_NAME}_curva_ganancia_{ts}.png"

    plt.figure(figsize=(10, 6))
    plt.plot(range(piso_envios, len(ganancia_cum[piso_envios:techo_envios]) + piso_envios), ganancia_cum[piso_envios:techo_envios], label='Ganancia LGBM')
    plt.axvline(x=corte_optimo, color='g', linestyle='--', label=f'Punto de corte a la ganancia máxima {corte_optimo}')
    plt.axhline(y=ganancia_max, color='r', linestyle='--', label=f'Ganancia máxima {ganancia_max}')
    plt.title('Curva de Ganancia')
    plt.xlabel('Clientes')
    plt.ylabel('Ganancia')
    plt.legend()
    plt.savefig(ruta_png, dpi=120)
    plt.close()

    return {"ganancia_máxima": ganancia_max, "corte_optimo": corte_optimo}

