import logging
import os
import datetime
import lightgbm as lgb
import numpy as np

from src.loader import cargar_datos, convertir_clase_ternaria_a_target
from src.features import feature_engineering_lag, feature_engineering_delta, obtener_columnas_validas
from src.config import *

os.makedirs("logs", exist_ok=True)
fecha = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
nombre_log = f"log_final_{fecha}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== ENTRENAMIENTO FINAL ===")

    # Parámetros
    TRAIN_f = [202101,202102,202103,202104]
    MES_PRED  = 202106
    best_iteration = 223
    CORTE_OPTIMO = 9500  

    # Cargar datos
    df = cargar_datos(DATA_PATH)

    # Feature Engineering
    atributos = obtener_columnas_validas(df)
    df = feature_engineering_lag(df, atributos, 2)
    logger.info(f"Dataset post-FE: {df.shape}")
    df = feature_engineering_delta(df, atributos, 2)
    logger.info(f"Dataset post-FE: {df.shape}")

    # Agrego pesos
    df['clase_peso'] = 1.0
    df.loc[df['clase_ternaria'] == 'BAJA+2', 'clase_peso'] = 1.00002
    df.loc[df['clase_ternaria'] == 'BAJA+1', 'clase_peso'] = 1.00001

    # Clase binaria
    df['clase_binaria2'] = 0
    df['clase_binaria2'] = np.where(df['clase_ternaria'] == 'CONTINUA', 0, 1)

    logger.info(f"Dataset post-FE: {df.shape}")

    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'first_metric_only': True,
        'boost_from_average': True,
        'feature_pre_filter': False,
        'max_bin': 31,
        "num_leaves": 174,
        "learning_rate": 0.036408091013602206,
        "min_data_in_leaf": 1892,
        "feature_fraction": 0.41518080123895895,
        "bagging_fraction": 0.6990945577012483,
        'seed': SEMILLA[0],
        'verbosity': -1
    }

    # Preparo datos para entrenamiento
    train_final = df[df['foto_mes'].isin(TRAIN_f)]
    X_final = train_final.drop(columns=['clase_ternaria','clase_peso','clase_binaria2'])
    y_final = train_final['clase_binaria2']
    w_final = train_final['clase_peso']

    dtrain_final = lgb.Dataset(X_final, label=y_final, weight=w_final)

    # Entreno modelo final
    model_final = lgb.train(params,
                    dtrain_final,
                    num_boost_round=best_iteration)

    # Generar predicciones
    df_pred = df[df['foto_mes'] == MES_PRED].copy()
    if df_pred.empty:
        logger.error(f"No hay filas para MES_PRED={MES_PRED}. Abortando.")
        return

    # ID a exportar (ajustá si tu columna se llama distinto)
    ID_COL = 'numero_de_cliente'
    if ID_COL not in df_pred.columns:
        # Si no existe, creamos un id reproducible
        df_pred[ID_COL] = np.arange(len(df_pred))

    X_predict = df_pred.drop(columns=['clase_ternaria', 'target', 'clase_peso','clase_binaria2'], errors='ignore')
    logger.info(f"Prediciendo sobre MES_PRED={MES_PRED} (n={len(df_pred)}) ...")
    df_pred['prob'] = model_final.predict(X_predict)

    # Ordenar por probabilidad desc (estable por si hay empates)
    df_pred.sort_values('prob', ascending=False, inplace=True, kind='mergesort')

    # Aplicar corte manual
    n_total = len(df_pred)
    corte = min(int(CORTE_OPTIMO), n_total)
    df_pred['Predicted'] = 0
    df_pred.iloc[:corte, df_pred.columns.get_loc('Predicted')] = 1

    # Armar submit (id + Predicted)
    df_submit = df_pred[[ID_COL, 'Predicted']].copy()

    # Guardar CSV (solo fecha y hora en el nombre)
    os.makedirs('predicciones', exist_ok=True)
    ts_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_salida = os.path.join('predicciones', f'{ts_name}.csv')
    df_submit.to_csv(archivo_salida, index=False)

    # Logs resumen
    cant_ones = int(df_submit['Predicted'].sum())
    cant_ceros = int(len(df_submit) - cant_ones)
    logger.info("✅ Entrenamiento final completado")
    logger.info(f"📁 Archivo guardado: {archivo_salida}")
    logger.info(f"Filas totales: {len(df_submit):,}")
    logger.info(f"Submit (Predicted=1): {cant_ones:,}")
    logger.info(f"No submit (Predicted=0): {cant_ceros:,}")
    logger.info(f"CORTE_OPTIMO: {CORTE_OPTIMO}, best_iteration: {best_iteration}")

if __name__ == "__main__":
    main()