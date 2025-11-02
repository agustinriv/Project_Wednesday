import pandas as pd
import os
import datetime
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

from src.loader import cargar_datos, convertir_clase_pesos
from src.features import feature_engineering_lag, feature_engineering_delta, obtener_columnas_validas
from src.optimization_cv import optimizar_con_cv
from src.testing import evaluar_en_test
from src.best_params import cargar_mejores_hiperparametros

from src.config import *

## config basico logging
os.makedirs("logs", exist_ok=True)

fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
nombre_log = f"log_{fecha}.log"
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s %(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/{nombre_log}", mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

## Funcion principal
def main():
    ##Pipeline principal con optimización usando configuración YAML.
    logger.info("=== INICIANDO OPTIMIZACIÓN CON CONFIGURACIÓN YAML ===")
  
    # 1. Cargar datos
    df = cargar_datos(DATA_PATH)

    # 2. Feature Engineering
    atributos = obtener_columnas_validas(df)
    cant_lag = 2
    cant_delta = 2
    df_fe = feature_engineering_lag(df, atributos, cant_lag)
    df_fe = feature_engineering_delta(df_fe, atributos, cant_delta)

    logger.info(f"Feature Engineering completado: {df_fe.shape}")

    print(df_fe).head(10)
    """
    #02 Convertir clase_ternaria a target binario
    df_fe = convertir_clase_pesos(df_fe)

    #03 Ejecutar optimizacion de hiperparametros
    study = optimizar_con_cv(df_fe, n_trials=3)
 
    #04 Análisis adicional
    logger.info("=== ANÁLISIS DE RESULTADOS ===")
    trials_df = study.trials_dataframe()
    if len(trials_df) > 0:
        top_5 = trials_df.nlargest(5, 'value')
        logger.info("Top 5 mejores trials:")
        for idx, trial in top_5.iterrows():
            logger.info(f"  Trial {trial['number']}: {trial['value']:,.0f}")

    logger.info("=== OPTIMIZACIÓN COMPLETADA ===")

    #05 Test en mes desconocido
    logger.info("=== EVALUACIÓN EN CONJUNTO DE TEST ===")
    # Cargar mejores hiperparámetros
    mejores_params, best_iter = cargar_mejores_hiperparametros()
  
    # Evaluar en test
    resultados_test = evaluar_en_test(df_fe, mejores_params, best_iter)
  
    # Resumen de evaluación en test
    logger.info("===EVALUACIÓN EN TEST FINALIZADA===")
    
    logger.info(f">>> Ejecución finalizada. Revisar logs para mas detalles.")
    """
if __name__ == "__main__":
    main()