import pandas as pd
import os
import datetime
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

from src.loader import cargar_datos
from src.features import feature_engineering_lag, feature_engineering_delta, obtener_columnas_validas
from src.data_drifting import drift_inf, ind
from src.target import clase_ternaria, pivot_clase_ternaria
from src.fe_intrames import fe_intrames

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
    logger.info("=== INICIANDO PIPELINE CON CONFIGURACIÓN YAML ===")
  
    # 1. Cargar datos
    df = cargar_datos(DATA_PATH)

    # 2. Target y Pivot (por si necesito)
    df = clase_ternaria("~/buckets/b1/datasets/competencia_01_crudo.csv.gz")
    #df = pivot_clase_ternaria(df)

    #3. Eliminación Features
    df.drop(columns=['mprestamos_personales', 'cprestamos_personales'], inplace=True)
    logger.info(f"Etapa completada: {df.shape}")

    #4. Data Quality
    # COMPLETAR

    #5. Data Drifting
    # Defino campos monetarios a usar y aplico función
    campos_monetarios = [col for col in df.columns if col.startswith(('m', 'Visa_m', 'Master_m', 'vm_m'))]
    df = drift_inf(df, campos_monetarios, ind)

    #6. Feature Engineering Intra-Mes
    df = fe_intrames(df)
    logger.info(f"Etapa completada: {df.shape}")

    #7. Feature Engineering Histórico
    atributos = obtener_columnas_validas(df)
    cant_lag = 2
    cant_delta = 2
    df = feature_engineering_lag(df, atributos, cant_lag)
    df = feature_engineering_delta(df_fe, atributos, cant_delta)  
    logger.info(f"Etapa completada: {df.shape}")

    #8. Output en parquet
    df.to_parquet("df.parquet", index=False) 

    logger.info(f">>> Workflow A completado. Continuar con la siguiente etapa")

if __name__ == "__main__":
    main()