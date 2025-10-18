import pandas as pd
import os
import datetime
import logging

from src.loader import cargar_datos
from src.features import feature_engineering_lag

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
    logger.info(f"Inicio de ejecución")
    print("Inicio de ejecución")

    #00 Cargar dataset
    os.makedirs("datasets", exist_ok=True)
    path = "datasets/competencia_01_test.csv"
    df = cargar_datos(path)

    #01 Lags
    atributos = ["mrentabilidad"]
    cant_lag = 2
    df = feature_engineering_lag(df, columnas=atributos, cant_lag=cant_lag)
  
    #02 Guardar datos
    path = "datasets/competencia_01_lag.csv"
    df.to_csv(path, index=False)

    logger.info(f">>> Ejecución finalizada. Revisar logs para más detalle. {nombre_log}")

if __name__ == "__main__":
    main()