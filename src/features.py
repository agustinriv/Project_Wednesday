import pandas as pd
import duckdb
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

def obtener_columnas_validas(df: pd.DataFrame,
                             excluir: Optional[List[str]] = None) -> List[str]:
    """
    Devuelve la lista de columnas numéricas o válidas para feature engineering.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos
    excluir : list, optional
        Lista de columnas a excluir (por defecto: identificadores y targets)
    
    Returns
    -------
    list
        Lista de columnas seleccionadas
    """
    if excluir is None:
        excluir = ['numero_de_cliente', 'foto_mes', 'clase_ternaria', 'target']
    
    columnas = [c for c in df.columns if c not in excluir]
    
    return columnas

def feature_engineering_lag(df: pd.DataFrame, columnas: list[str], cant_lag: int = 1) -> pd.DataFrame:
    """
    Genera variables de lag para los atributos especificados utilizando SQL.
  
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con los datos
    columnas : list
        Lista de atributos para los cuales generar lags. Si es None, no se generan lags.
    cant_lag : int, default=1
        Cantidad de lags a generar para cada atributo
  
    Returns:
    --------
    pd.DataFrame
        DataFrame con las variables de lag agregadas
    """

    logger.info(f"Realizando feature engineering con {cant_lag} lags para {len(columnas) if columnas else 0} atributos")

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar lags")
        return df
  
    # Construir la consulta SQL
    sql = "SELECT *"
  
    # Agregar los lags para los atributos especificados
    for attr in columnas:
        if attr in df.columns:
            for i in range(1, cant_lag + 1):
                sql += f", lag({attr}, {i}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) AS {attr}_lag_{i}"
        else:
            logger.warning(f"El atributo {attr} no existe en el DataFrame")
  
    # Completar la consulta
    sql += " FROM df"

    logger.debug(f"Consulta SQL: {sql}")

    # Ejecutar la consulta SQL
    con = duckdb.connect(database=":memory:")
    con.register("df", df)
    df = con.execute(sql).df()
    con.close()
  
    logger.info(f"Feature engineering completado. DataFrame resultante con {df.shape[1]} columnas")

    return df

from typing import List, Optional

def feature_engineering_delta(
    df: pd.DataFrame,
    columnas: Optional[List[str]],
    cant_delta: int = 2
) -> pd.DataFrame:
    """
    Genera variables de delta (cambio absoluto) para los atributos especificados,
    usando SQL sobre ventanas por cliente ordenadas por foto_mes.

    Delta 1: attr - lag(attr, 1)
    Delta 2: attr - lag(attr, 2)

    Args:
        df: DataFrame base que contiene 'numero_de_cliente' y 'foto_mes'
        columnas: lista de atributos numéricos para generar deltas
        cant_delta: cuántos deltas generar (máx 2; si pasás >2, se trunca a 2)

    Returns:
        DataFrame con las columnas delta agregadas.
    """
    logger.info(
        f"Generando deltas (hasta {cant_delta}) para "
        f"{len(columnas) if columnas else 0} atributos"
    )

    if columnas is None or len(columnas) == 0:
        logger.warning("No se especificaron atributos para generar deltas")
        return df

    # Aseguramos máximo 2 deltas (delta_1 y delta_2)
    cant_delta = max(1, int(cant_delta))

    # Construcción dinámica del SELECT
    sql = ["SELECT *"]
    for attr in columnas:
        if attr not in df.columns:
            logger.warning(f"El atributo '{attr}' no existe en el DataFrame; se omite.")
            continue
        # Delta 1
        sql.append(
            f", ({attr} - lag({attr}, 1) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)) "
            f"AS {attr}_delta_1"
        )
        # Delta 2 (solo si corresponde)
        if cant_delta >= 2:
            sql.append(
                f", ({attr} - lag({attr}, 2) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)) "
                f"AS {attr}_delta_2"
            )

    # Si ninguna columna válida, devolvemos df sin cambios
    if len(sql) == 1:
        logger.warning("No se agregaron deltas porque no hubo atributos válidos.")
        return df

    query = " ".join(sql) + " FROM df"

    logger.debug(f"Consulta SQL (deltas): {query}")

    con = duckdb.connect(database=":memory:")
    try:
        con.register("df", df)
        df_out = con.execute(query).df()
    finally:
        con.close()

    logger.info(f"Feature engineering (deltas) completado. Columnas ahora: {df_out.shape[1]}")
    return df_out

## Opción mas eficiente con ayuda de GPT

def feature_engineering_lag_delta(
    df: pd.DataFrame,
    columnas: Optional[List[str]] = None,
    cant_lag: int = 1,
    cant_delta: int = 2,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """
    Genera en UNA SOLA QUERY:
      - lags:   col_lag_k
      - deltas: col_delta_k = col - lag_k(col)
    
    Usando ventanas por numero_de_cliente ordenadas por foto_mes.
    """
    # -----------------------------------------
    # 1) Definir columnas a usar
    # -----------------------------------------
    if columnas is None:
        columnas = obtener_columnas_validas(df)

    if not columnas:
        logger.warning("No se especificaron columnas para FE; se devuelve df original.")
        return df

    # Sólo numéricas para lags/deltas (evita bardear con strings)
    numeric_cols = [
        c for c in columnas
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_cols:
        logger.warning("No hay columnas numéricas válidas para generar lags/deltas.")
        return df

    cant_lag = max(0, int(cant_lag))
    cant_delta = min(2, max(0, int(cant_delta)))  # acotamos deltas a 0–2

    logger.info(
        f"FE: {len(numeric_cols)} columnas numéricas | "
        f"{cant_lag} lags | {cant_delta} deltas"
    )

    # -----------------------------------------
    # 2) Armar conexión (reutilizable)
    # -----------------------------------------
    own_con = False
    if con is None:
        con = duckdb.connect(database=":memory:")
        own_con = True

    # Registramos df UNA sola vez
    con.register("df", df)

    # -----------------------------------------
    # 3) Construir query eficiente
    #    Sin SELECT *, sólo lo necesario
    # -----------------------------------------
    # columnas base que queremos preservar
    base_cols = ['numero_de_cliente', 'foto_mes']
    base_cols += [c for c in df.columns if c not in base_cols and c not in numeric_cols]
    base_cols += numeric_cols
    # quitamos duplicados preservando orden
    seen = set()
    base_cols = [c for c in base_cols if not (c in seen or seen.add(c))]

    select_parts = [", ".join(base_cols)]

    # Lags
    for col in numeric_cols:
        for k in range(1, cant_lag + 1):
            select_parts.append(
                f"lag({col}, {k}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes) "
                f"AS {col}_lag_{k}"
            )

    # Deltas
    for col in numeric_cols:
        for k in range(1, cant_delta + 1):
            select_parts.append(
                f"({col} - lag({col}, {k}) OVER (PARTITION BY numero_de_cliente ORDER BY foto_mes)) "
                f"AS {col}_delta_{k}"
            )

    query = f"""
        SELECT
            {", ".join(select_parts)}
        FROM df
    """

    logger.debug(f"Consulta SQL FE:\n{query[:1000]}...")  # log parcial si es muy larga

    # -----------------------------------------
    # 4) Ejecutar
    # -----------------------------------------
    df_out = con.execute(query).df()

    if own_con:
        con.close()

    logger.info(f"FE completado. Filas: {df_out.shape[0]} | Columnas: {df_out.shape[1]}")

    return df_out
