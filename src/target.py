import logging
import pandas as pd
import duckdb

logger = logging.getLogger(__name__)

def clase_ternaria(csv_path: str = "~/buckets/b1/datasets/competencia_01_crudo.csv",
                            con: duckdb.DuckDBPyConnection | None = None) -> duckdb.DuckDBPyConnection:
    """
    Crea/repone las tablas competencia_01_crudo y competencia_01 en DuckDB
    a partir del CSV indicado, replicando la lógica SQL provista.
    Devuelve la conexión (para reuso).
    """
    if con is None:
        con = duckdb.connect(database=":memory:")

    # Asegurar que el archivo exista (mensaje claro si no)
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"No se encontró el archivo CSV en: {csv_path}")

    # 1) Tabla cruda desde el CSV
    con.execute(f"""
        CREATE OR REPLACE TABLE competencia_01_crudo AS
        SELECT * FROM read_csv_auto('{csv_path}')
    """)

    # 2) Tabla procesada con clase_ternaria (CTEs b, w, mx)
    con.execute("""
        CREATE OR REPLACE TABLE competencia_01 AS
        WITH b AS (
          SELECT
            numero_de_cliente,
            CAST(foto_mes AS INT) AS foto_mes,
            (CAST(foto_mes/100 AS INT) * 12) + (CAST(foto_mes AS INT) % 100) AS periodo0
          FROM competencia_01_crudo
        ),
        w AS (
          SELECT
            *,
            LEAD(periodo0, 1) OVER (PARTITION BY numero_de_cliente ORDER BY periodo0) AS periodo1,
            LEAD(periodo0, 2) OVER (PARTITION BY numero_de_cliente ORDER BY periodo0) AS periodo2
          FROM b
        ),
        mx AS (
          SELECT
            MAX(periodo0) AS periodo_ultimo,
            MAX(periodo0) - 1 AS periodo_anteultimo
          FROM b
        )
        SELECT
          d.*,
          CASE
            WHEN w.periodo0 < (SELECT periodo_anteultimo FROM mx)
                 AND w.periodo1 = w.periodo0 + 1
                 AND (w.periodo2 IS NULL OR w.periodo2 > w.periodo0 + 2)
              THEN 'BAJA+2'
            WHEN w.periodo0 < (SELECT periodo_ultimo FROM mx)
                 AND (w.periodo1 IS NULL OR w.periodo1 > w.periodo0 + 1)
              THEN 'BAJA+1'
            WHEN w.periodo0 < (SELECT periodo_anteultimo FROM mx)
              THEN 'CONTINUA'
            ELSE NULL
          END AS clase_ternaria
        FROM competencia_01_crudo d
        LEFT JOIN w
          ON w.numero_de_cliente = d.numero_de_cliente
         AND w.foto_mes = CAST(d.foto_mes AS INT)
    """)
    return con


def pivot_clase_ternaria(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Ejecuta el PIVOT de counts por foto_mes y clase_ternaria, ordenado por foto_mes.
    Devuelve un DataFrame de pandas para inspección/impresión.
    """
    query = """
        SELECT *
        FROM (
          PIVOT competencia_01
          ON clase_ternaria
          USING count(numero_de_cliente)
          GROUP BY foto_mes
        ) t
        ORDER BY foto_mes
    """
    df = con.execute(query).df()
    print(df)  # para corroborar en consola
    return df