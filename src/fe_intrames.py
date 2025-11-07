import duckdb
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def fe_intrames(df_fe: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica feature engineering usando DuckDB sobre las columnas de tarjetas,
    plazos fijos, inversiones, etc. Devuelve un nuevo DataFrame con las
    variables agregadas.
    """
    logger.info("Iniciando feature engineering de tarjetas con DuckDB")

    # Conexión en memoria
    con = duckdb.connect(database=":memory:")

    try:
        # Registramos el DataFrame como tabla
        con.register("competencia_01", df_fe)

        # Macros
        con.execute("""
            CREATE OR REPLACE MACRO suma_sin_null(a, b) AS ifnull(a, 0) + ifnull(b, 0);
        """)
        con.execute("""
            CREATE OR REPLACE MACRO division_segura(a, b) AS 
                CASE 
                    WHEN ifnull(b, 0) = 0 THEN NULL 
                    ELSE ifnull(a, 0) / ifnull(b, 1) 
                END;
        """)

        query = """
        WITH sumas AS (
            SELECT
                *
              , suma_sin_null(mtarjeta_visa_consumo, mtarjeta_master_consumo) AS tc_consumo_total
              , suma_sin_null(Master_mfinanciacion_limite, Visa_mfinanciacion_limite) AS tc_financiacionlimite_total
              , suma_sin_null(Master_msaldopesos, Visa_msaldopesos) AS tc_saldopesos_total
              , suma_sin_null(Master_msaldodolares, Visa_msaldodolares) AS tc_saldodolares_total
              , suma_sin_null(Master_mconsumospesos, Visa_mconsumospesos) AS tc_consumopesos_total
              , suma_sin_null(Master_mconsumosdolares, Visa_mconsumosdolares) AS tc_consumodolares_total
              , suma_sin_null(Master_mlimitecompra, Visa_mlimitecompra) AS tc_limitecompra_total
              , suma_sin_null(Master_madelantopesos, Visa_madelantopesos) AS tc_adelantopesos_total
              , suma_sin_null(Master_madelantodolares, Visa_madelantodolares) AS tc_adelantodolares_total
              , suma_sin_null(tc_adelantopesos_total, tc_adelantodolares_total) AS tc_adelanto_total
              , suma_sin_null(Master_mpagado, Visa_mpagado) AS tc_pagado_total
              , suma_sin_null(Master_mpagospesos, Visa_mpagospesos) AS tc_pagadopesos_total
              , suma_sin_null(Master_mpagosdolares, Visa_mpagosdolares) AS tc_pagadodolares_total
              , suma_sin_null(Master_msaldototal, Visa_msaldototal) AS tc_saldototal_total
              , suma_sin_null(Master_mconsumototal, Visa_mconsumototal) AS tc_consumototal_total
              , suma_sin_null(Master_cconsumos, Visa_cconsumos) AS tc_cconsumos_total
              , suma_sin_null(Master_delinquency, Visa_delinquency) AS tc_morosidad_total
              , suma_sin_null(mplazo_fijo_dolares, mplazo_fijo_pesos) AS m_plazofijo_total
              , suma_sin_null(minversion1_dolares, minversion1_pesos) AS m_inversion1_total
              , suma_sin_null(mpayroll, mpayroll2) AS m_payroll_total
              , suma_sin_null(cpayroll_trx, cpayroll2_trx) AS c_payroll_total
              , suma_sin_null(
                    suma_sin_null(suma_sin_null(cseguro_vida, cseguro_auto), cseguro_vivienda),
                    cseguro_accidentes_personales
                ) AS c_seguros_total
            FROM competencia_01
        )
        SELECT
            sumas.*
          -- Ratios / proporciones
          , division_segura(m_plazofijo_total, cplazo_fijo) AS m_promedio_plazofijo_total
          , division_segura(m_inversion1_total, cinversion1) AS m_promedio_inversion_total
          , division_segura(mcaja_ahorro, ccaja_ahorro) AS m_promedio_caja_ahorro
          , division_segura(mtarjeta_visa_consumo, ctarjeta_visa_transacciones) AS m_promedio_tarjeta_visa_consumo_por_transaccion
          , division_segura(mtarjeta_master_consumo, ctarjeta_master_transacciones) AS m_promedio_tarjeta_master_consumo_por_transaccion
          , division_segura(mprestamos_prendarios, cprestamos_prendarios) AS m_promedio_prestamos_prendarios
          , division_segura(mprestamos_hipotecarios, cprestamos_hipotecarios) AS m_promedio_prestamos_hipotecarios
          , division_segura(minversion2, cinversion2) AS m_promedio_inversion2
          , division_segura(mpagodeservicios, cpagodeservicios) AS m_promedio_pagodeservicios
          , division_segura(mpagomiscuentas, cpagomiscuentas) AS m_promedio_pagomiscuentas
          , division_segura(mcajeros_propios_descuentos, ccajeros_propios_descuentos) AS m_promedio_cajeros_propios_descuentos
          , division_segura(mtarjeta_visa_descuentos, ctarjeta_visa_descuentos) AS m_promedio_tarjeta_visa_descuentos
          , division_segura(mtarjeta_master_descuentos, ctarjeta_master_descuentos) AS m_promedio_tarjeta_master_descuentos
          , division_segura(mcomisiones_mantenimiento, ccomisiones_mantenimiento) AS m_promedio_comisiones_mantenimiento
          , division_segura(mcomisiones_otras, ccomisiones_otras) AS m_promedio_comisiones_otras
          , division_segura(mforex_buy, cforex_buy) AS m_promedio_forex_buy
          , division_segura(mforex_sell, cforex_sell) AS m_promedio_forex_sell
          , division_segura(mtransferencias_recibidas, ctransferencias_recibidas) AS m_promedio_transferencias_recibidas
          , division_segura(mtransferencias_emitidas, ctransferencias_emitidas) AS m_promedio_transferencias_emitidas
          , division_segura(mextraccion_autoservicio, cextraccion_autoservicio) AS m_promedio_extraccion_autoservicio
          , division_segura(mcheques_depositados, ccheques_depositados) AS m_promedio_cheques_depositados
          , division_segura(mcheques_emitidos, ccheques_emitidos) AS m_promedio_cheques_emitidos
          , division_segura(mcheques_depositados_rechazados, ccheques_depositados_rechazados) AS m_promedio_cheques_depositados_rechazados
          , division_segura(mcheques_emitidos_rechazados, ccheques_emitidos_rechazados) AS m_promedio_cheques_emitidos_rechazados
          , division_segura(matm, catm_trx) AS m_promedio_atm
          , division_segura(matm_other, catm_trx_other) AS m_promedio_atm_other
          , division_segura(Master_msaldototal, Master_mfinanciacion_limite) AS proporcion_financiacion_master_cubierto
          , division_segura(Master_msaldototal, Master_mlimitecompra) AS proporcion_limite_master_cubierto
          , division_segura(Visa_msaldototal, Visa_mfinanciacion_limite) AS proporcion_financiacion_visa_cubierto
          , division_segura(Visa_msaldototal, Visa_mlimitecompra) AS proporcion_limite_visa_cubierto
          , division_segura(tc_saldototal_total, tc_financiacionlimite_total) AS proporcion_financiacion_total_cubierto
          , division_segura(tc_saldototal_total, tc_limitecompra_total) AS proporcion_limite_total_cubierto
          , division_segura(tc_saldopesos_total, tc_saldototal_total) AS tc_proporcion_saldo_pesos
          , division_segura(tc_saldodolares_total, tc_saldototal_total) AS tc_proporcion_saldo_dolares
          , division_segura(tc_consumopesos_total, tc_consumototal_total) AS tc_proporcion_consumo_pesos
          , division_segura(tc_consumodolares_total, tc_consumototal_total) AS tc_proporcion_consumo_dolares
          , division_segura(tc_consumototal_total, tc_limitecompra_total) AS tc_proporcion_consumo_total_limite_total_cubierto
          , division_segura(tc_pagadopesos_total, tc_pagado_total) AS tc_proporcion_pago_pesos
          , division_segura(tc_pagadodolares_total, tc_pagado_total) AS tc_proporcion_pago_dolares
          , division_segura(tc_adelantopesos_total, tc_adelanto_total) AS tc_proporcion_adelanto_pesos
          , division_segura(tc_adelantodolares_total, tc_adelanto_total) AS tc_proporcion_adelanto_dolares

          -- Comparación de fechas Master vs Visa
          , greatest(Master_Fvencimiento, Visa_Fvencimiento) AS tc_fvencimiento_mayor
          , least(Master_Fvencimiento, Visa_Fvencimiento)    AS tc_fvencimiento_menor
          , greatest(Master_fechaalta, Visa_fechaalta)       AS tc_fechaalta_mayor
          , least(Master_fechaalta, Visa_fechaalta)          AS tc_fechalta_menor
          , greatest(Master_Finiciomora, Visa_Finiciomora)   AS tc_fechamora_mayor
          , least(Master_Finiciomora, Visa_Finiciomora)      AS tc_fechamora_menor
          , greatest(Master_fultimo_cierre, Visa_fultimo_cierre) AS tc_fechacierre_mayor
          , least(Master_fultimo_cierre, Visa_fultimo_cierre)    AS tc_fechacierre_menor
        FROM sumas
        """

        df_out = con.execute(query).df()
        logger.info("Feature engineering de tarjetas finalizado. Nuevas columnas: %s",
                    [c for c in df_out.columns if c not in df_fe.columns])

        return df_out

    finally:
        con.close()
