"""Servidor MCP "OracleOpsMCP": capacidad y backups de bases Oracle (datos sintéticos).

Mini-proyecto 6 — MCP con LLM y LangChain. Expone, con el protocolo MCP sobre transporte streamable
HTTP (igual que los notebooks guía), las consultas que un DBA hace a diario sobre tablespaces, Fast
Recovery Area (FRA), backups RMAN y eventos de espera. Los datos vienen de `data/oracle_ops.db`,
generado por `data/generar_datos.py`.

Primitivas MCP que expone (los notebooks guía solo usan tools):
  * 8 tools      estado_tablespaces, tendencia_crecimiento, proyeccion_llenado, estado_fra,
                 backups_recientes, rpo_actual, eventos_espera, consulta_sql
  * 1 resource   oracle://catalogo        diccionario de tablas, columnas y convenciones
  * 1 prompt     revision_diaria(base)    plantilla del chequeo matutino de un DBA

Variables de entorno (permiten levantar VARIANTES del mismo servidor para los experimentos):
  OPS_DB     ruta de la base SQLite        (por defecto ../data/oracle_ops.db)
  OPS_PORT   puerto HTTP                   (por defecto 9090)
  OPS_DESC   "completa" | "minima"         calidad de las descripciones que ve el LLM (experimento 4)
  OPS_TRANSPORT  "streamable-http" | "stdio"  transporte MCP (experimento de latencia)

Decisiones de diseño:
  * Solo lectura por construcción: SQLite se abre con `mode=ro` y `consulta_sql` pasa por un
    autorizador que solo permite SELECT sobre las tablas del catálogo. El LLM nunca recibe una
    conexión con permisos de escritura, el mismo principio que aplica a una base de producción.
  * Las tools devuelven texto compacto (tablas separadas por '|') en lugar de JSON extenso: menos
    tokens para un modelo local de 3B a 8B parámetros y más fácil de citar en la respuesta final.
  * Cada tool valida sus argumentos y devuelve un mensaje de error legible (nombre de base
    desconocido, fecha mal formada): el agente puede corregir y reintentar en lugar de abortar.
  * "Hoy" es una fecha fija (2026-10-14 23:59) para que los resultados sean reproducibles.
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import Field

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("OPS_DB", ROOT / "data" / "oracle_ops.db"))
PORT = int(os.environ.get("OPS_PORT", "9090"))
TRANSPORTE = os.environ.get("OPS_TRANSPORT", "streamable-http")      # "stdio" para el experimento de transporte
DESC_COMPLETA = os.environ.get("OPS_DESC", "completa") != "minima"
AHORA = datetime(2026, 10, 14, 23, 59, 0)
TABLAS_PERMITIDAS = {"bases", "tablespace_hist", "fra_hist", "rman_jobs", "snapshot_waits"}
MAX_FILAS = 40

mcp = FastMCP("OracleOpsMCP", host="127.0.0.1", port=PORT, log_level="WARNING")


def d(completa: str, minima: str = "") -> str:
    """Devuelve la descripción completa o la mínima según OPS_DESC (experimento sobre descripciones)."""
    return completa if DESC_COMPLETA else minima


BASE = Annotated[str, Field(description=d("Nombre de la base de datos: SFPROD18, BSSPROD19, BSSSTBY19 o RACBSS19.", "base"))]


# ------------------------------------------------------------------------------------------------
# Acceso a datos
# ------------------------------------------------------------------------------------------------
def _con() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _q(sql: str, params: tuple = ()) -> pd.DataFrame:
    with _con() as con:
        return pd.read_sql_query(sql, con, params=params)


@lru_cache(maxsize=1)
def _bases() -> pd.DataFrame:
    return _q("SELECT * FROM bases")


def _norm_base(base: str) -> str:
    """Acepta mayúsculas/minúsculas y espacios; falla con un mensaje que lista las bases válidas."""
    b = (base or "").strip().upper()
    validas = list(_bases()["base"])
    if b not in validas:
        raise ValueError(f"Base desconocida: '{base}'. Las bases disponibles son: {', '.join(validas)}.")
    return b


def _fecha(s: Optional[str], nombre: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s.strip()[:10], "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"'{nombre}' debe tener formato YYYY-MM-DD (recibido: '{s}').") from exc


def _tabla(df: pd.DataFrame, max_filas: int = MAX_FILAS) -> str:
    if df.empty:
        return "(sin filas)"
    cols = list(df.columns)
    filas = [" | ".join(cols)]
    for fila in df.head(max_filas).itertuples(index=False):
        filas.append(" | ".join("" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in fila))
    extra = f"\n({len(df) - max_filas} filas más omitidas)" if len(df) > max_filas else ""
    return "\n".join(filas) + extra


def _theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    """Pendiente de Theil-Sen (mediana de las pendientes entre pares): robusta a purgas y outliers."""
    n = len(x)
    if n < 2:
        return 0.0
    i, j = np.triu_indices(n, k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    return float(np.median((y[j] - y[i])[ok] / dx[ok])) if ok.any() else 0.0


# ------------------------------------------------------------------------------------------------
# Tools
# ------------------------------------------------------------------------------------------------
@mcp.tool(description=d(
    "Estado ACTUAL de los tablespaces de una base: GB usados, asignados, máximo con autoextend y porcentaje "
    "de uso sobre el máximo. Úsala para preguntas como '¿cómo están los tablespaces?', '¿cuál está más lleno?' "
    "o '¿cuánto espacio queda en X?'. umbral_pct filtra los que superan ese porcentaje (0 = todos).",
    "tablespaces"))
def estado_tablespaces(base: BASE, umbral_pct: Annotated[float, Field(description=d("Mostrar solo tablespaces con uso >= este porcentaje (0 = todos).", "umbral"))] = 0) -> str:
    b = _norm_base(base)
    df = _q("SELECT tablespace, gb_usados, gb_asignados, gb_maximo FROM tablespace_hist WHERE base=? AND fecha=(SELECT MAX(fecha) FROM tablespace_hist WHERE base=?) ORDER BY tablespace", (b, b))
    df["pct_uso_max"] = (100 * df["gb_usados"] / df["gb_maximo"]).round(1)
    df["gb_libres_max"] = (df["gb_maximo"] - df["gb_usados"]).round(1)
    df = df[df["pct_uso_max"] >= umbral_pct].sort_values("pct_uso_max", ascending=False)
    fecha = _q("SELECT MAX(fecha) f FROM tablespace_hist WHERE base=?", (b,))["f"][0]
    return f"Tablespaces de {b} al {fecha} (porcentaje sobre el máximo con autoextend):\n" + _tabla(df)


@mcp.tool(description=d(
    "Tasa de crecimiento de UN tablespace en GB por día, calculada sobre los últimos N días con regresión robusta "
    "(Theil-Sen). Úsala para '¿cuánto crece X por día?' o '¿cuál es la tendencia de X?'. Devuelve también el "
    "crecimiento total del periodo y los valores inicial y final.",
    "crecimiento"))
def tendencia_crecimiento(base: BASE, tablespace: Annotated[str, Field(description=d("Nombre del tablespace, por ejemplo USERS o CRM_DATA.", "tablespace"))],
                          dias: Annotated[int, Field(description=d("Ventana en días hacia atrás desde hoy (por defecto 30).", "dias"))] = 30) -> str:
    b = _norm_base(base)
    ts = tablespace.strip().upper()
    desde = (AHORA - timedelta(days=dias)).strftime("%Y-%m-%d")
    df = _q("SELECT fecha, gb_usados FROM tablespace_hist WHERE base=? AND tablespace=? AND fecha>=? ORDER BY fecha", (b, ts, desde))
    if df.empty:
        disponibles = ", ".join(_q("SELECT DISTINCT tablespace FROM tablespace_hist WHERE base=?", (b,))["tablespace"])
        return f"No existe el tablespace {ts} en {b}. Tablespaces de {b}: {disponibles}."
    x = np.arange(len(df), dtype=float)
    pend = _theil_sen(x, df["gb_usados"].to_numpy())
    return (f"{b}.{ts} en los últimos {dias} días ({df['fecha'].iloc[0]} a {df['fecha'].iloc[-1]}): "
            f"crecimiento de {pend:.2f} GB/día (Theil-Sen); uso inicial {df['gb_usados'].iloc[0]:.1f} GB, "
            f"uso final {df['gb_usados'].iloc[-1]:.1f} GB, variación total {df['gb_usados'].iloc[-1] - df['gb_usados'].iloc[0]:+.1f} GB.")


@mcp.tool(description=d(
    "Proyección de cuándo se llena un tablespace: días restantes y fecha estimada en que el uso alcanza el máximo "
    "con autoextend, usando la tasa de crecimiento de los últimos 14 días. Úsala para '¿cuándo se llena X?' o "
    "'¿cuántos días le quedan a X?'.",
    "proyeccion"))
def proyeccion_llenado(base: BASE, tablespace: Annotated[str, Field(description=d("Nombre del tablespace.", "tablespace"))]) -> str:
    b = _norm_base(base)
    ts = tablespace.strip().upper()
    desde = (AHORA - timedelta(days=14)).strftime("%Y-%m-%d")
    df = _q("SELECT fecha, gb_usados, gb_maximo FROM tablespace_hist WHERE base=? AND tablespace=? AND fecha>=? ORDER BY fecha", (b, ts, desde))
    if df.empty:
        return f"No existe el tablespace {ts} en {b}."
    pend = _theil_sen(np.arange(len(df), dtype=float), df["gb_usados"].to_numpy())
    usado, maximo = float(df["gb_usados"].iloc[-1]), float(df["gb_maximo"].iloc[-1])
    libres = maximo - usado
    if pend <= 0.01:
        return f"{b}.{ts}: {usado:.1f} GB usados de {maximo:.0f} GB máximo ({libres:.1f} GB libres). Crecimiento de {pend:.2f} GB/día: sin riesgo de llenado proyectable."
    dias = libres / pend
    fecha = (AHORA + timedelta(days=dias)).strftime("%Y-%m-%d")
    riesgo = "CRÍTICO (menos de 30 días)" if dias < 30 else "ATENCIÓN (menos de 90 días)" if dias < 90 else "bajo"
    return (f"{b}.{ts}: {usado:.1f} GB usados de {maximo:.0f} GB máximo ({libres:.1f} GB libres). Crecimiento reciente "
            f"{pend:.2f} GB/día. Se llenaría en {dias:.0f} días, aproximadamente el {fecha}. Nivel de riesgo: {riesgo}.")


@mcp.tool(description=d(
    "Estado de la Fast Recovery Area (FRA) de una base: GB totales, usados, reclamables, porcentaje de uso y "
    "tendencia de las últimas 24 horas. Úsala para '¿cómo está la FRA?', '¿la FRA está llena?' o "
    "'¿cuánto espacio reclamable hay?'.",
    "fra"))
def estado_fra(base: BASE) -> str:
    b = _norm_base(base)
    df = _q("SELECT ts, gb_total, gb_usados, gb_reclamables FROM fra_hist WHERE base=? ORDER BY ts DESC LIMIT 25", (b,))
    act, hace24 = df.iloc[0], df.iloc[-1]
    pct = 100 * act["gb_usados"] / act["gb_total"]
    rol = _bases().set_index("base").loc[b, "rol"]
    nota = " (standby: la FRA solo guarda archivelogs recibidos de la primaria)" if "STANDBY" in rol else ""
    return (f"FRA de {b} al {act['ts']}{nota}: {act['gb_usados']:.1f} GB usados de {act['gb_total']:.0f} GB ({pct:.1f} %), "
            f"{act['gb_reclamables']:.1f} GB reclamables, espacio neto comprometido {act['gb_usados'] - act['gb_reclamables']:.1f} GB. "
            f"Variación en 24 h: {act['gb_usados'] - hace24['gb_usados']:+.1f} GB. Máximo histórico: "
            f"{_q('SELECT MAX(gb_usados) m FROM fra_hist WHERE base=?', (b,))['m'][0]:.1f} GB.")


@mcp.tool(description=d(
    "Lista los trabajos de backup RMAN recientes de una base: tipo (DB FULL, DB INCR L1, ARCHIVELOG), inicio, "
    "duración en minutos, estado (COMPLETED/FAILED), GB y mensaje de error. Úsala para '¿terminó el backup?', "
    "'¿falló algún backup?', '¿cuánto tardó el full?'. tipo filtra por tipo de backup ('' = todos).",
    "backups"))
def backups_recientes(base: BASE, tipo: Annotated[str, Field(description=d("Filtro: 'DB FULL', 'DB INCR L1', 'ARCHIVELOG' o '' para todos.", "tipo"))] = "",
                      dias: Annotated[int, Field(description=d("Ventana en días hacia atrás (por defecto 7).", "dias"))] = 7) -> str:
    b = _norm_base(base)
    desde = (AHORA - timedelta(days=dias)).strftime("%Y-%m-%d %H:%M:%S")
    t = tipo.strip().upper()
    if t and t not in {"DB FULL", "DB INCR L1", "ARCHIVELOG"}:
        return "tipo debe ser 'DB FULL', 'DB INCR L1', 'ARCHIVELOG' o vacío."
    sql = "SELECT job_id, tipo, inicio, ROUND((julianday(fin)-julianday(inicio))*1440,1) AS minutos, estado, gb, mensaje FROM rman_jobs WHERE base=? AND inicio>=?"
    params: tuple = (b, desde)
    if t:
        sql += " AND tipo=?"
        params += (t,)
    df = _q(sql + " ORDER BY inicio DESC", params)
    if df.empty:
        rol = _bases().set_index("base").loc[b, "rol"]
        return f"{b} no tiene backups en los últimos {dias} días." + (" Es un standby: los backups los toma la primaria BSSPROD19." if "STANDBY" in rol else "")
    resumen = df.groupby("tipo").agg(n=("job_id", "count"), fallidos=("estado", lambda s: int((s == "FAILED").sum())), min_prom=("minutos", "mean")).round(1)
    resumen_txt = "; ".join(f"{t}: {r.n} jobs, {r.fallidos} fallidos, {r.min_prom} min promedio" for t, r in resumen.iterrows())
    # Para no saturar al modelo: los ARCHIVELOG exitosos se resumen y se listan solo los relevantes
    detalle = df[(df["tipo"] != "ARCHIVELOG") | (df["estado"] != "COMPLETED") | df["mensaje"].notna()]
    return f"Backups de {b} en los últimos {dias} días. Resumen: {resumen_txt}.\nDetalle (se omiten archivelogs exitosos):\n" + _tabla(detalle)


@mcp.tool(description=d(
    "RPO actual de una base: horas transcurridas desde el último backup EXITOSO de archivelog y desde el último "
    "backup exitoso de datos (FULL o INCR). Úsala para '¿cuál es el RPO?', '¿cuánto perderíamos si falla ahora?' "
    "o '¿cuándo fue el último backup bueno?'.",
    "rpo"))
def rpo_actual(base: BASE) -> str:
    b = _norm_base(base)
    rol = _bases().set_index("base").loc[b, "rol"]
    if "STANDBY" in rol:
        return f"{b} es un standby (Active Data Guard): no toma backups propios, los toma la primaria BSSPROD19. El RPO no aplica a esta base; consulta rpo_actual('BSSPROD19')."
    def ultimo(filtro: str) -> pd.Series:
        return _q(f"SELECT fin, tipo FROM rman_jobs WHERE base=? AND estado='COMPLETED' AND {filtro} ORDER BY fin DESC LIMIT 1", (b,)).iloc[0]
    arch, datos = ultimo("tipo='ARCHIVELOG'"), ultimo("tipo IN ('DB FULL','DB INCR L1')")
    h_arch = (AHORA - datetime.strptime(arch["fin"], "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600
    h_datos = (AHORA - datetime.strptime(datos["fin"], "%Y-%m-%d %H:%M:%S")).total_seconds() / 3600
    fallidos = _q("SELECT COUNT(*) n FROM rman_jobs WHERE base=? AND estado='FAILED' AND inicio>=?", (b, (AHORA - timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")))["n"][0]
    alerta = f" ALERTA: {fallidos} backup(s) fallido(s) en las últimas 24 h." if fallidos else ""
    return (f"RPO de {b} (referencia {AHORA:%Y-%m-%d %H:%M}): último archivelog exitoso terminó el {arch['fin']} "
            f"→ {h_arch:.1f} horas de exposición. Último backup de datos exitoso: {datos['tipo']} del {datos['fin']} "
            f"({h_datos:.1f} h).{alerta} Objetivo de la política: 4 horas.")


@mcp.tool(description=d(
    "Top de eventos de espera (Statspack) de una base en el snapshot más cercano a una fecha y hora dadas, o en el "
    "promedio del día si solo se da la fecha. Úsala para '¿qué esperas dominaban el día X?', '¿hubo contención?' o "
    "'¿qué pasó a las 12 del 5 de octubre?'.",
    "esperas"))
def eventos_espera(base: BASE, fecha: Annotated[str, Field(description=d("Fecha YYYY-MM-DD, opcionalmente con hora 'YYYY-MM-DD HH:MM'.", "fecha"))],
                   top: Annotated[int, Field(description=d("Cuántos eventos mostrar (por defecto 5).", "top"))] = 5) -> str:
    b = _norm_base(base)
    f = fecha.strip()
    if len(f) <= 10:
        _fecha(f, "fecha")
        df = _q("SELECT evento, SUM(waits) waits, ROUND(SUM(tiempo_s),1) tiempo_s FROM snapshot_waits WHERE base=? AND ts LIKE ? GROUP BY evento ORDER BY tiempo_s DESC", (b, f + "%"))
        if df.empty:
            return f"No hay snapshots de {b} el {f}. La auditoría cubre 2026-07-17 a 2026-10-14."
        df["pct"] = (100 * df["tiempo_s"] / df["tiempo_s"].sum()).round(1)
        return f"Eventos de espera de {b} el {f} (suma de los 4 snapshots del día):\n" + _tabla(df.head(top))
    try:
        objetivo = datetime.strptime(f[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return "fecha debe ser 'YYYY-MM-DD' o 'YYYY-MM-DD HH:MM'."
    snaps = _q("SELECT DISTINCT snap_id, ts FROM snapshot_waits WHERE base=?", (b,))
    snaps["dist"] = (pd.to_datetime(snaps["ts"]) - objetivo).abs()
    s = snaps.sort_values("dist").iloc[0]
    df = _q("SELECT evento, waits, tiempo_s FROM snapshot_waits WHERE base=? AND snap_id=? ORDER BY tiempo_s DESC", (b, int(s["snap_id"])))
    df["pct"] = (100 * df["tiempo_s"] / df["tiempo_s"].sum()).round(1)
    return f"Eventos de espera de {b}, snapshot {s['snap_id']} del {s['ts']} (el más cercano a {f}):\n" + _tabla(df.head(top))


# ---- consulta_sql: la vía de escape, con autorizador -------------------------------------------
def _autorizador(accion, arg1, arg2, dbname, source):
    """Política de SQLite: solo lectura sobre las tablas permitidas; todo lo demás se deniega."""
    if accion in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION):
        return sqlite3.SQLITE_OK
    if accion == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if arg1 in TABLAS_PERMITIDAS else sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_DENY


@mcp.tool(description=d(
    "Ejecuta una consulta SQL de SOLO LECTURA (SELECT) sobre el repositorio operativo cuando ninguna otra "
    "herramienta responde la pregunta. Tablas: bases(base, version, edicion, tipo, nodos, rol), "
    "tablespace_hist(base, fecha, tablespace, gb_usados, gb_asignados, gb_maximo), fra_hist(base, ts, gb_total, "
    "gb_usados, gb_reclamables), rman_jobs(base, job_id, tipo, inicio, fin, estado, gb, mensaje), "
    "snapshot_waits(base, snap_id, ts, evento, waits, tiempo_s). Dialecto SQLite; fechas como texto "
    "'YYYY-MM-DD HH:MM:SS'. Devuelve máximo 40 filas.",
    "sql"))
def consulta_sql(sql: Annotated[str, Field(description=d("Sentencia SELECT en dialecto SQLite.", "sql"))]) -> str:
    s = sql.strip().rstrip(";")
    if not re.match(r"^\s*(select|with)\b", s, re.IGNORECASE) or ";" in s:
        return "Solo se permite UNA sentencia SELECT (o WITH ... SELECT)."
    try:
        with _con() as con:
            con.set_authorizer(_autorizador)
            df = pd.read_sql_query(s, con)
    except Exception as exc:  # errores de sintaxis o de autorización
        return f"Error al ejecutar la consulta: {exc}"
    return f"{len(df)} fila(s):\n" + _tabla(df)


# ------------------------------------------------------------------------------------------------
# Resource y prompt (primitivas MCP que los notebooks guía no usan)
# ------------------------------------------------------------------------------------------------
@mcp.resource("oracle://catalogo", description="Diccionario del repositorio operativo: tablas, columnas, unidades y convenciones.")
def catalogo() -> str:
    bases = "\n".join(f"  - {r.base}: Oracle {r.version} {r.edicion}, {r.tipo}, {r.nodos} nodo(s), rol {r.rol}" for r in _bases().itertuples())
    return f"""REPOSITORIO OPERATIVO ORACLE (sintético). Periodo: 2026-07-17 a 2026-10-14. "Hoy" = {AHORA:%Y-%m-%d %H:%M}.
Bases:
{bases}
Tablas:
  bases(base, version, edicion, tipo, nodos, rol)
  tablespace_hist(base, fecha, tablespace, gb_usados, gb_asignados, gb_maximo)  -- un snapshot diario; gb_maximo = tope con autoextend
  fra_hist(base, ts, gb_total, gb_usados, gb_reclamables)                       -- un snapshot por hora
  rman_jobs(base, job_id, tipo, inicio, fin, estado, gb, mensaje)                -- tipo: DB FULL | DB INCR L1 | ARCHIVELOG; estado: COMPLETED | FAILED
  snapshot_waits(base, snap_id, ts, evento, waits, tiempo_s)                     -- Statspack, un snapshot cada 6 h
Convenciones: unidades en GB; FULL los domingos 02:00, INCR L1 el resto de días 02:00, ARCHIVELOG cada 4 h; política de RPO: 4 horas.
"""


@mcp.prompt(description="Plantilla del chequeo matutino de un DBA sobre una base.")
def revision_diaria(base: str) -> str:
    return (f"Eres el DBA de turno. Haz la revisión diaria de la base {base} usando las herramientas disponibles y "
            f"entrega un informe breve con cuatro secciones: (1) backups de las últimas 24 h y RPO actual, (2) estado de la "
            f"FRA, (3) tablespaces con más del 80 % de uso y su proyección de llenado, (4) eventos de espera dominantes de ayer. "
            f"Cita las cifras que devuelven las herramientas y termina con una lista de acciones recomendadas.")


if __name__ == "__main__":
    if TRANSPORTE == "stdio":
        mcp.run(transport="stdio")          # con stdio no se puede imprimir a stdout: el canal es el protocolo
    else:
        print(f"OracleOpsMCP en http://127.0.0.1:{PORT}/mcp | db={DB_PATH} | descripciones={'completas' if DESC_COMPLETA else 'mínimas'}")
        mcp.run(transport="streamable-http")
