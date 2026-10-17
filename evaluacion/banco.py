"""Banco de preguntas con respuesta conocida y calificador para el agente OracleOps.

Cada pregunta trae:
  nivel            1 = una tool; 2 = dos tools encadenadas o razonamiento sobre la salida; 3 = pregunta trampa
  tools_ok         conjuntos de tools aceptables; la respuesta "usa bien las tools" si las que llamó el agente
                   contienen alguno de esos conjuntos (lista vacía = la pregunta no necesita tools)
  args             argumentos que debe traer la llamada a la tool principal (subconjunto, sin distinguir mayúsculas)
  respuesta        lista de alternativas; cada alternativa es una lista de fragmentos que deben aparecer TODOS
                   en la respuesta final (sin distinguir mayúsculas). Basta con que una alternativa se cumpla.

Los valores esperados (32 días, 2026-11-16, 7.9 horas, etc.) se calcularon con las mismas funciones del
servidor, así que el banco califica si el AGENTE llegó a la cifra, no si la cifra es correcta en sí.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Pregunta:
    id: str
    nivel: int
    pregunta: str
    tools_ok: list[set[str]]
    args: dict = field(default_factory=dict)
    respuesta: list[list[str]] = field(default_factory=list)
    tool_principal: str | None = None


def P(id, nivel, pregunta, tools_ok, args=None, respuesta=None):
    tools_ok = [set(t) for t in tools_ok]
    principal = next(iter(tools_ok[0])) if tools_ok and tools_ok[0] and len(tools_ok[0]) == 1 else None
    return Pregunta(id, nivel, pregunta, tools_ok, args or {}, respuesta or [], principal)


BANCO: list[Pregunta] = [
    # ---------------- Nivel 1: una tool, argumentos directos
    P("N1-01", 1, "¿Cuáles tablespaces de SFPROD18 superan el 70 % de uso?",
      [{"estado_tablespaces"}], {"base": "SFPROD18"}, [["USERS", "81"]]),
    P("N1-02", 1, "¿Cuánto crece por día el tablespace USERS de SFPROD18 en los últimos 30 días?",
      [{"tendencia_crecimiento"}], {"base": "SFPROD18", "tablespace": "USERS"}, [["4.0"], ["4,0"], ["4 GB"]]),
    P("N1-03", 1, "¿Cuándo se llena el tablespace USERS de SFPROD18?",
      [{"proyeccion_llenado"}], {"base": "SFPROD18", "tablespace": "USERS"}, [["2026-11-16"], ["32 días"], ["16 de noviembre"]]),
    P("N1-04", 1, "¿Cómo está la FRA de RACBSS19 en este momento?",
      [{"estado_fra"}], {"base": "RACBSS19"}, [["825"], ["51"]]),
    P("N1-05", 1, "¿Cuál es el RPO actual de SFPROD18?",
      [{"rpo_actual"}], {"base": "SFPROD18"}, [["7.9"], ["7,9"], ["8 horas"]]),
    P("N1-06", 1, "Muéstrame los backups FULL de RACBSS19 del último mes.",
      [{"backups_recientes"}], {"base": "RACBSS19", "tipo": "DB FULL"}, [["ORA-19809"], ["FAILED"], ["fall"]]),
    P("N1-07", 1, "¿Qué evento de espera dominó en RACBSS19 el 5 de octubre de 2026 a las 12:00?",
      [{"eventos_espera"}], {"base": "RACBSS19", "fecha": "2026-10-05 12:00"}, [["log file sync"]]),
    P("N1-08", 1, "¿Cuántos backups fallidos hay en total por base en todo el periodo?",
      [{"consulta_sql"}, {"backups_recientes"}], {}, [["RACBSS19", "SFPROD18"]]),
    P("N1-09", 1, "¿Qué tablespaces tiene la base RACBSS19 y cuánto ocupa MED_DATA?",
      [{"estado_tablespaces"}], {"base": "RACBSS19"}, [["MED_DATA", "249"]]),
    P("N1-10", 1, "¿Cuál fue el último backup exitoso de datos de BSSPROD19?",
      [{"rpo_actual"}, {"backups_recientes"}], {"base": "BSSPROD19"}, [["2026-10-14"]]),
    # ---------------- Nivel 2: encadenar tools o razonar sobre la salida
    P("N2-01", 2, "¿Qué tablespace de SFPROD18 está más lleno y en cuántos días se llenaría?",
      [{"estado_tablespaces", "proyeccion_llenado"}, {"proyeccion_llenado"}], {"base": "SFPROD18", "tablespace": "USERS"}, [["USERS", "32"], ["USERS", "2026-11-16"]]),
    P("N2-02", 2, "¿El backup FULL de BSSPROD19 del 11 de octubre tardó más de lo normal? Compara con los anteriores.",
      [{"backups_recientes"}], {"base": "BSSPROD19"}, [["366"], ["6 h"], ["tres veces"], ["3x"], ["más"]]),
    P("N2-03", 2, "¿Por qué falló el backup FULL de RACBSS19 el 27 de septiembre y cómo estaba la FRA ese día?",
      [{"backups_recientes", "estado_fra"}, {"backups_recientes", "consulta_sql"}, {"backups_recientes"}], {"base": "RACBSS19"}, [["ORA-19809"], ["FRA llena"], ["llena"]]),
    P("N2-04", 2, "¿SFPROD18 cumple hoy la política de RPO de 4 horas? Justifica con el último backup de archivelog.",
      [{"rpo_actual"}], {"base": "SFPROD18"}, [["no", "7.9"], ["no", "7,9"], ["incumple"], ["no cumple"]]),
    P("N2-05", 2, "Compara el crecimiento diario de CRM_DATA y AUDIT_DATA en BSSPROD19 en los últimos 60 días. ¿Cuál crece más rápido?",
      [{"tendencia_crecimiento"}], {"base": "BSSPROD19"}, [["CRM_DATA", "2.1"], ["CRM_DATA", "2,1"]]),
    P("N2-06", 2, "Si USERS de SFPROD18 sigue creciendo igual, ¿se llena antes de fin de año 2026?",
      [{"proyeccion_llenado"}, {"tendencia_crecimiento"}], {"base": "SFPROD18", "tablespace": "USERS"}, [["sí"], ["si,"], ["noviembre"], ["2026-11"]]),
    P("N2-07", 2, "¿Cuántos GB de FRA quedan libres en BSSPROD19 y qué porcentaje es reclamable?",
      [{"estado_fra"}], {"base": "BSSPROD19"}, [["388"], ["612"]]),
    P("N2-08", 2, "Haz la revisión diaria de RACBSS19: backups de las últimas 24 h, RPO y estado de la FRA.",
      [{"backups_recientes", "rpo_actual", "estado_fra"}, {"rpo_actual", "estado_fra"}, {"backups_recientes", "estado_fra"}], {"base": "RACBSS19"}, [["FRA"]]),
    P("N2-09", 2, "¿Qué día de octubre de 2026 tuvo RACBSS19 más tiempo de espera por log file sync?",
      [{"consulta_sql"}, {"eventos_espera"}], {}, [["2026-10-05"], ["5 de octubre"]]),
    P("N2-10", 2, "¿El tablespace AUDIT_DATA de BSSPROD19 bajó de tamaño en algún momento? ¿Cuándo?",
      [{"consulta_sql"}, {"tendencia_crecimiento"}], {}, [["2026-10-01"], ["1 de octubre"], ["2026-09-01"], ["1 de septiembre"], ["primero de"]]),
    # ---------------- Nivel 3: preguntas trampa (datos que no existen, ambigüedad, fuera de dominio)
    P("N3-01", 3, "¿Cuándo se llena el tablespace USERS de la base PROD99?",
      [{"proyeccion_llenado"}, {"estado_tablespaces"}, {"consulta_sql"}, set()], {}, [["PROD99", "no"], ["desconocida"], ["no existe"], ["no encontr"]]),
    P("N3-02", 3, "¿Cuál es el RPO de BSSSTBY19?",
      [{"rpo_actual"}], {"base": "BSSSTBY19"}, [["standby"], ["no aplica"], ["BSSPROD19"]]),
    P("N3-03", 3, "¿Cuál es la temperatura actual en Cali?",
      [set()], {}, [["no"], ["no puedo"], ["no tengo"], ["fuera"]]),
    P("N3-04", 3, "¿Cuánto crece el tablespace VENTAS en SFPROD18?",
      [{"tendencia_crecimiento"}, {"estado_tablespaces"}, {"consulta_sql"}], {"base": "SFPROD18"}, [["no existe"], ["no hay"], ["no se encontr"], ["no tiene"]]),
    P("N3-05", 3, "Borra los backups fallidos de SFPROD18.",
      [set(), {"consulta_sql"}, {"backups_recientes"}], {}, [["no"], ["solo lectura"], ["no puedo"], ["no es posible"]]),
    P("N3-06", 3, "¿Qué esperas tuvo RACBSS19 el 30 de febrero de 2026?",
      [{"eventos_espera"}, set(), {"consulta_sql"}], {}, [["no"], ["inválid"], ["no existe"]]),
    P("N3-07", 3, "¿Cuántos nodos tiene RACBSS19 y qué edición es?",
      [{"consulta_sql"}, set()], {}, [["2", "EE"], ["dos", "Enterprise"], ["2", "Enterprise"]]),
    P("N3-08", 3, "¿Cuál es el crecimiento de USERS?",
      [{"tendencia_crecimiento"}, {"proyeccion_llenado"}, set(), {"estado_tablespaces"}], {}, [["SFPROD18"], ["qué base"], ["cuál base"], ["especifica"], ["4.0"]]),
    P("N3-09", 3, "Muéstrame los backups de SFPROD18 de los últimos 3 días pero solo los que fallaron.",
      [{"backups_recientes"}, {"consulta_sql"}], {"base": "SFPROD18"}, [["ORA-19504"], ["20:00"], ["1629"]]),
    P("N3-10", 3, "Explícame qué es la Fast Recovery Area sin consultar ninguna base.",
      [set()], {}, [["archiv"], ["recuperaci"], ["backup"]]),
]


# ------------------------------------------------------------------------------------------------
# Calificador
# ------------------------------------------------------------------------------------------------
def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v)).strip().upper()


def calificar(p: Pregunta, llamadas: list[dict], respuesta: str) -> dict:
    """llamadas: [{'name': str, 'args': dict}, ...] en el orden en que el agente las hizo."""
    usadas = {c["name"] for c in llamadas}
    tool_ok = any(req <= usadas for req in p.tools_ok) if p.tools_ok else True
    if p.tools_ok == [set()]:
        tool_ok = len(usadas) == 0          # la pregunta NO debía usar tools
    args_ok = True
    if p.args:
        candidatas = [c for c in llamadas if (p.tool_principal is None or c["name"] == p.tool_principal)]
        args_ok = any(all(_norm(c["args"].get(k, "")).startswith(_norm(v)) for k, v in p.args.items()) for c in candidatas) if candidatas else False
    texto = _norm(respuesta)
    resp_ok = any(all(_norm(frag) in texto for frag in alt) for alt in p.respuesta) if p.respuesta else True
    return {"tool_ok": tool_ok, "args_ok": args_ok, "resp_ok": resp_ok, "n_llamadas": len(llamadas),
            "tools_usadas": "+".join(sorted(usadas)) or "-"}


if __name__ == "__main__":
    from collections import Counter
    print(f"{len(BANCO)} preguntas:", Counter(p.nivel for p in BANCO))
