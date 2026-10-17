"""Harness de experimentos: ejecuta el banco de preguntas contra el agente en distintas configuraciones.

Dimensiones que se cruzan (cada una responde una pregunta del mini-proyecto):
  modelo       llama3.2:3b | qwen2.5:7b | llama3.1:8b      ¿el tamaño del modelo cambia la calidad del uso de tools?
  modo         autonomo | guiado | deterministico           ¿cuánto control hace falta para que la tool se use?
  tools        base (8 MCP) | ampliado (8 MCP + 8 distractoras)   ¿más herramientas confunden al modelo?
  descripcion  completa | minima                            ¿cuánto importa el docstring que ve el LLM?

Cada corrida se guarda fila a fila en un CSV (resultados/experimentos.csv) para poder reanudar si Ollama se
cae o la sesión se interrumpe: las combinaciones ya ejecutadas se saltan.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient

from evaluacion.banco import BANCO, Pregunta, calificar

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "resultados" / "experimentos.csv"
URLS = {"completa": "http://127.0.0.1:9090/mcp", "minima": "http://127.0.0.1:9091/mcp"}

SYSTEM_PROMPT = (
    "Eres un asistente de operación de bases de datos Oracle. Respondes en español, de forma breve y citando las "
    "cifras exactas que devuelven las herramientas. Usa las herramientas para cualquier dato de tablespaces, FRA, "
    "backups, RPO o eventos de espera: nunca inventes cifras. Si la pregunta no se puede responder con las "
    "herramientas o pide una acción que no es de consulta, dilo claramente."
)

# Enrutador por palabras clave: lo usa el modo guiado (sugerencia) y el determinístico (decisión)
RUTAS = [
    (r"por base|todo el periodo|nodos|edici[oó]n|qu[eé] d[ií]a de|baj[oó] de tama[ñn]o", "consulta_sql"),
    (r"\brpo\b|perder[ií]amos|último backup (bueno|exitoso)", "rpo_actual"),
    (r"\bfra\b|recovery area|reclamable", "estado_fra"),
    (r"se llena|llenar|cu[aá]ntos d[ií]as le quedan|antes de fin de a[ñn]o", "proyeccion_llenado"),
    (r"crece|crecimiento|tendencia", "tendencia_crecimiento"),
    (r"espera|wait|contenci[oó]n|log file sync", "eventos_espera"),
    (r"backup|rman|full|incremental|archivelog", "backups_recientes"),
    (r"tablespace|ocupa|lleno|espacio", "estado_tablespaces"),
]


def sugerir_tool(pregunta: str) -> str | None:
    q = pregunta.lower()
    for patron, nombre in RUTAS:
        if re.search(patron, q):
            return nombre
    return None


# ------------------------------------------------------------------------------------------------
# Tools distractoras (locales, no MCP): mismas que un DBA podría tener en un asistente real
# ------------------------------------------------------------------------------------------------
@tool
def calculadora(expression: str) -> str:
    """Evalúa una expresión aritmética con +, -, *, /, % y paréntesis."""
    import ast, operator as op
    ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Mod: op.mod, ast.Pow: op.pow}
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops: return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand)
        raise ValueError("operador no soportado")
    return f"Resultado: {ev(ast.parse(expression, mode='eval'))}"


@tool
def clima_actual(city: str) -> str:
    """Consulta el clima actual de una ciudad."""
    return f"Clima de {city}: sin datos (servicio simulado)."


@tool
def sesiones_activas(base: str) -> str:
    """Número de sesiones activas e inactivas de una base de datos en este momento."""
    return f"{base}: 142 sesiones activas, 318 inactivas (dato simulado)."


@tool
def estado_listener(base: str) -> str:
    """Estado del listener de red de una base de datos (arriba/abajo) y servicios registrados."""
    return f"Listener de {base}: arriba, 3 servicios registrados (dato simulado)."


@tool
def listar_parches(base: str) -> str:
    """Parches de Oracle aplicados a una base de datos (opatch lsinventory)."""
    return f"{base}: Release Update 19.23 aplicado (dato simulado)."


@tool
def buscar_documentacion(tema: str) -> str:
    """Busca en la documentación de Oracle un tema o código de error."""
    return f"Documentación sobre '{tema}': consulte el manual de administración (resultado simulado)."


@tool
def convertir_unidades(valor: float, de: str, a: str) -> str:
    """Convierte entre unidades de almacenamiento (MB, GB, TB) o de tiempo (min, h, d)."""
    return f"{valor} {de} = (conversión simulada) {a}"


@tool
def fecha_actual() -> str:
    """Devuelve la fecha y hora actual del sistema."""
    return "2026-10-14 23:59"


DISTRACTORAS = [calculadora, clima_actual, sesiones_activas, estado_listener, listar_parches, buscar_documentacion,
                convertir_unidades, fecha_actual]


# ------------------------------------------------------------------------------------------------
# Carga de tools MCP y construcción del agente
# ------------------------------------------------------------------------------------------------
async def cargar_tools_mcp(descripcion: str = "completa") -> list:
    client = MultiServerMCPClient({"oracle_ops": {"transport": "http", "url": URLS[descripcion]}})
    return await client.get_tools()


def conjunto_tools(mcp_tools: list, variante: str) -> list:
    return list(mcp_tools) + (DISTRACTORAS if variante == "ampliado" else [])


def extraer_llamadas(messages) -> list[dict]:
    llamadas = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            llamadas.extend({"name": tc["name"], "args": tc.get("args", {})} for tc in m.tool_calls)
    return llamadas


def texto(contenido) -> str:
    if isinstance(contenido, str):
        return contenido
    if isinstance(contenido, list):
        return "\n".join(texto(c) for c in contenido)
    if isinstance(contenido, dict):
        return str(contenido.get("text") or contenido.get("content") or contenido)
    return str(contenido)


# ------------------------------------------------------------------------------------------------
# Modos de ejecución
# ------------------------------------------------------------------------------------------------
async def modo_autonomo(llm, tools, pregunta: str, hint: str | None = None) -> tuple[str, list[dict]]:
    """El agente decide solo. Con hint (modo guiado) se le sugiere la tool en el system prompt."""
    sp = SYSTEM_PROMPT + (f"\nSugerencia: para esta pregunta la herramienta más apropiada suele ser '{hint}'." if hint else "")
    agente = create_agent(model=llm, tools=tools, system_prompt=sp)
    res = await agente.ainvoke({"messages": [HumanMessage(content=pregunta)]}, config={"recursion_limit": 12})
    return texto(res["messages"][-1].content), extraer_llamadas(res["messages"])


async def modo_guiado(llm, tools, pregunta: str) -> tuple[str, list[dict]]:
    """Sugerencia en el prompt + fallback: si no usó ninguna tool, se le exige usar la sugerida."""
    hint = sugerir_tool(pregunta)
    respuesta, llamadas = await modo_autonomo(llm, tools, pregunta, hint)
    if hint and not llamadas:
        sp = SYSTEM_PROMPT + f"\nDEBES llamar a la herramienta '{hint}' antes de responder."
        agente = create_agent(model=llm, tools=tools, system_prompt=sp)
        res = await agente.ainvoke({"messages": [HumanMessage(content=pregunta)]}, config={"recursion_limit": 12})
        respuesta, llamadas = texto(res["messages"][-1].content), extraer_llamadas(res["messages"])
    return respuesta, llamadas


async def modo_deterministico(llm, tools, pregunta: str) -> tuple[str, list[dict]]:
    """El código elige la tool con el enrutador; el LLM solo extrae los argumentos (JSON) y redacta."""
    nombre = sugerir_tool(pregunta)
    if nombre is None:
        return await modo_autonomo(llm, [], pregunta)        # sin tool aplicable: responde el LLM solo
    tool_obj = next((t for t in tools if t.name == nombre), None)
    if tool_obj is None:
        return await modo_autonomo(llm, tools, pregunta)
    esquema = json.dumps(tool_obj.args, ensure_ascii=False)
    extractor = llm.bind(format="json") if hasattr(llm, "bind") else llm
    msg = await extractor.ainvoke([
        SystemMessage(content="Extrae los argumentos para llamar a una herramienta. Responde SOLO con un objeto JSON con las claves del esquema. "
                              "Bases válidas: SFPROD18, BSSPROD19, BSSSTBY19, RACBSS19. Hoy es 2026-10-14."),
        HumanMessage(content=f"Herramienta: {nombre}\nEsquema de argumentos: {esquema}\nPregunta: {pregunta}"),
    ])
    try:
        args = json.loads(texto(msg.content))
        args = {k: v for k, v in args.items() if k in tool_obj.args}
    except Exception:
        args = {}
    try:
        salida = texto(await tool_obj.ainvoke(args))
    except Exception as exc:
        salida = f"Error de la herramienta: {exc}"
    sintesis = await llm.ainvoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Pregunta: {pregunta}\n\nSalida de {nombre}({args}):\n{salida}\n\nRedacta la respuesta final en español, breve y con las cifras."),
    ])
    return texto(sintesis.content), [{"name": nombre, "args": args}]


MODOS = {"autonomo": modo_autonomo, "guiado": modo_guiado, "deterministico": modo_deterministico}


# ------------------------------------------------------------------------------------------------
# Ejecución de una pregunta y de la matriz completa
# ------------------------------------------------------------------------------------------------
async def ejecutar(llm, tools, p: Pregunta, modo: str) -> dict:
    t0 = time.perf_counter()
    try:
        respuesta, llamadas = await MODOS[modo](llm, tools, p.pregunta)
        error = ""
    except Exception as exc:        # un error del agente cuenta como fallo, no detiene la matriz
        respuesta, llamadas, error = "", [], f"{type(exc).__name__}: {exc}"[:200]
        if "not found" in str(exc).lower() and "model" in str(exc).lower():
            # Modelo ausente en Ollama: detener la matriz en vez de registrar cientos de fallos que luego se "reanudan" como hechos
            raise RuntimeError(f"Ollama no tiene el modelo ({exc}). Ejecuta `ollama pull <modelo>` y vuelve a correr.") from exc
    fila = {"id": p.id, "nivel": p.nivel, "pregunta": p.pregunta, "modo": modo, "latencia_s": round(time.perf_counter() - t0, 1),
            "respuesta": respuesta[:600], "llamadas": json.dumps(llamadas, ensure_ascii=False)[:600], "error": error}
    fila.update(calificar(p, llamadas, respuesta))
    return fila


async def correr_matriz(fabrica_llm, modelos: list[str], modos: list[str], variantes_tools: list[str],
                        descripciones: list[str], preguntas: list[Pregunta] | None = None, csv: Path = CSV, verbose: bool = True) -> pd.DataFrame:
    """fabrica_llm(nombre_modelo) -> chat model. Guarda fila a fila y reanuda lo ya ejecutado."""
    preguntas = preguntas or BANCO
    hechas = set()
    if csv.exists():
        prev = pd.read_csv(csv)
        hechas = set(zip(prev["modelo"], prev["modo"], prev["tools"], prev["descripcion"], prev["id"]))
    cache_tools = {}
    for desc in descripciones:
        mcp_tools = await cargar_tools_mcp(desc)
        for variante in variantes_tools:
            cache_tools[(desc, variante)] = conjunto_tools(mcp_tools, variante)
    for modelo in modelos:
        llm = fabrica_llm(modelo)
        for desc in descripciones:
            for variante in variantes_tools:
                tools = cache_tools[(desc, variante)]
                for modo in modos:
                    for p in preguntas:
                        clave = (modelo, modo, variante, desc, p.id)
                        if clave in hechas:
                            continue
                        fila = await ejecutar(llm, tools, p, modo)
                        fila.update({"modelo": modelo, "tools": variante, "descripcion": desc, "n_tools": len(tools)})
                        pd.DataFrame([fila]).to_csv(csv, mode="a", header=not csv.exists(), index=False)
                        hechas.add(clave)
                        if verbose:
                            ok = "".join("✓" if fila[k] else "✗" for k in ("tool_ok", "args_ok", "resp_ok"))
                            print(f"{modelo:12s} {desc:8s} {variante:8s} {modo:14s} {p.id} {ok} {fila['latencia_s']:5.1f}s {fila['tools_usadas']}")
    return pd.read_csv(csv)


def resumen(df: pd.DataFrame, por: list[str]) -> pd.DataFrame:
    g = df.groupby(por).agg(tool_ok=("tool_ok", "mean"), args_ok=("args_ok", "mean"), resp_ok=("resp_ok", "mean"),
                            llamadas=("n_llamadas", "mean"), latencia_s=("latencia_s", "mean"), n=("id", "count"))
    return g.round(3)


if __name__ == "__main__":
    # Prueba de humo sin LLM: solo carga tools y muestra sus nombres
    async def main():
        t = await cargar_tools_mcp("completa")
        print([x.name for x in t])
    asyncio.run(main())
