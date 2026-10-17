"""Núcleo de la consola: ejecuta una pregunta con el agente y emite la traza como eventos.

Reutiliza los tres modos del harness (autónomo, guiado, determinístico) sin modificarlos. La traza sale de los
*callbacks* de LangChain: cada llamada al modelo y cada llamada a una herramienta MCP genera un evento con su
instante (ms desde que empezó la pregunta). La interfaz usa esos eventos para animar el flujo y dibujar la cascada
de tiempos, y el script de captura los guarda para poder repetirlos sin Ollama.
"""
from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_ollama import ChatOllama

from evaluacion.banco import BANCO, calificar
from evaluacion.harness import MODOS, cargar_tools_mcp, conjunto_tools, texto

ROOT = Path(__file__).resolve().parent.parent
SERVIDOR_MCP = ROOT / "mcp_servers" / "oracle_ops_mcp_server.py"
MODELOS = ["llama3.2:3b", "qwen3.5:9b", "qwen3:14b"]
PUERTOS = {"completa": 9090, "minima": 9091}
BANCO_POR_TEXTO = {p.pregunta: p for p in BANCO}       # permite calificar si la pregunta es del banco

_tools_cache: dict[str, list] = {}


def crear_llm(nombre: str, callbacks: list):
    # Qwen3 razona por defecto ("thinking"); se desactiva para que las latencias sean comparables con los experimentos
    extra = {"reasoning": False} if nombre.startswith("qwen3") else {}
    return ChatOllama(model=nombre, temperature=0, callbacks=callbacks, **extra)


# ------------------------------------------------------------------------------------------------
# Servidores MCP: se levantan solos si no están corriendo (el mismo patrón del notebook)
# ------------------------------------------------------------------------------------------------
def puerto_abierto(puerto: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", puerto)) == 0


def asegurar_servidores() -> list[subprocess.Popen]:
    procesos = []
    for desc, puerto in PUERTOS.items():
        if not puerto_abierto(puerto):
            env = {**os.environ, "OPS_PORT": str(puerto), "OPS_DESC": desc}
            procesos.append(subprocess.Popen([sys.executable, str(SERVIDOR_MCP)], env=env,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for _ in range(40):                                  # hasta 10 s para que abran el puerto
        if all(puerto_abierto(p) for p in PUERTOS.values()):
            break
        time.sleep(0.25)
    return procesos


async def tools_mcp(descripcion: str) -> list:
    if descripcion not in _tools_cache:
        _tools_cache[descripcion] = await cargar_tools_mcp(descripcion)
    return _tools_cache[descripcion]


async def estado() -> dict:
    """Qué está disponible ahora mismo: Ollama, modelos instalados y servidores MCP."""
    out = {"modelos": MODELOS, "ollama": False, "instalados": [], "mcp": {d: puerto_abierto(p) for d, p in PUERTOS.items()}}
    try:
        async with httpx.AsyncClient(timeout=1.5) as c:
            r = await c.get("http://127.0.0.1:11434/api/tags")
            out["instalados"] = sorted(m["name"] for m in r.json().get("models", []))
            out["ollama"] = True
    except Exception:
        pass
    return out


# ------------------------------------------------------------------------------------------------
# Traza
# ------------------------------------------------------------------------------------------------
class Traza(AsyncCallbackHandler):
    """Convierte los callbacks de LangChain en eventos {t, at, ...} (at = ms desde el inicio de la pregunta)."""

    def __init__(self, emit, t0: float):
        self.emit, self.t0, self.ini, self.n_llm = emit, t0, {}, 0

    def _ms(self) -> int:
        return round((time.perf_counter() - self.t0) * 1000)

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.n_llm += 1
        self.ini[run_id] = self._ms()
        await self.emit({"t": "llm_inicio", "at": self.ini[run_id], "n": self.n_llm})

    async def on_llm_end(self, response, *, run_id, **kwargs):
        msg = getattr(response.generations[0][0], "message", None)
        uso = (getattr(msg, "usage_metadata", None) or {}) if msg is not None else {}
        llamadas = [tc["name"] for tc in (getattr(msg, "tool_calls", None) or [])]
        await self.emit({"t": "llm_fin", "at": self._ms(), "desde": self.ini.get(run_id), "decide": llamadas,
                         "texto": texto(getattr(msg, "content", ""))[:700],
                         "tokens_in": uso.get("input_tokens"), "tokens_out": uso.get("output_tokens")})

    async def on_llm_error(self, error, *, run_id, **kwargs):
        await self.emit({"t": "error", "at": self._ms(), "mensaje": f"{type(error).__name__}: {error}"[:300]})

    async def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kwargs):
        nombre = kwargs.get("name") or (serialized or {}).get("name") or "herramienta"
        self.ini[run_id] = self._ms()
        await self.emit({"t": "tool_inicio", "at": self.ini[run_id], "name": nombre, "args": inputs if inputs is not None else {"input": input_str}})

    async def on_tool_end(self, output, *, run_id, **kwargs):
        await self.emit({"t": "tool_fin", "at": self._ms(), "desde": self.ini.get(run_id),
                         "salida": texto(getattr(output, "content", output))[:4000]})

    async def on_tool_error(self, error, *, run_id, **kwargs):
        await self.emit({"t": "tool_fin", "at": self._ms(), "desde": self.ini.get(run_id), "salida": f"Error: {error}"[:600], "error": True})


# ------------------------------------------------------------------------------------------------
# Una pregunta, de principio a fin
# ------------------------------------------------------------------------------------------------
async def correr(emit, pregunta: str, modelo: str, modo: str = "autonomo", variante: str = "base",
                 descripcion: str = "completa", fabrica=None, timeout: float = 240) -> dict:
    """Ejecuta la pregunta y emite eventos. Devuelve el resumen (también se emite como evento 'final')."""
    fabrica = fabrica or crear_llm
    t0 = time.perf_counter()
    h = Traza(emit, t0)
    await emit({"t": "inicio", "at": 0, "pregunta": pregunta, "modelo": modelo, "modo": modo, "tools": variante, "descripcion": descripcion})
    try:
        base = await tools_mcp(descripcion)
        # Cada llamada a herramienta pasa por una copia con el handler: así la traza no depende del modo de ejecución
        tools = [t.model_copy(update={"callbacks": [h]}) for t in conjunto_tools(base, variante)]
        llm = fabrica(modelo, [h])
        respuesta, llamadas = await asyncio.wait_for(MODOS[modo](llm, tools, pregunta), timeout)
    except Exception as exc:
        _tools_cache.pop(descripcion, None)              # por si el servidor MCP se reinició
        msg = f"{type(exc).__name__}: {exc}"[:300]
        await emit({"t": "error", "at": round((time.perf_counter() - t0) * 1000), "mensaje": msg})
        return {"error": msg}
    p = BANCO_POR_TEXTO.get(pregunta)
    calif = None
    if p is not None:
        c = calificar(p, llamadas, respuesta)
        calif = {k: bool(c[k]) for k in ("tool_ok", "args_ok", "resp_ok")}
    resumen = {"t": "final", "at": round((time.perf_counter() - t0) * 1000), "respuesta": respuesta, "llamadas": llamadas,
               "n_llm": h.n_llm, "calificacion": calif, "id_banco": p.id if p else None}
    await emit(resumen)
    return resumen
