"""Servidor local de la consola. Uso (desde la carpeta del proyecto):  python consola/servidor_consola.py

Abre http://127.0.0.1:8765 . Levanta solo los servidores MCP (9090 y 9091) si no están corriendo y los apaga al salir.
Solo escucha en 127.0.0.1 y rechaza peticiones con un Host distinto (defensa contra DNS rebinding).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Route

from consola import nucleo
from evaluacion.harness import MODOS, texto

PUERTO = 8765
HTML = ROOT / "consola" / "consola.html"
CANDADO = asyncio.Lock()          # Ollama atiende una pregunta a la vez; las demás esperan turno
procesos: list = []


async def inicio(request):
    if not HTML.exists():
        return JSONResponse({"error": "Falta consola/consola.html. Ejecuta: python consola/construir_consola.py"}, status_code=404)
    return FileResponse(HTML, media_type="text/html; charset=utf-8")


async def api_estado(request):
    return JSONResponse(await nucleo.estado())


async def api_preguntar(request):
    d = await request.json()
    pregunta = str(d.get("pregunta", "")).strip()[:500]
    modelo, modo = d.get("modelo"), d.get("modo", "autonomo")
    variante, desc = d.get("tools", "base"), d.get("descripcion", "completa")
    # Lista cerrada de valores: la consola no permite elegir modelos ni modos arbitrarios
    if not pregunta or modelo not in nucleo.MODELOS or modo not in MODOS or variante not in ("base", "ampliado") or desc not in nucleo.PUERTOS:
        return JSONResponse({"error": "parámetros no válidos"}, status_code=400)

    cola: asyncio.Queue = asyncio.Queue()

    async def emit(ev):
        await cola.put(ev)

    async def trabajo():
        try:
            if CANDADO.locked():
                await emit({"t": "en_cola", "at": 0})
            async with CANDADO:
                await nucleo.correr(emit, pregunta, modelo, modo, variante, desc)
        finally:
            await cola.put(None)

    tarea = asyncio.create_task(trabajo())

    async def flujo():
        try:
            while (ev := await cola.get()) is not None:
                yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
        finally:
            tarea.cancel()

    return StreamingResponse(flujo(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def clasificar(salida: str) -> dict:
    """Qué capa de defensa respondió. Las capas son las del servidor: validación de sentencia, authorizer, mode=ro."""
    if salida.startswith("Solo se permite UNA sentencia SELECT"):
        return {"permitido": False, "capa": "Validación de la sentencia (solo un SELECT)"}
    if "not authorized" in salida.lower() or "prohibited" in salida.lower():
        return {"permitido": False, "capa": "Authorizer de SQLite (tablas y operaciones permitidas)"}
    if "readonly" in salida.lower():
        return {"permitido": False, "capa": "Base abierta en solo lectura (mode=ro)"}
    if salida.startswith("Error al ejecutar"):
        return {"permitido": False, "capa": "Error de ejecución"}
    return {"permitido": True, "capa": "Consulta ejecutada"}


async def api_guardrail(request):
    d = await request.json()
    sql = str(d.get("sql", ""))[:1000]
    try:
        herramientas = await nucleo.tools_mcp("completa")
        t = next(x for x in herramientas if x.name == "consulta_sql")
        t0 = time.perf_counter()
        salida = texto(await t.ainvoke({"sql": sql}))
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"[:300]}, status_code=500)
    return JSONResponse({"sql": sql, "salida": salida[:3000], "ms": round((time.perf_counter() - t0) * 1000, 1), **clasificar(salida)})


async def api_resultados(request):
    from consola.construir_consola import resumen_resultados
    return JSONResponse(resumen_resultados())


async def api_trazas(request):
    f = ROOT / "consola" / "trazas.json"
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")) if f.exists() else None)


@asynccontextmanager
async def ciclo_de_vida(app):
    # Arranca los servidores MCP al iniciar (en un hilo: espera hasta 10 s a que abran el puerto) y los apaga al salir
    procesos.extend(await asyncio.to_thread(nucleo.asegurar_servidores))
    yield
    for p in procesos:
        p.terminate()


app = Starlette(
    routes=[Route("/", inicio), Route("/api/estado", api_estado), Route("/api/preguntar", api_preguntar, methods=["POST"]),
            Route("/api/guardrail", api_guardrail, methods=["POST"]), Route("/api/resultados", api_resultados), Route("/api/trazas", api_trazas)],
    middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])],
    lifespan=ciclo_de_vida,
)

if __name__ == "__main__":
    print(f"Consola en http://127.0.0.1:{PUERTO}  (Ctrl+C para salir)")
    uvicorn.run(app, host="127.0.0.1", port=PUERTO, log_level="warning")
