"""Demo conversacional: asistente de operación Oracle con herramientas MCP, Ollama y Gradio.

Requisitos: `ollama serve` activo y el servidor MCP levantado (python mcp_servers/oracle_ops_mcp_server.py).
Uso:        python app_gradio.py            → http://127.0.0.1:7860

La interfaz muestra, además de la respuesta, la traza de llamadas MCP (herramienta, argumentos y salida),
que es lo que permite auditar si el modelo consultó datos o respondió de memoria.
"""
from __future__ import annotations

import asyncio
import json

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_ollama import ChatOllama

from evaluacion.harness import SYSTEM_PROMPT, cargar_tools_mcp, conjunto_tools, modo_autonomo, modo_deterministico, modo_guiado, sugerir_tool, texto
from langchain.agents import create_agent

MODELOS = ["llama3.2:3b", "qwen3.5:9b", "qwen3:14b"]
_tools_cache: dict[str, list] = {}


async def tools_para(variante: str) -> list:
    if "mcp" not in _tools_cache:
        _tools_cache["mcp"] = await cargar_tools_mcp("completa")
    return conjunto_tools(_tools_cache["mcp"], variante)


def linea_fuentes(llamadas: list[dict]) -> str:
    """Línea 'Fuentes' construida con las llamadas MCP que se ejecutaron de verdad.
    No la redacta el modelo, así que no puede inventar una referencia. Se eliminan duplicados conservando el orden."""
    vistas, fuentes = set(), []
    for c in llamadas:
        args = ", ".join(f"{k}={v}" for k, v in c["args"].items())
        ref = f"`{c['name']}({args})`"
        if ref not in vistas:
            vistas.add(ref)
            fuentes.append(ref)
    if not fuentes:
        return "**Fuentes:** ninguna (el modelo respondió sin consultar herramientas)"
    return "**Fuentes** (herramientas MCP sobre `oracle_ops.db`): " + " · ".join(fuentes)


async def responder(pregunta: str, historial: list, modelo: str, modo: str, variante: str, mostrar_traza: bool):
    historial = historial or []
    llm = ChatOllama(model=modelo, temperature=0, **({'reasoning': False} if modelo.startswith('qwen3') else {}))   # sin 'thinking' en Qwen3
    tools = await tools_para(variante)
    traza_txt = ""
    llamadas: list[dict] = []   # llamadas MCP ejecutadas, para la línea de fuentes
    try:
        if modo == "autonomo":
            # Con historial: el agente recibe la conversación completa para que las preguntas de seguimiento funcionen
            agente = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
            msgs = []
            for h in historial:
                msgs.append(HumanMessage(content=h["content"]) if h["role"] == "user" else AIMessage(content=h["content"]))
            msgs.append(HumanMessage(content=pregunta))
            res = await agente.ainvoke({"messages": msgs}, config={"recursion_limit": 12})
            respuesta = texto(res["messages"][-1].content)
            pasos = []
            for m in res["messages"]:
                if isinstance(m, AIMessage) and m.tool_calls:
                    llamadas += [{"name": tc["name"], "args": tc["args"]} for tc in m.tool_calls]
                    pasos += [f"→ {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})" for tc in m.tool_calls]
                elif isinstance(m, ToolMessage):
                    pasos.append("← " + texto(m.content)[:800].replace("\n", " / "))
            traza_txt = "\n".join(pasos) or "(sin llamadas a herramientas)"
        elif modo == "guiado":
            respuesta, llamadas = await modo_guiado(llm, tools, pregunta)
            traza_txt = f"sugerida: {sugerir_tool(pregunta)}\n" + "\n".join(f"→ {c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in llamadas)
        else:
            respuesta, llamadas = await modo_deterministico(llm, tools, pregunta)
            traza_txt = "\n".join(f"→ {c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in llamadas) or "(el enrutador no eligió herramienta)"
    except Exception as exc:
        respuesta = f"Error: {exc}. ¿Está Ollama activo (`ollama serve`) y el servidor MCP en el puerto 9090?"
    # La referencia a las fuentes se muestra siempre; la traza detallada es opcional
    respuesta += "\n\n" + linea_fuentes(llamadas)
    if mostrar_traza:
        respuesta += f"\n\n---\n**Traza MCP**\n```\n{traza_txt}\n```"
    historial = historial + [{"role": "user", "content": pregunta}, {"role": "assistant", "content": respuesta}]
    return "", historial


with gr.Blocks(title="Asistente de operación Oracle · MCP") as demo:
    gr.Markdown("## Asistente de operación Oracle (tablespaces, FRA, backups, esperas) con herramientas MCP\n"
                "Datos sintéticos: bases SFPROD18, BSSPROD19, BSSSTBY19 y RACBSS19. \"Hoy\" es 2026-10-14.")
    with gr.Row():
        modelo = gr.Dropdown(MODELOS, value="qwen3.5:9b", label="Modelo (Ollama)")
        modo = gr.Dropdown(["autonomo", "guiado", "deterministico"], value="autonomo", label="Modo de ejecución")
        variante = gr.Dropdown(["base", "ampliado"], value="base", label="Conjunto de herramientas")
        traza = gr.Checkbox(value=True, label="Mostrar traza MCP")
    chat = gr.Chatbot(label="Conversación", height=420)
    msg = gr.Textbox(label="Pregunta", placeholder="Ejemplo: ¿Cuándo se llena el tablespace USERS de SFPROD18?")
    gr.Examples(["¿Cuál es el RPO actual de SFPROD18?", "¿Por qué falló el backup FULL de RACBSS19 el 27 de septiembre?",
                 "Haz la revisión diaria de RACBSS19", "¿Qué evento de espera dominó en RACBSS19 el 5 de octubre a las 12:00?",
                 "¿Cuál es el RPO de BSSSTBY19?"], inputs=msg)
    limpiar = gr.Button("Limpiar")
    msg.submit(responder, [msg, chat, modelo, modo, variante, traza], [msg, chat])
    limpiar.click(lambda: ("", []), None, [msg, chat], queue=False)

if __name__ == "__main__":
    demo.launch(inline=False)
