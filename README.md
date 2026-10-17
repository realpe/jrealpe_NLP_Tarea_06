# Mini-proyecto 6 — Agente de operación Oracle con MCP, LangChain y Ollama

**Autores:** José Luis Realpe M., Alejandra Forero, Santiago Aristizabal, Sandra Orozco
**Curso:** Procesamiento de Lenguaje Natural — Maestría en IA Aplicada, Universidad Icesi
**Basado en:** notebooks guía de la Sesión 6 (`1-herramientas-y-agentes-con-ollama`, `2-mcp-langchain-ollama`, `3-chatbot-con-mcp-y-gradio`)

> Un servidor **MCP** propio expone la operación de bases Oracle (tablespaces, FRA, backups RMAN, RPO, esperas) como herramientas que un LLM
> local (Ollama) descubre y llama a través de LangChain. El proyecto mide con 450 corridas qué tan confiable es ese acoplamiento y trae una
> **consola web** que muestra, paso a paso, cómo viaja cada pregunta por el protocolo.

## Por qué MCP

Un DBA responde a diario preguntas como *¿cuándo se llena USERS?*, *¿terminó bien el FULL del domingo?* o *¿cuál es el RPO ahora?*. La
información vive en vistas del diccionario y tableros, y exige cruzar fuentes y hacer cuentas. Escribir esas consultas como funciones
dentro de un solo agente las ata a ese agente. Con **Model Context Protocol** las consultas viven en un servidor independiente que publica
sus *tools*, un *resource* y un *prompt*, y cualquier cliente MCP puede usarlas: este notebook, la consola, Gradio o Claude Desktop.

Esa separación permite tres cosas que el proyecto demuestra con datos:

1. **Descubrimiento.** El cliente lista las tools con su esquema y descripción; el modelo decide cuál llamar sin código de enrutamiento.
2. **Mínimo privilegio en un solo punto.** La base se abre en solo lectura (`mode=ro`) y el SQL libre pasa por un *authorizer* de SQLite.
   Cualquier cliente hereda esas defensas (OWASP LLM01 inyección de prompt y LLM06 agencia excesiva).
3. **Medición reproducible.** El mismo servidor se prueba con descripciones completas o mínimas, con herramientas distractoras y con dos
   transportes (stdio y streamable HTTP).

## Consola MCP (demo principal)

La consola es una página web que se conecta al mismo harness de los experimentos. Lo que se ve en pantalla es lo que se midió.

| Pestaña | Qué muestra |
|---|---|
| **Consola** | El recorrido de la pregunta por cinco estaciones (usuario, agente, cliente MCP, servidor MCP, SQLite) con un paquete animado, la cascada de tiempos (modelo frente a herramientas) y, para cada llamada MCP, argumentos y salida completos |
| **Comparar modelos** | La misma pregunta con `llama3.2:3b`, `qwen3.5:9b` y `qwen3:14b`, lado a lado, con tiempos, llamadas y tokens |
| **Guardrails** | Ataques contra `consulta_sql` (escritura, tablas fuera de lista, múltiples sentencias) y la capa que detuvo cada uno |
| **Resultados** | Las métricas de los experimentos 1 a 5 |

Frente a la interfaz de Gradio (un chat con un bloque de texto de traza), la consola muestra el protocolo en lugar de solo su resultado:
tiempos por llamada, paralelismo de tools, tokens por llamada al modelo, comparación entre modelos, defensas del servidor y resultados
agregados en una sola pantalla. Se puede usar en dos modos:

```bash
# En vivo, con Ollama activo. Levanta los servidores MCP (9090 y 9091) y sirve http://127.0.0.1:8765
python consola/servidor_consola.py

# Repetición, sin Ollama ni servidores: abrir consola/consola.html con doble clic.
# Reproduce 16 trazas reales capturadas en un MacBook Pro (consola/trazas.json).

# Volver a capturar las trazas y reconstruir consola/consola.html (3 a 6 minutos con Ollama)
python consola/capturar_trazas.py
```

El servidor local escucha solo en `127.0.0.1`, rechaza cualquier otro `Host` y acepta únicamente modelos y modos de una lista cerrada.
Ejecuta una pregunta a la vez porque Ollama atiende un modelo a la vez.

## El servidor MCP

`mcp_servers/oracle_ops_mcp_server.py` (FastMCP): 8 tools, 1 resource y 1 prompt. Solo lectura.

| Tool | Qué responde |
|---|---|
| `estado_tablespaces(base, umbral_pct)` | Uso actual por tablespace sobre el máximo con autoextend |
| `tendencia_crecimiento(base, tablespace, dias)` | GB/día con regresión robusta (Theil-Sen) |
| `proyeccion_llenado(base, tablespace)` | Días y fecha estimada de llenado, con nivel de riesgo |
| `estado_fra(base)` | Uso, reclamable y tendencia 24 h de la Fast Recovery Area |
| `backups_recientes(base, tipo, dias)` | Jobs RMAN con duración, estado y mensaje de error |
| `rpo_actual(base)` | Horas desde el último archivelog y último backup de datos exitosos |
| `eventos_espera(base, fecha, top)` | Top de esperas Statspack del snapshot más cercano o del día |
| `consulta_sql(sql)` | `SELECT` de solo lectura con *authorizer* para lo que ninguna tool cubre |

Resource `oracle://catalogo` (diccionario de datos) y prompt `revision_diaria(base)` (plantilla del chequeo matutino). Decisiones de
diseño: salidas compactas, errores que listan las opciones válidas, cálculo en el servidor (el modelo no hace aritmética sobre series) y
dos variantes de descripciones por variable de entorno (`OPS_DESC=completa|minima`).

## Datos

Repositorio operativo sintético (SQLite, 90 días, cuatro bases: 18c SE2, 19c EE con Data Guard, su standby y un RAC). Incluye siete
escenarios plantados de respuesta conocida: un tablespace que se acelera, una purga mensual, una FRA que se llena y hace fallar un
backup, un FULL tres veces más lento, un pico de `log file sync`, un archivelog fallido que duplica el RPO y un standby sin backups
propios. `data/generar_datos.py` lo regenera de forma determinística.

## Experimentos y resultados

450 corridas con `temperature=0` sobre un banco de 30 preguntas en tres niveles. Métricas por respuesta: **tool_ok** (herramienta
correcta), **args_ok** (argumentos correctos) y **resp_ok** (la respuesta contiene las cifras esperadas). Con n = 30 por celda, el
intervalo de Wilson al 95 % mide cerca de ±15 puntos: las diferencias menores a eso no son concluyentes.

| # | Pregunta | Diseño |
|---|---|---|
| 1 | ¿Cambia el tamaño del modelo la calidad del uso de tools? | `llama3.2:3b`, `qwen3.5:9b`, `qwen3:14b` × 30 preguntas |
| 2 | ¿Cuánto control hace falta? | Modos autónomo, guiado (sugerencia + fallback) y determinístico (enrutador + extracción JSON) |
| 3 | ¿Más tools confunden? | 8 tools MCP frente a 8 MCP + 8 distractoras locales |
| 4 | ¿Cuánto importa la descripción? | Descripciones completas frente a mínimas |
| 5 | ¿Qué cuesta el transporte? | Latencia por llamada: stdio frente a streamable HTTP (sin LLM) |

Resultados en modo autónomo con 8 tools y descripciones completas (porcentaje de las 30 preguntas):

| Modelo | tool_ok | args_ok | resp_ok |
|---|---|---|---|
| `llama3.2:3b` | 93 | 93 | 60 |
| `qwen3:14b` | 97 | 100 | 77 |
| `qwen3.5:9b` | 100 | 100 | 93 |

- Elegir la herramienta es casi trivial para los tres modelos; la diferencia está en convertir la salida en una respuesta correcta.
- Con descripciones mínimas, `args_ok` baja de 100 a 90 en los dos Qwen y `resp_ok` del 9B baja de 93 a 77.
- Agregar 8 herramientas distractoras cambia poco: `tool_ok` baja entre 0 y 3 puntos.
- El transporte no decide el rendimiento: stdio y HTTP responden en milisegundos, frente a los segundos que tarda el modelo.

El análisis completo, con gráficas y lectura de errores típicos, está en el notebook (secciones 10 a 13).

## Cómo ejecutarlo (local, con Ollama)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install starlette uvicorn      # solo para la consola en vivo

ollama pull llama3.2:3b && ollama pull qwen3.5:9b && ollama pull qwen3:14b
ollama serve                       # si no corre como servicio

jupyter lab mini_proyecto_mcp_oracle_ops.ipynb
```

El notebook genera los datos, levanta los servidores MCP en segundo plano (puertos 9090 y 9091), ejecuta los experimentos y guarda
`resultados/experimentos.csv` fila a fila: si la sesión se interrumpe, al volver a ejecutar continúa donde iba. Con `QUICK_MODE = True`
corre un modelo y 10 preguntas por configuración (unos 10 minutos); la corrida completa toma entre 60 y 90 minutos en un MacBook Pro con
Apple Silicon. Los puertos 9090 y 9091 los usan tanto el notebook como la consola, así que conviene ejecutarlos de uno en uno.

## Demo en Gradio

`app_gradio.py` monta el mismo agente en un chat con selector de modelo, modo y conjunto de herramientas. Cada respuesta termina con una
línea de **Fuentes** armada con las llamadas MCP que se ejecutaron de verdad (el modelo no la redacta) y, opcionalmente, la traza
completa. Se lanza desde la sección 12 del notebook o con `python app_gradio.py` (http://127.0.0.1:7860).

## Contenido del repositorio

| Ruta | Descripción |
|---|---|
| `mini_proyecto_mcp_oracle_ops.ipynb` | Notebook principal: planteamiento, EDA, servidor y cliente MCP, agente LangChain, banco de preguntas, experimentos 1–5, demo y conclusiones |
| `mcp_servers/oracle_ops_mcp_server.py` | Servidor MCP `OracleOpsMCP` |
| `consola/` | Consola web: `servidor_consola.py` (en vivo), `nucleo.py`, `capturar_trazas.py`, `trazas.json`, `consola.html` (repetición autónoma) |
| `app_gradio.py` | Demo en Gradio |
| `evaluacion/banco.py`, `evaluacion/harness.py` | Banco de 30 preguntas con calificador y harness de experimentos |
| `data/generar_datos.py`, `data/oracle_ops.db` | Generador y repositorio operativo sintético |
| `resultados/experimentos.csv` | Resultados fila a fila |
| `figuras/` | Gráficas del EDA y de los experimentos |
| `construir_notebook.py` | Script que arma el notebook |

## Seguridad

Marco de referencia: OWASP Top 10 para aplicaciones LLM (LLM01 inyección de prompt, LLM06 agencia excesiva).
La defensa está en el servidor y no en el prompt: base abierta en `mode=ro`, *authorizer* con lista de tablas y un solo `SELECT` por
llamada. En la captura de trazas, `llama3.2:3b` intentó un `DELETE FROM rman_jobs` por `consulta_sql` ante una instrucción maliciosa y
el servidor lo rechazó en la primera capa. Los datos son sintéticos: antes de conectar el agente a una base real hace falta un usuario
de solo lectura con privilegios mínimos y auditoría unificada sobre sus sesiones.

## Límites

Datos sintéticos con escenarios plantados; calificador por fragmentos de texto que puede aceptar respuestas correctas con otra
redacción o dejar pasar interpretaciones erróneas; una sola corrida por configuración con `temperature=0`; tres modelos locales.

## Referencias

- Model Context Protocol — Specification. https://modelcontextprotocol.io
- LangChain MCP Adapters y `create_agent`. https://docs.langchain.com
- Yao, S. et al. (2023). *ReAct: Synergizing Reasoning and Acting in Language Models*. ICLR.
- Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. NeurIPS.
- Sen, P. K. (1968). *Estimates of the regression coefficient based on Kendall's tau*. JASA.
- OWASP (2025). *Top 10 for LLM Applications*.
- Oracle (2024). *Database Backup and Recovery User's Guide*, 19c.
- Notebooks guía del curso: https://github.com/Ohtar10/icesi-nlp
