"""Construye consola/consola.html: toma la plantilla y le incrusta los resultados de los experimentos y las trazas capturadas.

El archivo resultante funciona de dos maneras:
  - abierto desde el servidor (http://127.0.0.1:8765): modo En vivo, con Ollama y los servidores MCP reales;
  - abierto con doble clic: modo Repetición, con las trazas guardadas (no necesita nada instalado).
Uso:  python consola/construir_consola.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

AQUI = Path(__file__).resolve().parent
ROOT = AQUI.parent
CSV = ROOT / "resultados" / "experimentos.csv"

# Experimento 5 (celda 7.2 del notebook, medido en el Mac): latencia por llamada en ms y arranque en s
TRANSPORTE = [{"transporte": "streamable HTTP", "mediana_ms": 2.2, "p95_ms": 2.5, "arranque_s": 0.0},
              {"transporte": "stdio", "mediana_ms": 1.4, "p95_ms": 1.7, "arranque_s": 0.68}]


def resumen_resultados() -> dict | None:
    if not CSV.exists():
        return None
    d = pd.read_csv(CSV)
    for c in ("tool_ok", "args_ok", "resp_ok"):
        d[c] = d[c].astype(float)
    g = d.groupby(["modelo", "tools", "descripcion", "modo"])
    filas = [{"modelo": k[0], "tools": k[1], "descripcion": k[2], "modo": k[3], "n": int(len(v)),
              "tool_ok": round(v.tool_ok.mean() * 100, 1), "args_ok": round(v.args_ok.mean() * 100, 1),
              "resp_ok": round(v.resp_ok.mean() * 100, 1), "lat_mediana_s": round(float(v.latencia_s.median()), 1)}
             for k, v in g]
    base = d[(d.tools == "base") & (d.descripcion == "completa")]
    niveles = [{"modelo": k[0], "modo": k[1], "nivel": int(k[2]), "resp_ok": round(v.resp_ok.mean() * 100, 1)}
               for k, v in base.groupby(["modelo", "modo", "nivel"])]
    return {"filas": filas, "niveles": niveles, "transporte": TRANSPORTE, "n_corridas": int(len(d))}


def main():
    plantilla = (AQUI / "plantilla.html").read_text(encoding="utf-8")
    trazas_f = AQUI / "trazas.json"
    trazas = json.loads(trazas_f.read_text(encoding="utf-8")) if trazas_f.exists() else None
    # El JSON va dentro de <script>: "</" se escribe "<\/" (JSON válido) para que una respuesta del modelo no cierre la etiqueta
    incrustar = lambda obj: json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")
    html = (plantilla.replace("/*__TRAZAS__*/null", incrustar(trazas))
                     .replace("/*__RESULTADOS__*/null", incrustar(resumen_resultados())))
    (AQUI / "consola.html").write_text(html, encoding="utf-8")
    print(f"consola.html escrito ({len(html) / 1024:.0f} KB) | trazas: {'sí' if trazas else 'no'} | resultados: {'sí' if CSV.exists() else 'no'}")


if __name__ == "__main__":
    main()
