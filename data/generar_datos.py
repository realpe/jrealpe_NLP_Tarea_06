"""Generador del repositorio operativo Oracle SINTÉTICO para el mini-proyecto 6 (MCP + LangChain).

Imita lo que un DBA recolecta a diario con vistas base (DBA_DATA_FILES, V$RECOVERY_FILE_DEST,
V$RMAN_BACKUP_JOB_DETAILS, STATSPACK) sin licencias de Diagnostics Pack. Todos los nombres de bases,
tablespaces y cifras son inventados. El generador es determinístico (semilla fija).

Cobertura: 90 días, del 2026-07-17 al 2026-10-14 ("hoy" para el agente es 2026-10-14 23:59).

Bases simuladas (perfiles parecidos a un entorno BSS de telecomunicaciones):
  SFPROD18   Oracle 18c SE2, instancia única            (facturación)
  BSSPROD19  Oracle 19c EE, primaria con Data Guard     (CRM + facturación)
  BSSSTBY19  Oracle 19c EE, standby Active Data Guard   (solo lectura; no toma backups)
  RACBSS19   Oracle 19c EE, RAC de dos nodos            (provisión y mediación)

Escenarios "plantados" (la verdad conocida con la que se califican las respuestas del agente):
  E1  SFPROD18.USERS         crecimiento acelerado desde 2026-09-15 (de ~1.2 a ~4.0 GB/día)
  E2  BSSPROD19.AUDIT_DATA   purga mensual el día 1: el uso cae ~45 % y vuelve a crecer
  E3  RACBSS19               tormenta de redo (carga masiva) llena la FRA el 2026-09-26; el FULL del domingo 27
                             falla con ORA-19809; reintento manual OK el lunes 28
  E4  BSSPROD19              FULL del 2026-10-11 tarda ~3x lo normal (contención de I/O)
  E5  RACBSS19               snapshot del 2026-10-05 12:00 con pico de 'log file sync'
  E6  SFPROD18               el backup de archivelog de las 20:00 del 2026-10-14 falló: RPO actual ≈ 8 h
  E7  BSSSTBY19              sin backups propios (los toma la primaria): rpo_actual debe decir "no aplica"

Uso:  python data/generar_datos.py [ruta_db]
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
INICIO = datetime(2026, 7, 17)
DIAS = 90
AHORA = datetime(2026, 10, 14, 23, 59, 0)      # referencia de "hoy" para el servidor MCP
GB = 1024 ** 3

BASES = pd.DataFrame([
    dict(base="SFPROD18", version="18.0.0.0", edicion="SE2", tipo="single-instance", nodos=1, rol="PRIMARY", fra_gb=600),
    dict(base="BSSPROD19", version="19.0.0.0", edicion="EE", tipo="primaria Data Guard", nodos=1, rol="PRIMARY", fra_gb=1000),
    dict(base="BSSSTBY19", version="19.0.0.0", edicion="EE", tipo="standby Active Data Guard", nodos=1, rol="PHYSICAL STANDBY", fra_gb=1000),
    dict(base="RACBSS19", version="19.0.0.0", edicion="EE", tipo="RAC", nodos=2, rol="PRIMARY", fra_gb=1600),
])

# tablespace: (base, nombre, GB usados al inicio, GB/día, GB asignados, GB máximo autoextend)
TABLESPACES = [
    ("SFPROD18", "SYSTEM", 2.1, 0.001, 4, 32), ("SFPROD18", "SYSAUX", 5.4, 0.02, 8, 32),
    ("SFPROD18", "USERS", 380.0, 1.2, 480, 700), ("SFPROD18", "FACT_DATA", 820.0, 0.9, 1000, 1500),
    ("SFPROD18", "FACT_IDX", 310.0, 0.35, 400, 800), ("SFPROD18", "UNDOTBS1", 18.0, 0.0, 32, 32),
    ("BSSPROD19", "SYSTEM", 2.3, 0.001, 4, 32), ("BSSPROD19", "SYSAUX", 9.8, 0.03, 16, 64),
    ("BSSPROD19", "CRM_DATA", 1450.0, 2.1, 1800, 4000), ("BSSPROD19", "CRM_IDX", 540.0, 0.8, 700, 2000),
    ("BSSPROD19", "AUDIT_DATA", 120.0, 3.5, 260, 400), ("BSSPROD19", "UNDOTBS1", 40.0, 0.0, 64, 64),
    ("BSSSTBY19", "SYSTEM", 2.3, 0.001, 4, 32), ("BSSSTBY19", "SYSAUX", 9.8, 0.03, 16, 64),
    ("BSSSTBY19", "CRM_DATA", 1450.0, 2.1, 1800, 4000), ("BSSSTBY19", "CRM_IDX", 540.0, 0.8, 700, 2000),
    ("BSSSTBY19", "AUDIT_DATA", 120.0, 3.5, 260, 400), ("BSSSTBY19", "UNDOTBS1", 40.0, 0.0, 64, 64),
    ("RACBSS19", "SYSTEM", 2.0, 0.001, 4, 32), ("RACBSS19", "SYSAUX", 12.1, 0.04, 20, 64),
    ("RACBSS19", "MED_DATA", 2100.0, 4.4, 2600, 6000), ("RACBSS19", "MED_IDX", 760.0, 1.3, 1000, 3000),
    ("RACBSS19", "PROV_DATA", 330.0, 0.6, 450, 1200), ("RACBSS19", "UNDOTBS1", 60.0, 0.0, 96, 96),
    ("RACBSS19", "UNDOTBS2", 60.0, 0.0, 96, 96),
]

# tamaño de la base (GB) para dimensionar backups y archivelog
TAMANO_GB = {"SFPROD18": 1600, "BSSPROD19": 2200, "BSSSTBY19": 2200, "RACBSS19": 3300}
ARCHIVELOG_GB_H = {"SFPROD18": 2.0, "BSSPROD19": 6.0, "BSSSTBY19": 6.0, "RACBSS19": 9.0}   # generación por hora (promedio)
WAITS = ["db file sequential read", "log file sync", "db file scattered read", "CPU time",
         "enq: TX - row lock contention", "gc buffer busy acquire", "direct path read", "log file parallel write"]


# ------------------------------------------------------------------ tablespaces (snapshot diario)
def gen_tablespaces(rng: np.random.Generator) -> pd.DataFrame:
    filas = []
    for base, ts, usado0, gb_dia, asignado0, maximo in TABLESPACES:
        usado = usado0
        asignado = asignado0
        for d in range(DIAS):
            fecha = INICIO + timedelta(days=d)
            tasa = gb_dia
            if base == "SFPROD18" and ts == "USERS" and fecha >= datetime(2026, 9, 15):
                tasa = 4.0                                             # E1: crecimiento acelerado
            if ts == "AUDIT_DATA" and fecha.day == 1 and d > 0:
                usado *= 0.55                                          # E2: purga mensual
            usado += tasa * (1 + rng.normal(0, 0.15)) if tasa > 0 else rng.normal(0, 0.05)
            usado = min(max(usado, 0.5), maximo * 0.999)
            # autoextend por bloques de 32 GB cuando el uso supera el 92 % de lo asignado
            while usado > asignado * 0.92 and asignado < maximo:
                asignado = min(asignado + 32, maximo)
            filas.append(dict(base=base, fecha=fecha.strftime("%Y-%m-%d"), tablespace=ts,
                              gb_usados=round(usado, 2), gb_asignados=float(asignado), gb_maximo=float(maximo)))
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ RMAN + FRA (horario)
def gen_rman_y_fra(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs, fra = [], []
    job_id = 1000
    for _, b in BASES.iterrows():
        base, fra_total = b["base"], float(b["fra_gb"])
        if b["rol"] != "PRIMARY":
            # E7: el standby no toma backups; su FRA solo guarda archivelogs aplicados
            usado = fra_total * 0.35
            for h in range(DIAS * 24):
                ts = INICIO + timedelta(hours=h)
                usado = fra_total * 0.35 + 25 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 2)
                fra.append(dict(base=base, ts=ts.strftime("%Y-%m-%d %H:%M:%S"), gb_total=fra_total,
                                gb_usados=round(usado, 1), gb_reclamables=round(usado * 0.6, 1)))
            continue

        usado = fra_total * 0.45
        reclamable = usado * 0.5
        tam = TAMANO_GB[base]
        for h in range(DIAS * 24):
            ts = INICIO + timedelta(hours=h)
            # archivelog: más generación en horario laboral y en el cierre de facturación (días 1-3)
            factor = 1.6 if 8 <= ts.hour <= 18 else 0.6
            if ts.day <= 3:
                factor *= 1.8
            tormenta = base == "RACBSS19" and datetime(2026, 9, 26, 0) <= ts <= datetime(2026, 9, 28, 1)   # E3: carga masiva
            usado += ARCHIVELOG_GB_H[base] * factor * (3.0 if tormenta else 1.0) * (1 + rng.normal(0, 0.2))

            # backups de archivelog cada 4 h; FULL domingos 02:00; INCR L1 el resto de días 02:00
            if ts.hour % 4 == 0:
                ok = True
                if base == "SFPROD18" and ts == datetime(2026, 10, 14, 20, 0):
                    ok = False                                         # E6: falla el último archivelog
                dur = 4 + rng.normal(0, 1)
                jobs.append(dict(base=base, job_id=job_id, tipo="ARCHIVELOG", inicio=ts, fin=ts + timedelta(minutes=max(dur, 1)),
                                 estado="COMPLETED" if ok else "FAILED", gb=round(ARCHIVELOG_GB_H[base] * 4 * factor, 1),
                                 mensaje=None if ok else "ORA-19504: fallo al crear archivo; ORA-27040: error de apertura (NFS no disponible)"))
                job_id += 1
                if ok and not tormenta:          # en la tormenta los backups a cinta van retrasados: nada queda reclamable
                    reclamable = min(usado, reclamable + ARCHIVELOG_GB_H[base] * 4 * factor)   # archivelogs ya respaldados

            if ts.hour == 2:
                es_full = ts.weekday() == 6
                tipo = "DB FULL" if es_full else "DB INCR L1"
                gb = tam * (1.0 if es_full else 0.07) * (1 + rng.normal(0, 0.05))
                base_min = (tam / 1024 * 55) if es_full else (tam / 1024 * 6)      # ~55 min/TB full, 6 min/TB incr
                dur = base_min * (1 + rng.normal(0, 0.08))
                ok, msg = True, None
                if base == "RACBSS19" and ts.date() == datetime(2026, 9, 27).date():
                    ok, msg = False, "ORA-19809: límite de archivos de recuperación excedido (FRA llena); ORA-19804: no se pueden recuperar 264 GB de espacio en 1600 GB"   # E3 (domingo)
                    dur = 12
                if base == "BSSPROD19" and ts.date() == datetime(2026, 10, 11).date():
                    dur = base_min * 3.1                                             # E4: 3x más lento
                    msg = "Advertencia: throughput del canal ORA_DISK_1 38 MB/s (normal ~120 MB/s)"
                jobs.append(dict(base=base, job_id=job_id, tipo=tipo, inicio=ts, fin=ts + timedelta(minutes=dur),
                                 estado="COMPLETED" if ok else "FAILED", gb=round(gb, 1) if ok else 0.0, mensaje=msg))
                job_id += 1
                if ok:
                    usado += gb * 0.08                                               # backupset comprimido a FRA (~8 % del tamaño)
                    reclamable += gb * 0.08                                  # el backup anterior queda obsoleto: reclamable
                if base == "RACBSS19" and ts.date() == datetime(2026, 9, 28).date():
                    # E3: reintento manual del FULL el lunes 28 a las 03:00, exitoso
                    jobs.append(dict(base=base, job_id=job_id, tipo="DB FULL", inicio=ts + timedelta(hours=1),
                                     fin=ts + timedelta(hours=1, minutes=tam / 1024 * 55 * 1.05), estado="COMPLETED",
                                     gb=round(tam * 1.0, 1), mensaje="Reintento manual tras liberar FRA (CHG-4402)"))
                    job_id += 1

            # E3: durante la tormenta de redo el espacio reclamable se agota y la FRA se llena
            if tormenta:
                reclamable = max(0.0, reclamable - 40)
            else:
                # Oracle libera espacio reclamable automáticamente cuando el uso pasa del 85 %
                if usado > fra_total * 0.75 and reclamable > 0:
                    liberar = min(reclamable, usado - fra_total * 0.50)
                    usado -= liberar
                    reclamable -= liberar
            if base == "RACBSS19" and ts == datetime(2026, 9, 28, 3):
                usado, reclamable = fra_total * 0.45, fra_total * 0.20     # E3: el DBA respalda a cinta y borra obsoletos (CHG-4402)
            usado = min(usado, fra_total)
            fra.append(dict(base=base, ts=ts.strftime("%Y-%m-%d %H:%M:%S"), gb_total=fra_total,
                            gb_usados=round(usado, 1), gb_reclamables=round(max(reclamable, 0), 1)))
    jobs_df = pd.DataFrame(jobs)
    jobs_df["inicio"] = jobs_df["inicio"].dt.strftime("%Y-%m-%d %H:%M:%S")
    jobs_df["fin"] = jobs_df["fin"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return jobs_df[jobs_df["inicio"] <= AHORA.strftime("%Y-%m-%d %H:%M:%S")], pd.DataFrame(fra)


# ------------------------------------------------------------------ Statspack: top waits por snapshot (cada 6 h)
def gen_waits(rng: np.random.Generator) -> pd.DataFrame:
    filas = []
    snap = 5000
    for _, b in BASES.iterrows():
        base = b["base"]
        perfil = {"db file sequential read": 35, "CPU time": 25, "db file scattered read": 12, "log file sync": 8,
                  "direct path read": 7, "enq: TX - row lock contention": 4, "log file parallel write": 4,
                  "gc buffer busy acquire": 5 if base == "RACBSS19" else 0}
        for h in range(0, DIAS * 24, 6):
            ts = INICIO + timedelta(hours=h)
            pesos = {k: v * (1 + rng.normal(0, 0.25)) for k, v in perfil.items() if v > 0}
            if base == "RACBSS19" and ts == datetime(2026, 10, 5, 12):
                pesos["log file sync"] *= 9                                          # E5
            if 8 <= ts.hour <= 18:
                pesos["CPU time"] *= 1.5
            total_s = (900 if 8 <= ts.hour <= 18 else 350) * (1 + rng.normal(0, 0.2))
            suma = sum(pesos.values())
            for ev, w in pesos.items():
                t = total_s * w / suma
                filas.append(dict(base=base, snap_id=snap, ts=ts.strftime("%Y-%m-%d %H:%M:%S"), evento=ev,
                                  waits=int(t * rng.uniform(80, 400)), tiempo_s=round(t, 1)))
            snap += 1
    return pd.DataFrame(filas)


ESCENARIOS = pd.DataFrame([
    dict(id="E1", base="SFPROD18", objeto="USERS", descripcion="Crecimiento acelerado desde 2026-09-15 (~4 GB/día)"),
    dict(id="E2", base="BSSPROD19", objeto="AUDIT_DATA", descripcion="Purga mensual el día 1 (cae ~45 %)"),
    dict(id="E3", base="RACBSS19", objeto="FRA/DB FULL", descripcion="Tormenta de redo llena la FRA el 26; FULL del 2026-09-27 falla ORA-19809; reintento OK el 28"),
    dict(id="E4", base="BSSPROD19", objeto="DB FULL", descripcion="FULL del 2026-10-11 tarda ~3x por I/O lento"),
    dict(id="E5", base="RACBSS19", objeto="log file sync", descripcion="Pico en el snapshot del 2026-10-05 12:00"),
    dict(id="E6", base="SFPROD18", objeto="ARCHIVELOG", descripcion="Backup de archivelog de 2026-10-14 20:00 falló: RPO ~8 h"),
    dict(id="E7", base="BSSSTBY19", objeto="backups", descripcion="Standby sin backups propios"),
])


def generar(ruta: str | Path, seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tablas = {"bases": BASES.drop(columns=["fra_gb"]), "tablespace_hist": gen_tablespaces(rng)}
    tablas["rman_jobs"], tablas["fra_hist"] = gen_rman_y_fra(rng)
    tablas["snapshot_waits"] = gen_waits(rng)
    tablas["escenarios"] = ESCENARIOS
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        ruta.unlink()
    with sqlite3.connect(ruta) as con:
        for nombre, df in tablas.items():
            df.to_sql(nombre, con, index=False)
        con.execute("CREATE INDEX ix_ts_base_fecha ON tablespace_hist(base, fecha)")
        con.execute("CREATE INDEX ix_fra_base_ts ON fra_hist(base, ts)")
        con.execute("CREATE INDEX ix_rman_base_inicio ON rman_jobs(base, inicio)")
        con.execute("CREATE INDEX ix_waits_base_ts ON snapshot_waits(base, ts)")
    return tablas


if __name__ == "__main__":
    destino = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("oracle_ops.db")
    t = generar(destino)
    for k, v in t.items():
        print(f"{k:16s} {len(v):>7d} filas")
    print(f"Base escrita en {destino}")
