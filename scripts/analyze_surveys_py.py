"""
Comprehensive analysis of db_semaforo survey data filtered to Paraguay only.
Uses Ollama (gpt-oss:20b) for narrative insights.
Output: data/analisis_encuestas_paraguay.md
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import requests

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "db_semaforo",
    "user": "sgadmin",
    "password": "sg4dm1n!",
}

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "gpt-oss:20b"
OUTPUT_PATH = "data/analisis_encuestas_paraguay.md"

# Paraguay filter: empresa.codpais = 1
# Main semaforo: USAID 2020 (idsemaforo=70), questions idpregunta 2759-2793
PY_JOIN = """
    JOIN dtsocioeconomica ds ON dp.idsocioeconomica = ds.iddtsocioeconomica
    JOIN usuario u ON ds.usuario = u.idusuario
    JOIN empresa e ON u.idempresa = e.idempresa
"""
PY_WHERE = "e.codpais = 1"
MAIN_SEMAFORO = 70
MAIN_QUESTIONS = "p.idpregunta BETWEEN 2759 AND 2793"


def query(sql: str) -> list[dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def llm_analyze(prompt: str) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["response"]


# ── Section generators ──────────────────────────────────────────────


def section_overview() -> str:
    print("  [1/10] Panorama general...")

    total_enc = query(f"""
        SELECT COUNT(DISTINCT ds.iddtsocioeconomica) as n
        FROM dtsocioeconomica ds
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE}
    """)[0]["n"]

    total_resp = query(f"""
        SELECT COUNT(*) as n FROM dtpregunta dp {PY_JOIN} WHERE {PY_WHERE}
    """)[0]["n"]

    total_personas = query(f"""
        SELECT COUNT(DISTINCT ds.idpersona) as n
        FROM dtsocioeconomica ds
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE} AND ds.idpersona IS NOT NULL
    """)[0]["n"]

    total_users = query(f"""
        SELECT COUNT(DISTINCT u.idusuario) as n
        FROM usuario u JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE}
    """)[0]["n"]

    total_orgs = query(f"""
        SELECT COUNT(*) as n FROM empresa WHERE codpais = 1
    """)[0]["n"]

    dates = query(f"""
        SELECT MIN(ds.fechaencuesta) as desde, MAX(ds.fechaencuesta) as hasta
        FROM dtsocioeconomica ds
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE} AND ds.fechaencuesta IS NOT NULL AND ds.fechaencuesta > '2010-01-01'
    """)[0]

    by_type = query(f"""
        SELECT te.descripcion as tipo, COUNT(DISTINCT ds.iddtsocioeconomica) as total
        FROM dtsocioeconomica ds
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        LEFT JOIN tipoencuesta te ON ds.idtipoencuesta = te.idtipoencuesta
        WHERE {PY_WHERE}
        GROUP BY te.descripcion ORDER BY total DESC
    """)

    by_platform = query(f"""
        SELECT ds.plataforma, COUNT(*) as total
        FROM dtsocioeconomica ds
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE}
        GROUP BY ds.plataforma ORDER BY total DESC
    """)

    s = "## 1. Panorama General - Paraguay\n\n"
    s += f"| Metrica | Valor |\n|---------|-------|\n"
    s += f"| Encuestas realizadas | {total_enc:,} |\n"
    s += f"| Respuestas totales (indicadores evaluados) | {total_resp:,} |\n"
    s += f"| Personas unicas encuestadas | {total_personas:,} |\n"
    s += f"| Usuarios/mentores activos | {total_users:,} |\n"
    s += f"| Organizaciones participantes | {total_orgs:,} |\n"
    s += f"| Periodo de datos | {str(dates['desde'])[:10]} a {str(dates['hasta'])[:10]} |\n\n"

    s += "### Por tipo de encuesta\n\n| Tipo | Encuestas |\n|------|----------|\n"
    for r in by_type:
        s += f"| {r['tipo']} | {r['total']:,} |\n"

    s += "\n### Por plataforma\n\n| Plataforma | Encuestas |\n|------------|----------|\n"
    for r in by_platform:
        s += f"| {r['plataforma']} | {r['total']:,} |\n"

    return s


def section_yearly() -> str:
    print("  [2/10] Evolucion anual...")

    by_year = query(f"""
        SELECT EXTRACT(YEAR FROM ds.fechaencuesta)::int as anio,
               COUNT(DISTINCT ds.iddtsocioeconomica) as encuestas
        FROM dtsocioeconomica ds
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE} AND ds.fechaencuesta IS NOT NULL
          AND EXTRACT(YEAR FROM ds.fechaencuesta) >= 2017
        GROUP BY anio ORDER BY anio
    """)

    color_year = query(f"""
        SELECT EXTRACT(YEAR FROM ds.fechaencuesta)::int as anio,
               dp.nroopcion,
               COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        WHERE {PY_WHERE} AND dp.nroopcion IN (1,2,3)
          AND ds.fechaencuesta IS NOT NULL
          AND EXTRACT(YEAR FROM ds.fechaencuesta) >= 2017
        GROUP BY anio, dp.nroopcion ORDER BY anio, dp.nroopcion
    """)

    years = {}
    for r in color_year:
        y = r["anio"]
        if y not in years:
            years[y] = {1: 0, 2: 0, 3: 0}
        years[y][r["nroopcion"]] = r["total"]

    s = "## 2. Evolucion Anual\n\n"
    s += "### Encuestas por anio\n\n| Anio | Encuestas |\n|------|----------|\n"
    for r in by_year:
        s += f"| {r['anio']} | {r['encuestas']:,} |\n"

    s += "\n### Distribucion semaforo por anio\n\n"
    s += "| Anio | Rojo | Amarillo | Verde | Total | % Rojo | % Amarillo | % Verde |\n"
    s += "|------|------|----------|-------|-------|--------|------------|----------|\n"
    for y in sorted(years):
        c = years[y]
        t = sum(c.values())
        if t == 0:
            continue
        s += f"| {y} | {c[1]:,} | {c[2]:,} | {c[3]:,} | {t:,} | {c[1]/t*100:.1f}% | {c[2]/t*100:.1f}% | {c[3]/t*100:.1f}% |\n"

    return s


def section_monthly() -> str:
    print("  [3/10] Evolucion mensual (ultimos 3 anios)...")

    monthly = query(f"""
        SELECT TO_CHAR(ds.fechaencuesta, 'YYYY-MM') as periodo,
               COUNT(DISTINCT ds.iddtsocioeconomica) as encuestas,
               COUNT(*) FILTER (WHERE dp.nroopcion=1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=2) as amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde
        FROM dtpregunta dp {PY_JOIN}
        WHERE {PY_WHERE} AND dp.nroopcion IN (1,2,3)
          AND ds.fechaencuesta >= '2023-01-01'
        GROUP BY periodo ORDER BY periodo
    """)

    s = "## 3. Evolucion Mensual (2023-2025)\n\n"
    s += "| Periodo | Encuestas | Rojo | Amarillo | Verde | % Verde |\n"
    s += "|---------|-----------|------|----------|-------|---------|\n"
    for r in monthly:
        t = r["rojo"] + r["amarillo"] + r["verde"]
        pct = (r["verde"] / t * 100) if t else 0
        s += f"| {r['periodo']} | {r['encuestas']:,} | {r['rojo']:,} | {r['amarillo']:,} | {r['verde']:,} | {pct:.1f}% |\n"

    return s


def section_global_distribution() -> str:
    print("  [4/10] Distribucion semaforo global...")

    overall = query(f"""
        SELECT dp.nroopcion, COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        WHERE {PY_WHERE} AND dp.nroopcion IN (1,2,3)
        GROUP BY dp.nroopcion ORDER BY dp.nroopcion
    """)

    color_map = {1: "Rojo (pobreza extrema)", 2: "Amarillo (pobreza)", 3: "Verde (no pobreza)"}
    total_all = sum(r["total"] for r in overall)

    s = "## 4. Distribucion Semaforo Global - Paraguay\n\n"
    s += "| Color | Total | Porcentaje |\n|-------|-------|------------|\n"
    for r in overall:
        pct = (r["total"] / total_all * 100) if total_all else 0
        s += f"| {color_map.get(r['nroopcion'])} | {r['total']:,} | {pct:.1f}% |\n"
    s += f"| **Total** | **{total_all:,}** | **100%** |\n"

    return s


def section_indicators_detail() -> str:
    print("  [5/10] Detalle por indicador (35 indicadores USAID)...")

    indicators = query(f"""
        SELECT p.abreviatura, p.nropregunta,
               COUNT(*) FILTER (WHERE dp.nroopcion=1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=2) as amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde,
               COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND {MAIN_QUESTIONS} AND dp.nroopcion IN (1,2,3)
        GROUP BY p.idpregunta, p.abreviatura, p.nropregunta
        ORDER BY p.nropregunta
    """)

    s = "## 5. Detalle por Indicador (Encuesta USAID 2020 - 35 indicadores)\n\n"
    s += "| # | Indicador | Rojo | Amarillo | Verde | Total | % Rojo | % Amarillo | % Verde |\n"
    s += "|---|-----------|------|----------|-------|-------|--------|------------|----------|\n"
    for r in indicators:
        t = r["total"]
        pr = r["rojo"] / t * 100 if t else 0
        pa = r["amarillo"] / t * 100 if t else 0
        pv = r["verde"] / t * 100 if t else 0
        s += f"| {r['nropregunta']} | {r['abreviatura']} | {r['rojo']:,} | {r['amarillo']:,} | {r['verde']:,} | {t:,} | {pr:.1f}% | {pa:.1f}% | {pv:.1f}% |\n"

    return s


def section_critical_indicators() -> str:
    print("  [6/10] Indicadores criticos y fortalezas...")

    indicators = query(f"""
        SELECT p.abreviatura,
               COUNT(*) FILTER (WHERE dp.nroopcion=1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=2) as amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde,
               COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND {MAIN_QUESTIONS} AND dp.nroopcion IN (1,2,3)
        GROUP BY p.idpregunta, p.abreviatura
    """)

    for ind in indicators:
        t = ind["total"]
        ind["pct_rojo"] = (ind["rojo"] / t * 100) if t else 0
        ind["pct_verde"] = (ind["verde"] / t * 100) if t else 0
        ind["pct_no_verde"] = 100 - ind["pct_verde"]

    worst = sorted(indicators, key=lambda x: -x["pct_rojo"])[:10]
    best = sorted(indicators, key=lambda x: -x["pct_verde"])[:10]

    s = "## 6. Top 10 Indicadores Criticos (mayor % en rojo)\n\n"
    s += "| Indicador | Rojo | Amarillo | Verde | % Rojo | % No-Verde |\n"
    s += "|-----------|------|----------|-------|--------|------------|\n"
    for r in worst:
        s += f"| {r['abreviatura']} | {r['rojo']:,} | {r['amarillo']:,} | {r['verde']:,} | {r['pct_rojo']:.1f}% | {r['pct_no_verde']:.1f}% |\n"

    s += "\n## 7. Top 10 Indicadores Fortaleza (mayor % en verde)\n\n"
    s += "| Indicador | Rojo | Amarillo | Verde | % Verde |\n"
    s += "|-----------|------|----------|-------|---------|\n"
    for r in best:
        s += f"| {r['abreviatura']} | {r['rojo']:,} | {r['amarillo']:,} | {r['verde']:,} | {r['pct_verde']:.1f}% |\n"

    return s


def section_baseline_vs_compliance() -> str:
    print("  [7/10] Linea base vs cumplimiento...")

    data = query(f"""
        SELECT te.descripcion as tipo, dp.nroopcion, COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        JOIN tipoencuesta te ON ds.idtipoencuesta = te.idtipoencuesta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND dp.nroopcion IN (1,2,3)
          AND te.descripcion IN ('LINEA DE BASE', 'CUMPLIMIENTO')
        GROUP BY te.descripcion, dp.nroopcion
        ORDER BY te.descripcion, dp.nroopcion
    """)

    tipos = {}
    for r in data:
        if r["tipo"] not in tipos:
            tipos[r["tipo"]] = {1: 0, 2: 0, 3: 0}
        tipos[r["tipo"]][r["nroopcion"]] = r["total"]

    s = "## 8. Comparacion: Linea de Base vs Cumplimiento\n\n"
    s += "| Tipo | Rojo | Amarillo | Verde | Total | % Rojo | % Amarillo | % Verde |\n"
    s += "|------|------|----------|-------|-------|--------|------------|----------|\n"
    for tipo in ["LINEA DE BASE", "CUMPLIMIENTO"]:
        c = tipos.get(tipo, {1: 0, 2: 0, 3: 0})
        t = sum(c.values())
        if t == 0:
            continue
        s += f"| {tipo} | {c[1]:,} | {c[2]:,} | {c[3]:,} | {t:,} | {c[1]/t*100:.1f}% | {c[2]/t*100:.1f}% | {c[3]/t*100:.1f}% |\n"

    # Per-indicator comparison
    indicator_data = query(f"""
        SELECT p.abreviatura, te.descripcion as tipo,
               COUNT(*) FILTER (WHERE dp.nroopcion=1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde,
               COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        JOIN tipoencuesta te ON ds.idtipoencuesta = te.idtipoencuesta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND {MAIN_QUESTIONS} AND dp.nroopcion IN (1,2,3)
          AND te.descripcion IN ('LINEA DE BASE', 'CUMPLIMIENTO')
        GROUP BY p.abreviatura, p.nropregunta, te.descripcion
        ORDER BY p.nropregunta
    """)

    ind_map = {}
    for r in indicator_data:
        name = r["abreviatura"]
        if name not in ind_map:
            ind_map[name] = {}
        t = r["total"]
        ind_map[name][r["tipo"]] = {
            "pct_rojo": (r["rojo"] / t * 100) if t else 0,
            "pct_verde": (r["verde"] / t * 100) if t else 0,
        }

    s += "\n### Cambio por indicador (% verde: linea base -> cumplimiento)\n\n"
    s += "| Indicador | % Verde Base | % Verde Cumpl. | Cambio (pp) |\n"
    s += "|-----------|-------------|----------------|-------------|\n"
    rows_sorted = []
    for name, d in ind_map.items():
        base = d.get("LINEA DE BASE", {}).get("pct_verde", 0)
        cumpl = d.get("CUMPLIMIENTO", {}).get("pct_verde", 0)
        diff = cumpl - base
        rows_sorted.append((name, base, cumpl, diff))
    rows_sorted.sort(key=lambda x: -x[3])
    for name, base, cumpl, diff in rows_sorted:
        sign = "+" if diff > 0 else ""
        s += f"| {name} | {base:.1f}% | {cumpl:.1f}% | {sign}{diff:.1f} |\n"

    return s


def section_mapavida() -> str:
    print("  [8/10] Mapa de vida...")

    totals = query(f"""
        SELECT COUNT(*) as total,
               COUNT(*) FILTER (WHERE mv.priorizado = true) as priorizados,
               COUNT(*) FILTER (WHERE mv.logro = true) as logrados
        FROM mapavida mv
        JOIN dtsocioeconomica ds ON mv.idsocioeconomica = ds.iddtsocioeconomica
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE}
    """)[0]

    top_prio = query(f"""
        SELECT p.abreviatura,
               COUNT(*) as priorizados,
               COUNT(*) FILTER (WHERE mv.logro = true) as logrados
        FROM mapavida mv
        JOIN pregunta p ON mv.idpregunta = p.idpregunta
        JOIN dtsocioeconomica ds ON mv.idsocioeconomica = ds.iddtsocioeconomica
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE} AND mv.priorizado = true
        GROUP BY p.abreviatura ORDER BY priorizados DESC LIMIT 15
    """)

    s = "## 9. Mapa de Vida - Paraguay\n\n"
    s += f"| Metrica | Valor |\n|---------|-------|\n"
    s += f"| Registros en mapa de vida | {totals['total']:,} |\n"
    s += f"| Indicadores priorizados | {totals['priorizados']:,} |\n"
    s += f"| Logros alcanzados | {totals['logrados']:,} |\n"
    pct = (totals["logrados"] / totals["priorizados"] * 100) if totals["priorizados"] else 0
    s += f"| Tasa de logro | {pct:.1f}% |\n\n"

    s += "### Indicadores mas priorizados\n\n| Indicador | Priorizados | Logrados | % Logro |\n"
    s += "|-----------|-------------|----------|--------|\n"
    for r in top_prio:
        pl = (r["logrados"] / r["priorizados"] * 100) if r["priorizados"] else 0
        s += f"| {r['abreviatura']} | {r['priorizados']:,} | {r['logrados']:,} | {pl:.1f}% |\n"

    return s


def section_organizations() -> str:
    print("  [9/10] Organizaciones...")

    orgs = query(f"""
        SELECT e.descripcion, e.tipo,
               COUNT(DISTINCT ds.iddtsocioeconomica) as encuestas,
               COUNT(DISTINCT u.idusuario) as mentores
        FROM empresa e
        JOIN usuario u ON u.idempresa = e.idempresa
        JOIN dtsocioeconomica ds ON ds.usuario = u.idusuario
        WHERE e.codpais = 1
        GROUP BY e.descripcion, e.tipo
        ORDER BY encuestas DESC LIMIT 20
    """)

    by_tipo = query(f"""
        SELECT e.tipo, COUNT(DISTINCT e.idempresa) as orgs,
               COUNT(DISTINCT ds.iddtsocioeconomica) as encuestas
        FROM empresa e
        JOIN usuario u ON u.idempresa = e.idempresa
        JOIN dtsocioeconomica ds ON ds.usuario = u.idusuario
        WHERE e.codpais = 1
        GROUP BY e.tipo ORDER BY encuestas DESC
    """)

    s = "## 10. Organizaciones Participantes - Paraguay\n\n"
    s += "### Por tipo de organizacion\n\n| Tipo | Organizaciones | Encuestas |\n|------|---------------|----------|\n"
    for r in by_tipo:
        s += f"| {r['tipo'] or 'N/A'} | {r['orgs']} | {r['encuestas']:,} |\n"

    s += "\n### Top 20 organizaciones\n\n| Organizacion | Tipo | Encuestas | Mentores |\n"
    s += "|-------------|------|-----------|----------|\n"
    for r in orgs:
        s += f"| {r['descripcion'][:40]} | {r['tipo'] or 'N/A'} | {r['encuestas']:,} | {r['mentores']} |\n"

    return s


def section_llm_insights(report: str) -> str:
    print("  [10/10] Generando resumen ejecutivo con LLM...")

    prompt = f"""Eres un analista senior especializado en pobreza multidimensional y el Semaforo de Eliminacion de Pobreza de la Fundacion Paraguaya en Paraguay.

Analiza el siguiente reporte estadistico detallado y genera un RESUMEN EJECUTIVO profesional en espanol que incluya:

1. **Contexto**: Breve descripcion del programa y alcance en Paraguay
2. **Hallazgos principales** (5-7 puntos clave con datos concretos)
3. **Analisis de indicadores criticos**: Los indicadores con mayor pobreza, explicando por que son preocupantes
4. **Fortalezas identificadas**: Donde Paraguay muestra buenos resultados
5. **Impacto del programa**: Analisis de la comparacion linea base vs cumplimiento
6. **Tendencias temporales**: Como evoluciona la situacion y que significa
7. **El Mapa de Vida**: Analisis de la priorizacion y logros
8. **Ecosistema de organizaciones**: Rol de las diferentes organizaciones
9. **Recomendaciones estrategicas** (5-7 recomendaciones concretas basadas en datos)
10. **Conclusion**

Usa datos concretos del reporte. Se riguroso y profesional. Escribe para un publico de tomadores de decisiones.

--- REPORTE ESTADISTICO PARAGUAY ---
{report}
--- FIN REPORTE ---

RESUMEN EJECUTIVO:"""

    return "## 11. Resumen Ejecutivo (generado por IA)\n\n" + llm_analyze(prompt)


# ── Main ────────────────────────────────────────────────────────────


def main():
    print("Generando informe detallado del Semaforo de Pobreza - Paraguay...\n")

    sections = []
    sections.append("# Informe Estadistico del Semaforo de Eliminacion de Pobreza - Paraguay\n")
    sections.append("*Datos filtrados exclusivamente para Paraguay (codpais=1). Generado automaticamente desde db_semaforo.*\n")

    sections.append(section_overview())
    sections.append(section_yearly())
    sections.append(section_monthly())
    sections.append(section_global_distribution())
    sections.append(section_indicators_detail())
    sections.append(section_critical_indicators())
    sections.append(section_baseline_vs_compliance())
    sections.append(section_mapavida())
    sections.append(section_organizations())

    stats_report = "\n\n".join(sections)

    # LLM insights
    full_report = stats_report + "\n\n" + section_llm_insights(stats_report)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\nInforme guardado en: {OUTPUT_PATH}")
    print(f"Tamano: {len(full_report):,} caracteres")


if __name__ == "__main__":
    main()
