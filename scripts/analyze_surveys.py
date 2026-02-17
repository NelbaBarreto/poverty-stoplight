"""
Analyze db_semaforo survey data and generate statistical summaries.
Uses Ollama (gpt-oss:20b) to produce narrative insights from the statistics.
Outputs a preview report (Markdown) for review before RAG ingestion.
"""

import json
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
OUTPUT_PATH = "data/analisis_encuestas_semaforo.md"


def query(sql: str) -> list[dict]:
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def llm_analyze(prompt: str) -> str:
    """Send a prompt to Ollama and return the response."""
    resp = requests.post(
        OLLAMA_URL,
        json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def general_stats() -> str:
    """Overall survey statistics."""
    print("  [1/7] Estadisticas generales...")

    total_surveys = query("SELECT COUNT(*) as n FROM dtsocioeconomica")[0]["n"]
    total_answers = query("SELECT COUNT(*) as n FROM dtpregunta")[0]["n"]
    total_persons = query("SELECT COUNT(*) as n FROM persona")[0]["n"]
    total_companies = query("SELECT COUNT(*) as n FROM empresa")[0]["n"]
    date_range = query(
        "SELECT MIN(fechaencuesta) as min_date, MAX(fechaencuesta) as max_date "
        "FROM dtsocioeconomica WHERE fechaencuesta IS NOT NULL"
    )[0]
    surveys_by_year = query("""
        SELECT EXTRACT(YEAR FROM fechaencuesta)::int as anio, COUNT(*) as total
        FROM dtsocioeconomica
        WHERE fechaencuesta IS NOT NULL AND EXTRACT(YEAR FROM fechaencuesta) >= 2015
        GROUP BY anio ORDER BY anio
    """)
    semaforos = query("""
        SELECT s.idsemaforo, s.descripcion, COUNT(dp.iddtpregunta) as respuestas
        FROM semaforo s
        LEFT JOIN dtpregunta dp ON s.idsemaforo = dp.idsemaforo
        GROUP BY s.idsemaforo, s.descripcion
        ORDER BY respuestas DESC
    """)

    section = "## 1. Estadisticas Generales\n\n"
    section += f"- **Total de encuestas realizadas**: {total_surveys:,}\n"
    section += f"- **Total de respuestas (indicadores evaluados)**: {total_answers:,}\n"
    section += f"- **Personas registradas**: {total_persons:,}\n"
    section += f"- **Empresas registradas**: {total_companies:,}\n"
    section += f"- **Periodo de datos**: {date_range['min_date']} a {date_range['max_date']}\n\n"

    section += "### Encuestas por anio\n\n| Anio | Total |\n|------|-------|\n"
    for r in surveys_by_year:
        section += f"| {r['anio']} | {r['total']:,} |\n"

    section += "\n### Semaforos (tipos de encuesta)\n\n| Semaforo | Respuestas |\n|----------|------------|\n"
    for r in semaforos[:15]:
        section += f"| {r['descripcion']} | {r['respuestas']:,} |\n"

    return section


def traffic_light_distribution() -> str:
    """Distribution of red/yellow/green across all indicators."""
    print("  [2/7] Distribucion semaforo (rojo/amarillo/verde)...")

    # Overall distribution (nroopcion: 1=rojo, 2=amarillo, 3=verde)
    overall = query("""
        SELECT nroopcion, COUNT(*) as total
        FROM dtpregunta
        WHERE nroopcion IN (1,2,3)
        GROUP BY nroopcion ORDER BY nroopcion
    """)

    # By dimension
    by_dimension = query("""
        SELECT d.descripcion as dimension, dp.nroopcion, COUNT(*) as total
        FROM dtpregunta dp
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        JOIN dimension d ON p.iddimension = d.iddimension
        WHERE dp.nroopcion IN (1,2,3)
        GROUP BY d.descripcion, dp.nroopcion
        ORDER BY d.descripcion, dp.nroopcion
    """)

    color_map = {1: "Rojo (pobreza extrema)", 2: "Amarillo (pobreza)", 3: "Verde (no pobreza)"}
    total_all = sum(r["total"] for r in overall)

    section = "## 2. Distribucion Semaforo Global\n\n"
    section += "| Color | Total | Porcentaje |\n|-------|-------|------------|\n"
    for r in overall:
        pct = (r["total"] / total_all * 100) if total_all else 0
        section += f"| {color_map.get(r['nroopcion'], r['nroopcion'])} | {r['total']:,} | {pct:.1f}% |\n"

    # Pivot by dimension
    dims = {}
    for r in by_dimension:
        dim = r["dimension"]
        if dim not in dims:
            dims[dim] = {1: 0, 2: 0, 3: 0}
        dims[dim][r["nroopcion"]] = r["total"]

    section += "\n### Por dimension\n\n"
    section += "| Dimension | Rojo | Amarillo | Verde | % Verde |\n|-----------|------|----------|-------|----------|\n"
    for dim, counts in sorted(dims.items()):
        total = sum(counts.values())
        pct_green = (counts[3] / total * 100) if total else 0
        section += f"| {dim} | {counts[1]:,} | {counts[2]:,} | {counts[3]:,} | {pct_green:.1f}% |\n"

    return section


def top_bottom_indicators() -> str:
    """Indicators with most red and most green."""
    print("  [3/7] Indicadores criticos y fortalezas...")

    indicators = query("""
        SELECT p.titulo, p.abreviatura,
               COUNT(*) FILTER (WHERE dp.nroopcion = 1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion = 2) as amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion = 3) as verde,
               COUNT(*) as total
        FROM dtpregunta dp
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        WHERE dp.nroopcion IN (1,2,3) AND dp.idsemaforo = 1
        GROUP BY p.idpregunta, p.titulo, p.abreviatura
        HAVING COUNT(*) > 100
        ORDER BY p.idpregunta
    """)

    # Sort by % red (worst)
    for ind in indicators:
        ind["pct_rojo"] = (ind["rojo"] / ind["total"] * 100) if ind["total"] else 0
        ind["pct_verde"] = (ind["verde"] / ind["total"] * 100) if ind["total"] else 0

    worst = sorted(indicators, key=lambda x: -x["pct_rojo"])[:10]
    best = sorted(indicators, key=lambda x: -x["pct_verde"])[:10]

    section = "## 3. Indicadores Criticos (mayor % rojo)\n\n"
    section += "| Indicador | Rojo | Amarillo | Verde | % Rojo |\n|-----------|------|----------|-------|--------|\n"
    for r in worst:
        section += f"| {r['titulo'][:50]} | {r['rojo']:,} | {r['amarillo']:,} | {r['verde']:,} | {r['pct_rojo']:.1f}% |\n"

    section += "\n## 4. Indicadores Fortaleza (mayor % verde)\n\n"
    section += "| Indicador | Rojo | Amarillo | Verde | % Verde |\n|-----------|------|----------|-------|--------|\n"
    for r in best:
        section += f"| {r['titulo'][:50]} | {r['rojo']:,} | {r['amarillo']:,} | {r['verde']:,} | {r['pct_verde']:.1f}% |\n"

    return section


def yearly_evolution() -> str:
    """Evolution of traffic light colors over years."""
    print("  [4/7] Evolucion temporal...")

    evolution = query("""
        SELECT EXTRACT(YEAR FROM ds.fechaencuesta)::int as anio,
               dp.nroopcion, COUNT(*) as total
        FROM dtpregunta dp
        JOIN dtsocioeconomica ds ON dp.idsocioeconomica = ds.iddtsocioeconomica
        WHERE dp.nroopcion IN (1,2,3)
          AND ds.fechaencuesta IS NOT NULL
          AND EXTRACT(YEAR FROM ds.fechaencuesta) >= 2015
        GROUP BY anio, dp.nroopcion
        ORDER BY anio, dp.nroopcion
    """)

    years = {}
    for r in evolution:
        y = r["anio"]
        if y not in years:
            years[y] = {1: 0, 2: 0, 3: 0}
        years[y][r["nroopcion"]] = r["total"]

    section = "## 5. Evolucion Temporal\n\n"
    section += "| Anio | Rojo | Amarillo | Verde | Total | % Verde |\n|------|------|----------|-------|-------|---------|\n"
    for y in sorted(years):
        c = years[y]
        total = sum(c.values())
        pct = (c[3] / total * 100) if total else 0
        section += f"| {y} | {c[1]:,} | {c[2]:,} | {c[3]:,} | {total:,} | {pct:.1f}% |\n"

    return section


def mapavida_stats() -> str:
    """Life map (mapavida) prioritization and achievement stats."""
    print("  [5/7] Mapa de vida...")

    total_mapa = query("SELECT COUNT(*) as n FROM mapavida")[0]["n"]
    priorizados = query("SELECT COUNT(*) as n FROM mapavida WHERE priorizado = true")[0]["n"]
    logrados = query("SELECT COUNT(*) as n FROM mapavida WHERE logro = true")[0]["n"]

    top_priorizados = query("""
        SELECT p.titulo, COUNT(*) as total
        FROM mapavida mv
        JOIN pregunta p ON mv.idpregunta = p.idpregunta
        WHERE mv.priorizado = true
        GROUP BY p.titulo
        ORDER BY total DESC
        LIMIT 10
    """)

    section = "## 6. Mapa de Vida (priorizacion y logros)\n\n"
    section += f"- **Total registros mapa de vida**: {total_mapa:,}\n"
    section += f"- **Indicadores priorizados**: {priorizados:,}\n"
    section += f"- **Logros alcanzados**: {logrados:,}\n"
    pct_logro = (logrados / priorizados * 100) if priorizados else 0
    section += f"- **Tasa de logro**: {pct_logro:.1f}%\n\n"

    if top_priorizados:
        section += "### Indicadores mas priorizados\n\n| Indicador | Veces priorizado |\n|-----------|------------------|\n"
        for r in top_priorizados:
            section += f"| {r['titulo'][:50]} | {r['total']:,} |\n"

    return section


def company_stats() -> str:
    """Enterprise-level statistics."""
    print("  [6/7] Estadisticas por empresa...")

    by_company = query("""
        SELECT e.descripcion, COUNT(DISTINCT ds.iddtsocioeconomica) as encuestas
        FROM empresa e
        JOIN usuario u ON u.idempresa = e.idempresa
        JOIN dtsocioeconomica ds ON ds.usuario = u.idusuario
        GROUP BY e.descripcion
        ORDER BY encuestas DESC
        LIMIT 15
    """)

    section = "## 7. Principales organizaciones (por encuestas)\n\n"
    section += "| Organizacion | Encuestas |\n|-------------|----------|\n"
    for r in by_company:
        section += f"| {r['descripcion'][:50]} | {r['encuestas']:,} |\n"

    return section


def generate_llm_insights(report_text: str) -> str:
    """Use LLM to generate narrative insights from the statistics."""
    print("  [7/7] Generando analisis con LLM...")

    prompt = f"""Eres un analista de datos especializado en pobreza multidimensional y el Semaforo de Eliminacion de Pobreza de la Fundacion Paraguaya.

A continuacion tienes un reporte estadistico de las encuestas del Semaforo de Pobreza. Analiza los datos y escribe un resumen ejecutivo en espanol con:

1. **Hallazgos principales**: Los 3-5 hallazgos mas relevantes
2. **Areas criticas**: Donde se concentra la pobreza (indicadores rojos)
3. **Fortalezas**: Que indicadores muestran mejores resultados
4. **Tendencias temporales**: Como ha evolucionado la situacion
5. **Recomendaciones**: Basadas en los datos

Escribe de forma clara, concisa y profesional. Usa datos concretos del reporte.

--- REPORTE ESTADISTICO ---
{report_text}
--- FIN REPORTE ---

Resumen ejecutivo:"""

    return llm_analyze(prompt)


def main():
    print("Analizando encuestas del Semaforo de Pobreza...\n")

    sections = []

    # Header
    sections.append("# Analisis Estadistico - Semaforo de Eliminacion de Pobreza\n")
    sections.append("*Generado automaticamente a partir de la base de datos db_semaforo*\n")

    # Generate each statistical section
    sections.append(general_stats())
    sections.append(traffic_light_distribution())
    sections.append(top_bottom_indicators())
    sections.append(yearly_evolution())
    sections.append(mapavida_stats())
    sections.append(company_stats())

    # Combine stats
    stats_report = "\n\n".join(sections)

    # Generate LLM insights
    insights = generate_llm_insights(stats_report)
    full_report = stats_report + "\n\n## 8. Resumen Ejecutivo (generado por IA)\n\n" + insights

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(f"\nReporte guardado en: {OUTPUT_PATH}")
    print(f"Tamano: {len(full_report):,} caracteres")


if __name__ == "__main__":
    main()
