"""
Generate an HTML statistical report with embedded charts for Paraguay survey data.
Uses matplotlib/seaborn for charts, encodes them as base64 in the HTML.
Output: data/reporte_estadistico_paraguay.html
"""

import io
import base64
import statistics
import psycopg2
from psycopg2.extras import RealDictCursor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "db_semaforo",
    "user": "sgadmin",
    "password": "sg4dm1n!",
}

OUTPUT_PATH = "data/reporte_estadistico_paraguay.html"

PY_JOIN = """
    JOIN dtsocioeconomica ds ON dp.idsocioeconomica = ds.iddtsocioeconomica
    JOIN usuario u ON ds.usuario = u.idusuario
    JOIN empresa e ON u.idempresa = e.idempresa
"""
PY_WHERE = "e.codpais = 1"
MAIN_SEMAFORO = 70

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
COLORS = {"rojo": "#E74C3C", "amarillo": "#F39C12", "verde": "#27AE60"}

# Dimension mapping for USAID 2020 (idsemaforo=70) questions by nropregunta
# Based on standard Poverty Stoplight methodology
DIMENSION_MAP = {
    1: "Ingreso y empleo", 2: "Ingreso y empleo", 3: "Ingreso y empleo",
    4: "Ingreso y empleo", 5: "Ingreso y empleo",
    6: "Salud y medioambiente", 7: "Salud y medioambiente", 8: "Salud y medioambiente",
    9: "Salud y medioambiente", 10: "Salud y medioambiente", 11: "Salud y medioambiente",
    12: "Salud y medioambiente",
    13: "Vivienda e infraestructura", 14: "Vivienda e infraestructura",
    15: "Vivienda e infraestructura", 16: "Vivienda e infraestructura",
    17: "Vivienda e infraestructura", 18: "Vivienda e infraestructura",
    19: "Vivienda e infraestructura", 20: "Vivienda e infraestructura",
    21: "Vivienda e infraestructura", 22: "Vivienda e infraestructura",
    23: "Educacion y cultura", 24: "Educacion y cultura", 25: "Educacion y cultura",
    26: "Educacion y cultura", 27: "Educacion y cultura", 28: "Educacion y cultura",
    29: "Educacion y cultura",
    30: "Organizacion y participacion", 31: "Organizacion y participacion",
    32: "Organizacion y participacion",
    33: "Interioridad y motivacion", 34: "Interioridad y motivacion",
    35: "Interioridad y motivacion",
}

DIMENSION_COLORS = {
    "Ingreso y empleo": "#2E86C1",
    "Salud y medioambiente": "#27AE60",
    "Vivienda e infraestructura": "#E67E22",
    "Educacion y cultura": "#8E44AD",
    "Organizacion y participacion": "#E74C3C",
    "Interioridad y motivacion": "#F39C12",
}


def query(sql):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def img_tag(b64, alt="chart"):
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="max-width:100%;margin:10px 0;">'


# ── Chart generators ────────────────────────────────────────────────


def chart_global_distribution():
    """Pie chart of global red/yellow/green distribution."""
    print("  [1/9] Distribucion global...")
    data = query(f"""
        SELECT dp.nroopcion, COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        WHERE {PY_WHERE} AND dp.nroopcion IN (1,2,3)
        GROUP BY dp.nroopcion ORDER BY dp.nroopcion
    """)
    labels = ["Rojo\n(Pobreza extrema)", "Amarillo\n(Pobreza)", "Verde\n(No pobreza)"]
    sizes = [r["total"] for r in data]
    colors = [COLORS["rojo"], COLORS["amarillo"], COLORS["verde"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                                        startangle=90, textprops={"fontsize": 11})
    ax1.set_title("Distribucion Global del Semaforo - Paraguay", fontsize=13, fontweight="bold")

    ax2.barh(labels, sizes, color=colors)
    ax2.set_xlabel("Total de respuestas")
    ax2.set_title("Totales por categoria", fontsize=13, fontweight="bold")
    for i, v in enumerate(sizes):
        ax2.text(v + max(sizes) * 0.01, i, f"{v:,}", va="center", fontsize=10)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))

    fig.tight_layout()
    return img_tag(fig_to_base64(fig), "distribucion-global")


def chart_indicators_heatmap():
    """Heatmap of all 35 indicators with red/yellow/green percentages."""
    print("  [2/9] Heatmap de indicadores...")
    data = query(f"""
        SELECT p.abreviatura, p.nropregunta,
               COUNT(*) FILTER (WHERE dp.nroopcion=1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=2) as amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde,
               COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND p.idpregunta BETWEEN 2759 AND 2793
          AND dp.nroopcion IN (1,2,3)
        GROUP BY p.idpregunta, p.abreviatura, p.nropregunta
        ORDER BY p.nropregunta
    """)

    names = [r["abreviatura"] for r in data]
    pct_rojo = [r["rojo"] / r["total"] * 100 for r in data]
    pct_amarillo = [r["amarillo"] / r["total"] * 100 for r in data]
    pct_verde = [r["verde"] / r["total"] * 100 for r in data]

    matrix = np.array([pct_rojo, pct_amarillo, pct_verde])

    fig, ax = plt.subplots(figsize=(20, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["% Rojo", "% Amarillo", "% Verde"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)

    for i in range(3):
        for j in range(len(names)):
            ax.text(j, i, f"{matrix[i, j]:.0f}%", ha="center", va="center",
                    fontsize=7, color="white" if matrix[i, j] < 40 or matrix[i, j] > 80 else "black")

    plt.colorbar(im, ax=ax, label="Porcentaje", shrink=0.8)
    ax.set_title("Mapa de Calor: 35 Indicadores del Semaforo (USAID 2020) - Paraguay", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return img_tag(fig_to_base64(fig), "heatmap-indicadores")


def chart_indicators_bar():
    """Stacked bar chart of all indicators."""
    print("  [3/9] Barras apiladas por indicador...")
    data = query(f"""
        SELECT p.abreviatura, p.nropregunta,
               COUNT(*) FILTER (WHERE dp.nroopcion=1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=2) as amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde,
               COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND p.idpregunta BETWEEN 2759 AND 2793
          AND dp.nroopcion IN (1,2,3)
        GROUP BY p.idpregunta, p.abreviatura, p.nropregunta
        ORDER BY p.nropregunta
    """)

    names = [r["abreviatura"] for r in data]
    pr = [r["rojo"] / r["total"] * 100 for r in data]
    pa = [r["amarillo"] / r["total"] * 100 for r in data]
    pv = [r["verde"] / r["total"] * 100 for r in data]

    fig, ax = plt.subplots(figsize=(16, 8))
    y = range(len(names))
    ax.barh(y, pr, color=COLORS["rojo"], label="Rojo")
    ax.barh(y, pa, left=pr, color=COLORS["amarillo"], label="Amarillo")
    ax.barh(y, pv, left=[pr[i] + pa[i] for i in range(len(names))], color=COLORS["verde"], label="Verde")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title("Distribucion por Indicador - Paraguay", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    return img_tag(fig_to_base64(fig), "barras-indicadores")


def chart_yearly_evolution():
    """Line chart of yearly evolution."""
    print("  [4/9] Evolucion anual...")
    data = query(f"""
        SELECT EXTRACT(YEAR FROM ds.fechaencuesta)::int as anio,
               dp.nroopcion, COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        WHERE {PY_WHERE} AND dp.nroopcion IN (1,2,3)
          AND ds.fechaencuesta IS NOT NULL
          AND EXTRACT(YEAR FROM ds.fechaencuesta) BETWEEN 2017 AND 2025
        GROUP BY anio, dp.nroopcion ORDER BY anio, dp.nroopcion
    """)

    years_data = {}
    for r in data:
        y = r["anio"]
        if y not in years_data:
            years_data[y] = {1: 0, 2: 0, 3: 0}
        years_data[y][r["nroopcion"]] = r["total"]

    years = sorted(years_data.keys())
    totals = {y: sum(years_data[y].values()) for y in years}
    pct_r = [years_data[y][1] / totals[y] * 100 for y in years]
    pct_a = [years_data[y][2] / totals[y] * 100 for y in years]
    pct_v = [years_data[y][3] / totals[y] * 100 for y in years]
    enc_count = query(f"""
        SELECT EXTRACT(YEAR FROM ds.fechaencuesta)::int as anio,
               COUNT(DISTINCT ds.iddtsocioeconomica) as encuestas
        FROM dtsocioeconomica ds
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE} AND ds.fechaencuesta IS NOT NULL
          AND EXTRACT(YEAR FROM ds.fechaencuesta) BETWEEN 2017 AND 2025
        GROUP BY anio ORDER BY anio
    """)
    enc_by_year = {r["anio"]: r["encuestas"] for r in enc_count}

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(years, pct_r, "o-", color=COLORS["rojo"], linewidth=2, markersize=8, label="% Rojo")
    ax1.plot(years, pct_a, "s-", color=COLORS["amarillo"], linewidth=2, markersize=8, label="% Amarillo")
    ax1.plot(years, pct_v, "^-", color=COLORS["verde"], linewidth=2, markersize=8, label="% Verde")
    ax1.fill_between(years, pct_v, alpha=0.1, color=COLORS["verde"])
    ax1.set_ylabel("Porcentaje (%)")
    ax1.set_title("Evolucion Anual del Semaforo - Paraguay (2017-2025)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.set_xticks(years)
    ax1.grid(True, alpha=0.3)
    for i, y in enumerate(years):
        ax1.annotate(f"{pct_v[i]:.1f}%", (y, pct_v[i]), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=8, color=COLORS["verde"])

    enc_vals = [enc_by_year.get(y, 0) for y in years]
    ax2.bar(years, enc_vals, color="#3498DB", alpha=0.7)
    ax2.set_ylabel("Encuestas")
    ax2.set_xlabel("Anio")
    ax2.set_title("Volumen de encuestas por anio", fontsize=12)
    ax2.set_xticks(years)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    for i, v in enumerate(enc_vals):
        ax2.text(years[i], v + max(enc_vals) * 0.02, f"{v:,}", ha="center", fontsize=8)

    fig.tight_layout()
    return img_tag(fig_to_base64(fig), "evolucion-anual")


def chart_monthly_trend():
    """Monthly trend for 2023-2025."""
    print("  [5/9] Tendencia mensual...")
    data = query(f"""
        SELECT TO_CHAR(ds.fechaencuesta, 'YYYY-MM') as periodo,
               COUNT(*) FILTER (WHERE dp.nroopcion=1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=2) as amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde
        FROM dtpregunta dp {PY_JOIN}
        WHERE {PY_WHERE} AND dp.nroopcion IN (1,2,3)
          AND ds.fechaencuesta >= '2023-01-01'
        GROUP BY periodo ORDER BY periodo
    """)

    periodos = [r["periodo"] for r in data]
    pct_v = []
    for r in data:
        t = r["rojo"] + r["amarillo"] + r["verde"]
        pct_v.append(r["verde"] / t * 100 if t else 0)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(range(len(periodos)), pct_v, "-o", color=COLORS["verde"], markersize=4, linewidth=1.5)
    ax.fill_between(range(len(periodos)), pct_v, alpha=0.15, color=COLORS["verde"])
    ax.axhline(y=statistics.mean(pct_v), color="gray", linestyle="--", alpha=0.5, label=f"Media: {statistics.mean(pct_v):.1f}%")
    ax.set_xticks(range(0, len(periodos), 3))
    ax.set_xticklabels([periodos[i] for i in range(0, len(periodos), 3)], rotation=45, ha="right")
    ax.set_ylabel("% Verde")
    ax.set_title("Tendencia Mensual: % Verde (2023-2025) - Paraguay", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return img_tag(fig_to_base64(fig), "tendencia-mensual")


def chart_baseline_vs_compliance():
    """Grouped bar: baseline vs compliance per indicator."""
    print("  [6/9] Linea base vs cumplimiento...")
    data = query(f"""
        SELECT p.abreviatura, p.nropregunta, te.descripcion as tipo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde,
               COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        JOIN tipoencuesta te ON ds.idtipoencuesta = te.idtipoencuesta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND p.idpregunta BETWEEN 2759 AND 2793
          AND dp.nroopcion IN (1,2,3)
          AND te.descripcion IN ('LINEA DE BASE', 'CUMPLIMIENTO')
        GROUP BY p.abreviatura, p.nropregunta, te.descripcion
        ORDER BY p.nropregunta
    """)

    ind_map = {}
    order = []
    for r in data:
        name = r["abreviatura"]
        if name not in ind_map:
            ind_map[name] = {}
            order.append(name)
        ind_map[name][r["tipo"]] = r["verde"] / r["total"] * 100 if r["total"] else 0

    names = order
    base = [ind_map[n].get("LINEA DE BASE", 0) for n in names]
    cumpl = [ind_map[n].get("CUMPLIMIENTO", 0) for n in names]
    diff = [cumpl[i] - base[i] for i in range(len(names))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={"width_ratios": [3, 1]})

    x = np.arange(len(names))
    w = 0.35
    ax1.barh(x - w / 2, base, w, color="#85C1E9", label="Linea de Base")
    ax1.barh(x + w / 2, cumpl, w, color="#2E86C1", label="Cumplimiento")
    ax1.set_yticks(x)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("% Verde")
    ax1.set_title("% Verde: Linea Base vs Cumplimiento", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.invert_yaxis()

    colors_diff = [COLORS["verde"] if d >= 0 else COLORS["rojo"] for d in diff]
    ax2.barh(x, diff, color=colors_diff)
    ax2.set_yticks(x)
    ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xlabel("Cambio (pp)")
    ax2.set_title("Cambio (puntos porcentuales)", fontsize=12)
    ax2.axvline(x=0, color="black", linewidth=0.5)
    ax2.invert_yaxis()
    for i, v in enumerate(diff):
        ax2.text(v + (1 if v >= 0 else -1), i, f"{v:+.1f}", va="center", fontsize=7)

    fig.tight_layout()
    return img_tag(fig_to_base64(fig), "base-vs-cumplimiento")


def chart_critical_distribution():
    """Distribution (violin/box) of % red across indicators."""
    print("  [7/9] Distribucion estadistica de indicadores...")
    data = query(f"""
        SELECT p.abreviatura,
               COUNT(*) FILTER (WHERE dp.nroopcion=1)::float / COUNT(*)::float * 100 as pct_rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=2)::float / COUNT(*)::float * 100 as pct_amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3)::float / COUNT(*)::float * 100 as pct_verde
        FROM dtpregunta dp {PY_JOIN}
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND p.idpregunta BETWEEN 2759 AND 2793
          AND dp.nroopcion IN (1,2,3)
        GROUP BY p.idpregunta, p.abreviatura
    """)

    pct_rojo = [r["pct_rojo"] for r in data]
    pct_amarillo = [r["pct_amarillo"] for r in data]
    pct_verde = [r["pct_verde"] for r in data]

    mean_r = statistics.mean(pct_rojo)
    median_r = statistics.median(pct_rojo)
    mode_r = max(set([round(v, 0) for v in pct_rojo]), key=[round(v, 0) for v in pct_rojo].count)
    mean_v = statistics.mean(pct_verde)
    median_v = statistics.median(pct_verde)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Box plot
    bp = axes[0].boxplot([pct_rojo, pct_amarillo, pct_verde],
                          labels=["% Rojo", "% Amarillo", "% Verde"],
                          patch_artist=True, showmeans=True,
                          meanprops={"marker": "D", "markerfacecolor": "black", "markersize": 8})
    for patch, color in zip(bp["boxes"], [COLORS["rojo"], COLORS["amarillo"], COLORS["verde"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel("Porcentaje (%)")
    axes[0].set_title("Diagrama de Caja", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Histogram of % red
    axes[1].hist(pct_rojo, bins=15, color=COLORS["rojo"], alpha=0.7, edgecolor="white")
    axes[1].axvline(mean_r, color="black", linestyle="-", linewidth=2, label=f"Media: {mean_r:.1f}%")
    axes[1].axvline(median_r, color="blue", linestyle="--", linewidth=2, label=f"Mediana: {median_r:.1f}%")
    axes[1].axvline(mode_r, color="purple", linestyle=":", linewidth=2, label=f"Moda: {mode_r:.0f}%")
    axes[1].set_xlabel("% Rojo")
    axes[1].set_ylabel("Frecuencia (indicadores)")
    axes[1].set_title("Distribucion del % Rojo", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9)

    # Histogram of % green
    axes[2].hist(pct_verde, bins=15, color=COLORS["verde"], alpha=0.7, edgecolor="white")
    axes[2].axvline(mean_v, color="black", linestyle="-", linewidth=2, label=f"Media: {mean_v:.1f}%")
    axes[2].axvline(median_v, color="blue", linestyle="--", linewidth=2, label=f"Mediana: {median_v:.1f}%")
    axes[2].set_xlabel("% Verde")
    axes[2].set_ylabel("Frecuencia (indicadores)")
    axes[2].set_title("Distribucion del % Verde", fontsize=12, fontweight="bold")
    axes[2].legend(fontsize=9)

    fig.suptitle("Estadisticas Descriptivas de los 35 Indicadores", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    stats_html = f"""
    <div class="stats-box">
        <h4>Medidas de Tendencia Central (% Rojo por indicador)</h4>
        <table><tr><th>Metrica</th><th>Valor</th></tr>
        <tr><td>Media</td><td>{mean_r:.2f}%</td></tr>
        <tr><td>Mediana</td><td>{median_r:.2f}%</td></tr>
        <tr><td>Moda</td><td>{mode_r:.0f}%</td></tr>
        <tr><td>Desviacion Estandar</td><td>{statistics.stdev(pct_rojo):.2f}%</td></tr>
        <tr><td>Minimo</td><td>{min(pct_rojo):.2f}%</td></tr>
        <tr><td>Maximo</td><td>{max(pct_rojo):.2f}%</td></tr>
        </table>
        <h4>Medidas de Tendencia Central (% Verde por indicador)</h4>
        <table><tr><th>Metrica</th><th>Valor</th></tr>
        <tr><td>Media</td><td>{mean_v:.2f}%</td></tr>
        <tr><td>Mediana</td><td>{median_v:.2f}%</td></tr>
        <tr><td>Desviacion Estandar</td><td>{statistics.stdev(pct_verde):.2f}%</td></tr>
        <tr><td>Minimo</td><td>{min(pct_verde):.2f}%</td></tr>
        <tr><td>Maximo</td><td>{max(pct_verde):.2f}%</td></tr>
        </table>
    </div>
    """
    return img_tag(fig_to_base64(fig), "distribucion-estadistica") + stats_html


def chart_mapavida():
    """Life map priorities bar chart."""
    print("  [8/9] Mapa de vida...")
    data = query(f"""
        SELECT p.abreviatura, COUNT(*) as priorizados
        FROM mapavida mv
        JOIN pregunta p ON mv.idpregunta = p.idpregunta
        JOIN dtsocioeconomica ds ON mv.idsocioeconomica = ds.iddtsocioeconomica
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE} AND mv.priorizado = true
        GROUP BY p.abreviatura ORDER BY priorizados DESC LIMIT 15
    """)

    names = [r["abreviatura"] for r in data]
    vals = [r["priorizados"] for r in data]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(names)), vals, color=sns.color_palette("YlOrRd_r", len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Veces priorizado")
    ax.set_title("Indicadores mas Priorizados en el Mapa de Vida - Paraguay", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=9)
    fig.tight_layout()
    return img_tag(fig_to_base64(fig), "mapa-de-vida")


def chart_organizations():
    """Top organizations bar chart."""
    print("  [9/9] Organizaciones...")
    data = query(f"""
        SELECT e.descripcion, e.tipo,
               COUNT(DISTINCT ds.iddtsocioeconomica) as encuestas
        FROM empresa e
        JOIN usuario u ON u.idempresa = e.idempresa
        JOIN dtsocioeconomica ds ON ds.usuario = u.idusuario
        WHERE e.codpais = 1
        GROUP BY e.descripcion, e.tipo
        ORDER BY encuestas DESC LIMIT 15
    """)

    names = [r["descripcion"][:30] for r in data]
    vals = [r["encuestas"] for r in data]
    tipos = [r["tipo"] or "N/A" for r in data]
    color_tipo = {"ONG": "#3498DB", "EMPRESA": "#E67E22", "GOBIERNO": "#9B59B6"}

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(range(len(names)), vals,
                   color=[color_tipo.get(t, "#95A5A6") for t in tipos])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Encuestas realizadas")
    ax.set_title("Top 15 Organizaciones - Paraguay", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=v, label=k) for k, v in color_tipo.items()]
    ax.legend(handles=legend_items, loc="lower right")

    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=8)

    fig.tight_layout()
    return img_tag(fig_to_base64(fig), "organizaciones")


def chart_by_dimension():
    """Grouped bar chart and radar chart by dimension."""
    print("  [10/11] Agrupado por dimension...")
    data = query(f"""
        SELECT p.nropregunta,
               COUNT(*) FILTER (WHERE dp.nroopcion=1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=2) as amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde,
               COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND p.idpregunta BETWEEN 2759 AND 2793
          AND dp.nroopcion IN (1,2,3)
        GROUP BY p.nropregunta
        ORDER BY p.nropregunta
    """)

    # Aggregate by dimension
    dim_agg = {}
    dim_order = list(dict.fromkeys(DIMENSION_MAP.values()))
    for dim in dim_order:
        dim_agg[dim] = {"rojo": 0, "amarillo": 0, "verde": 0, "total": 0}
    for r in data:
        dim = DIMENSION_MAP.get(r["nropregunta"])
        if dim:
            dim_agg[dim]["rojo"] += r["rojo"]
            dim_agg[dim]["amarillo"] += r["amarillo"]
            dim_agg[dim]["verde"] += r["verde"]
            dim_agg[dim]["total"] += r["total"]

    dims = dim_order
    pct_r = [dim_agg[d]["rojo"] / dim_agg[d]["total"] * 100 if dim_agg[d]["total"] else 0 for d in dims]
    pct_a = [dim_agg[d]["amarillo"] / dim_agg[d]["total"] * 100 if dim_agg[d]["total"] else 0 for d in dims]
    pct_v = [dim_agg[d]["verde"] / dim_agg[d]["total"] * 100 if dim_agg[d]["total"] else 0 for d in dims]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [3, 2]})

    # Stacked horizontal bar
    y = np.arange(len(dims))
    ax1.barh(y, pct_r, color=COLORS["rojo"], label="Rojo")
    ax1.barh(y, pct_a, left=pct_r, color=COLORS["amarillo"], label="Amarillo")
    ax1.barh(y, pct_v, left=[pct_r[i] + pct_a[i] for i in range(len(dims))], color=COLORS["verde"], label="Verde")
    ax1.set_yticks(y)
    ax1.set_yticklabels(dims, fontsize=10)
    ax1.set_xlabel("Porcentaje (%)")
    ax1.set_title("Distribucion por Dimension", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower right")
    ax1.invert_yaxis()
    for i in range(len(dims)):
        if pct_r[i] > 3:
            ax1.text(pct_r[i] / 2, i, f"{pct_r[i]:.1f}%", ha="center", va="center", fontsize=8, color="white")
        ax1.text(pct_r[i] + pct_a[i] / 2, i, f"{pct_a[i]:.1f}%", ha="center", va="center", fontsize=8)

    # Radar chart
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]
    pct_v_radar = pct_v + [pct_v[0]]
    pct_r_radar = pct_r + [pct_r[0]]

    ax2 = fig.add_subplot(122, polar=True)
    ax2.plot(angles, pct_v_radar, "o-", color=COLORS["verde"], linewidth=2, label="% Verde")
    ax2.fill(angles, pct_v_radar, alpha=0.15, color=COLORS["verde"])
    ax2.plot(angles, pct_r_radar, "s-", color=COLORS["rojo"], linewidth=2, label="% Rojo")
    ax2.fill(angles, pct_r_radar, alpha=0.15, color=COLORS["rojo"])
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels([d.split(" y ")[0][:12] for d in dims], fontsize=8)
    ax2.set_ylim(0, 100)
    ax2.set_title("Radar: % Verde vs % Rojo por Dimension", fontsize=11, fontweight="bold", pad=20)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    fig.tight_layout()

    # Summary table
    table_html = """
    <div class="stats-box">
        <h4>Resumen por Dimension</h4>
        <table>
        <tr><th>Dimension</th><th>Indicadores</th><th>Respuestas</th><th>% Rojo</th><th>% Amarillo</th><th>% Verde</th></tr>
    """
    for i, d in enumerate(dims):
        n_ind = sum(1 for v in DIMENSION_MAP.values() if v == d)
        table_html += f"<tr><td>{d}</td><td>{n_ind}</td><td>{dim_agg[d]['total']:,}</td>"
        table_html += f"<td style='color:{COLORS['rojo']};font-weight:bold'>{pct_r[i]:.1f}%</td>"
        table_html += f"<td style='color:{COLORS['amarillo']};font-weight:bold'>{pct_a[i]:.1f}%</td>"
        table_html += f"<td style='color:{COLORS['verde']};font-weight:bold'>{pct_v[i]:.1f}%</td></tr>"
    table_html += "</table></div>"

    return img_tag(fig_to_base64(fig), "por-dimension") + table_html


def chart_by_question():
    """Detailed chart and table per question with dimension grouping."""
    print("  [11/11] Detalle por pregunta...")
    data = query(f"""
        SELECT p.nropregunta, p.titulo, p.abreviatura,
               COUNT(*) FILTER (WHERE dp.nroopcion=1) as rojo,
               COUNT(*) FILTER (WHERE dp.nroopcion=2) as amarillo,
               COUNT(*) FILTER (WHERE dp.nroopcion=3) as verde,
               COUNT(*) as total
        FROM dtpregunta dp {PY_JOIN}
        JOIN pregunta p ON dp.idpregunta = p.idpregunta
        WHERE {PY_WHERE} AND dp.idsemaforo = {MAIN_SEMAFORO}
          AND p.idpregunta BETWEEN 2759 AND 2793
          AND dp.nroopcion IN (1,2,3)
        GROUP BY p.idpregunta, p.nropregunta, p.titulo, p.abreviatura
        ORDER BY p.nropregunta
    """)

    # Group questions by dimension for the chart
    dim_order = list(dict.fromkeys(DIMENSION_MAP.values()))
    grouped = {d: [] for d in dim_order}
    for r in data:
        dim = DIMENSION_MAP.get(r["nropregunta"], "Sin dimension")
        grouped[dim].append(r)

    # Stacked bar chart with dimension color-coding on left axis
    fig, ax = plt.subplots(figsize=(16, 14))

    names = []
    pct_r = []
    pct_a = []
    pct_v = []
    dim_labels = []
    for dim in dim_order:
        for r in grouped[dim]:
            names.append(r["abreviatura"])
            pct_r.append(r["rojo"] / r["total"] * 100)
            pct_a.append(r["amarillo"] / r["total"] * 100)
            pct_v.append(r["verde"] / r["total"] * 100)
            dim_labels.append(dim)

    y = np.arange(len(names))
    ax.barh(y, pct_r, color=COLORS["rojo"], label="Rojo")
    ax.barh(y, pct_a, left=pct_r, color=COLORS["amarillo"], label="Amarillo")
    ax.barh(y, pct_v, left=[pct_r[i] + pct_a[i] for i in range(len(names))], color=COLORS["verde"], label="Verde")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Porcentaje (%)")
    ax.set_title("Detalle por Pregunta (agrupado por Dimension)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    # Add dimension color indicators on the right
    for i, dim in enumerate(dim_labels):
        ax.plot(102, i, "s", color=DIMENSION_COLORS.get(dim, "gray"), markersize=8, clip_on=False)

    # Add dimension separators
    current_dim = None
    for i, dim in enumerate(dim_labels):
        if dim != current_dim:
            if current_dim is not None:
                ax.axhline(y=i - 0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.text(103, i, dim.split(" y ")[0][:15], fontsize=7, va="center",
                    color=DIMENSION_COLORS.get(dim, "gray"), fontweight="bold", clip_on=False)
            current_dim = dim

    ax.set_xlim(0, 100)
    fig.tight_layout()

    # Detailed HTML table grouped by dimension
    table_html = """
    <div class="stats-box">
        <h4>Tabla Detallada por Pregunta y Dimension</h4>
        <table>
        <tr><th>Nro</th><th>Dimension</th><th>Indicador</th><th>Rojo</th><th>Amarillo</th><th>Verde</th><th>Total</th><th>% Rojo</th><th>% Verde</th></tr>
    """
    for dim in dim_order:
        for r in grouped[dim]:
            pr = r["rojo"] / r["total"] * 100
            pv = r["verde"] / r["total"] * 100
            # Color-code based on severity
            rojo_style = f"color:{COLORS['rojo']};font-weight:bold" if pr > 5 else ""
            verde_style = f"color:{COLORS['verde']};font-weight:bold" if pv > 90 else ""
            table_html += f"""<tr>
                <td>{r['nropregunta']}</td>
                <td style="color:{DIMENSION_COLORS.get(dim, 'gray')};font-weight:bold">{dim}</td>
                <td>{r['titulo']}</td>
                <td>{r['rojo']:,}</td><td>{r['amarillo']:,}</td><td>{r['verde']:,}</td><td>{r['total']:,}</td>
                <td style="{rojo_style}">{pr:.1f}%</td>
                <td style="{verde_style}">{pv:.1f}%</td>
            </tr>"""
    table_html += "</table></div>"

    return img_tag(fig_to_base64(fig), "detalle-pregunta") + table_html


# ── HTML template ───────────────────────────────────────────────────


def build_html(charts: dict) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Reporte Estadistico - Semaforo de Pobreza Paraguay</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    header {{ background: linear-gradient(135deg, #2c3e50, #3498db); color: white; padding: 40px 20px; text-align: center; margin-bottom: 30px; border-radius: 8px; }}
    header h1 {{ font-size: 2em; margin-bottom: 8px; }}
    header p {{ font-size: 1.1em; opacity: 0.9; }}
    .section {{ background: white; border-radius: 8px; padding: 25px; margin-bottom: 25px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .section h2 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; font-size: 1.4em; }}
    .section h3 {{ color: #34495e; margin: 15px 0 10px 0; }}
    .section img {{ display: block; margin: 15px auto; border-radius: 4px; }}
    .stats-box {{ background: #f8f9fa; border-left: 4px solid #3498db; padding: 15px 20px; margin: 15px 0; border-radius: 0 4px 4px 0; }}
    .stats-box h4 {{ color: #2c3e50; margin-bottom: 10px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
    th {{ background: #3498db; color: white; }}
    tr:nth-child(even) {{ background: #f2f2f2; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 20px; }}
    .kpi {{ background: #f8f9fa; border-radius: 8px; padding: 20px; text-align: center; border-top: 4px solid #3498db; }}
    .kpi .value {{ font-size: 1.8em; font-weight: bold; color: #2c3e50; }}
    .kpi .label {{ font-size: 0.85em; color: #7f8c8d; margin-top: 5px; }}
    footer {{ text-align: center; padding: 20px; color: #7f8c8d; font-size: 0.9em; }}
</style>
</head>
<body>
<div class="container">

<header>
    <h1>Reporte Estadistico del Semaforo de Eliminacion de Pobreza</h1>
    <p>Paraguay - Datos filtrados exclusivamente para codpais=1</p>
</header>

<div class="section">
    <h2>1. Panorama General</h2>
    <div class="kpi-grid">
        {charts['kpis']}
    </div>
</div>

<div class="section">
    <h2>2. Distribucion Global del Semaforo</h2>
    {charts['global_dist']}
</div>

<div class="section">
    <h2>3. Mapa de Calor por Indicador</h2>
    <p>Visualizacion de los 35 indicadores del Semaforo USAID 2020 con porcentajes de rojo, amarillo y verde.</p>
    {charts['heatmap']}
</div>

<div class="section">
    <h2>4. Distribucion por Indicador</h2>
    {charts['indicators_bar']}
</div>

<div class="section">
    <h2>5. Evolucion Temporal</h2>
    {charts['yearly']}
</div>

<div class="section">
    <h2>6. Tendencia Mensual (2023-2025)</h2>
    {charts['monthly']}
</div>

<div class="section">
    <h2>7. Estadisticas Descriptivas: Media, Mediana, Moda y Distribucion</h2>
    {charts['stats_dist']}
</div>

<div class="section">
    <h2>8. Impacto: Linea de Base vs Cumplimiento</h2>
    <p>Comparacion del porcentaje verde entre la encuesta inicial (Linea de Base) y la de seguimiento (Cumplimiento).</p>
    {charts['base_vs_cumpl']}
</div>

<div class="section">
    <h2>9. Mapa de Vida: Indicadores Priorizados</h2>
    {charts['mapavida']}
</div>

<div class="section">
    <h2>10. Organizaciones Participantes</h2>
    {charts['orgs']}
</div>

<footer>
    Generado automaticamente | Semaforo de Eliminacion de Pobreza - Fundacion Paraguaya
</footer>

</div>
</body>
</html>"""


# ── Main ────────────────────────────────────────────────────────────


def main():
    print("Generando reporte HTML estadistico - Paraguay...\n")

    # KPIs
    print("  [0/9] KPIs...")
    kpi_data = query(f"""
        SELECT
            COUNT(DISTINCT ds.iddtsocioeconomica) as encuestas,
            COUNT(DISTINCT ds.idpersona) FILTER (WHERE ds.idpersona IS NOT NULL) as personas
        FROM dtsocioeconomica ds
        JOIN usuario u ON ds.usuario = u.idusuario
        JOIN empresa e ON u.idempresa = e.idempresa
        WHERE {PY_WHERE}
    """)[0]
    resp_count = query(f"SELECT COUNT(*) as n FROM dtpregunta dp {PY_JOIN} WHERE {PY_WHERE}")[0]["n"]
    org_count = query("SELECT COUNT(*) as n FROM empresa WHERE codpais = 1")[0]["n"]

    kpis_html = f"""
        <div class="kpi"><div class="value">{kpi_data['encuestas']:,}</div><div class="label">Encuestas</div></div>
        <div class="kpi"><div class="value">{kpi_data['personas']:,}</div><div class="label">Personas</div></div>
        <div class="kpi"><div class="value">{resp_count:,}</div><div class="label">Respuestas</div></div>
        <div class="kpi"><div class="value">{org_count}</div><div class="label">Organizaciones</div></div>
    """

    charts = {
        "kpis": kpis_html,
        "global_dist": chart_global_distribution(),
        "heatmap": chart_indicators_heatmap(),
        "indicators_bar": chart_indicators_bar(),
        "yearly": chart_yearly_evolution(),
        "monthly": chart_monthly_trend(),
        "stats_dist": chart_critical_distribution(),
        "base_vs_cumpl": chart_baseline_vs_compliance(),
        "mapavida": chart_mapavida(),
        "orgs": chart_organizations(),
    }

    html = build_html(charts)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\nReporte HTML guardado en: {OUTPUT_PATH}")
    print(f"Tamano: {len(html):,} caracteres")


if __name__ == "__main__":
    main()
