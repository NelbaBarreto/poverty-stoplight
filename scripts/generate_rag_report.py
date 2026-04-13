#!/usr/bin/env python3
"""
Generate self-contained interactive HTML evaluation reports (EN + ES).

Usage:
    python scripts/generate_rag_report.py --questions 30 --lang both
    python scripts/generate_rag_report.py --questions 30 --lang en
    python scripts/generate_rag_report.py --questions 30 --lang es
"""

import os
import sys
import json
import math
import logging
import argparse
from datetime import datetime
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------

TRANSLATIONS = {
    "en": {
        # ── page chrome ─────────────────────────────────────────────────────
        "title":          "RAG Evaluation Report — Semáforo de Pobreza",
        "subtitle":       "Judge LLM: {judge_llm} &nbsp;|&nbsp; Judge Embeddings: bge-m3 &nbsp;|&nbsp; Generated: {generated_at}",
        "lang_label":     "EN",
        # ── info notice ──────────────────────────────────────────────────────
        "notice": (
            "<strong>ℹ {n_scored:,} runs scored</strong> across {n_combos} LLM × embedding × "
            "chunk combinations and {n_q} questions. "
            "Average score uses all non-null metrics (Faithfulness, Answer Relevancy, "
            "Context Precision, Context Recall) weighted equally."
        ),
        # ── stat boxes ───────────────────────────────────────────────────────
        "lbl_runs":       "Runs scored",
        "lbl_combos":     "Combinations",
        "lbl_questions":  "Questions",
        "lbl_best_avg":   "Best avg score",
        "lbl_avg_faith":  "Avg faithfulness",
        "lbl_avg_ar":     "Avg ans. relevancy",
        "lbl_avg_cr":     "Avg context recall",
        "lbl_avg_cp":     "Avg ctx. precision",
        # ── section headers ──────────────────────────────────────────────────
        "h_metric_guide":    "Metric Reference Guide",
        "h_observations":    "Key Observations & Recommendations",
        "h_best":            "Best Combination Overall",
        "h_top10":           "Top 10 Combinations",
        "h_heatmap":         "Embedding × Chunk Size Heatmap",
        "h_breakdown":       "Breakdown by Dimension",
        "h_format_cat":      "Document Format & Question Category",
        "h_charts":          "Avg Score by Dimension — Charts",
        "h_full_rankings":   "Full Rankings — All Combinations",
        "h_per_question":    "Per-Question Best Combination",
        # ── metric cards ─────────────────────────────────────────────────────
        "m_faith_name":   "Faithfulness",
        "m_faith_what":   "Measures whether the generated answer is <strong>factually grounded</strong> in the retrieved context — i.e., it does not hallucinate.",
        "m_faith_how":    "The judge LLM extracts all atomic claims from the answer, then checks each claim against the retrieved chunks.",
        "m_faith_formula":"supported_claims / total_claims",
        "m_faith_interp": (
            "<span class='tag tag-green'>≥ 0.80 Excellent</span> Answer stays within retrieved facts.<br>"
            "<span class='tag tag-yellow'>0.60–0.79 Good</span> Minor unsupported statements.<br>"
            "<span class='tag tag-orange'>0.40–0.59 Moderate</span> Some hallucinations present.<br>"
            "<span class='tag tag-red'>&lt; 0.40 Poor</span> Significant hallucination risk.<br>"
            "<strong>Affected by:</strong> LLM size, system prompt strictness, context clarity."
        ),
        "m_ar_name":      "Answer Relevancy",
        "m_ar_what":      "Measures how <strong>on-topic</strong> the answer is relative to the original question — not correctness, but relevance.",
        "m_ar_how":       "The judge LLM generates synthetic questions that the answer could respond to, then measures cosine similarity between those and the original question.",
        "m_ar_formula":   "avg cosine_sim(original_question, synthetic_questions)",
        "m_ar_interp":    (
            "<span class='tag tag-green'>≥ 0.80 Excellent</span> Highly focused, on-topic answer.<br>"
            "<span class='tag tag-yellow'>0.60–0.79 Good</span> Mostly relevant with minor drift.<br>"
            "<span class='tag tag-orange'>0.40–0.59 Moderate</span> Answer addresses different aspects.<br>"
            "<span class='tag tag-red'>&lt; 0.40 Poor</span> Answer is off-topic.<br>"
            "<strong>Affected by:</strong> LLM quality, prompt design, embedding model for judge."
        ),
        "m_cp_name":      "Context Precision",
        "m_cp_what":      "Measures the <strong>signal-to-noise ratio</strong> of retrieval — are the retrieved chunks actually useful for answering the question?",
        "m_cp_how":       "For each retrieved chunk, the judge LLM checks whether it is relevant to the question given the reference answer. Penalises irrelevant chunks that appear before relevant ones.",
        "m_cp_formula":   "weighted_precision@k (rank-aware)",
        "m_cp_interp":    (
            "<span class='tag tag-green'>≥ 0.80 Excellent</span> Most retrieved chunks are useful.<br>"
            "<span class='tag tag-yellow'>0.50–0.79 Good</span> Some noise but mostly relevant.<br>"
            "<span class='tag tag-orange'>0.30–0.49 Moderate</span> Many irrelevant chunks retrieved.<br>"
            "<span class='tag tag-red'>&lt; 0.30 Poor</span> Retrieval is mostly noise.<br>"
            "<strong>Affected by:</strong> Embedding model, chunk size, k (retrieved chunks), re-ranking."
        ),
        "m_cr_name":      "Context Recall",
        "m_cr_what":      "Measures <strong>completeness of retrieval</strong> — was all the information needed to answer the question retrieved?",
        "m_cr_how":       "Each sentence of the reference answer is checked against the retrieved context to see if it is covered.",
        "m_cr_formula":   "covered_sentences / total_sentences_in_reference",
        "m_cr_interp":    (
            "<span class='tag tag-green'>≥ 0.50 Good</span> Most needed information retrieved.<br>"
            "<span class='tag tag-yellow'>0.30–0.49 Acceptable</span> Key info retrieved, some gaps.<br>"
            "<span class='tag tag-red'>&lt; 0.30 Retrieval gap</span> Many needed passages missed.<br>"
            "<strong>Affected by:</strong> Embedding model, chunk size, k (increase to improve)."
        ),
        # ── top10 table ──────────────────────────────────────────────────────
        "top10_desc":     (
            "Sorted by average of all available metrics (Faithfulness, Answer Relevancy, "
            "Context Precision, Context Recall). Click column headers to sort. "
            "Color: <span class='badge badge-high'>≥0.70</span> "
            "<span class='badge badge-mid'>0.50–0.69</span> "
            "<span class='badge badge-low'>0.30–0.49</span> "
            "<span class='badge badge-poor'>&lt;0.30</span>"
        ),
        "col_rank":       "#",
        "col_llm":        "LLM",
        "col_embed":      "Embedding",
        "col_chunk":      "Chunk",
        "col_faith":      "Faithfulness ↕",
        "col_ar":         "Ans. Relevancy ↕",
        "col_cp":         "Ctx. Precision ↕",
        "col_cr":         "Ctx. Recall ↕",
        "col_avg":        "Avg Score ↕",
        "col_ret_ms":     "Retrieval ms ↕",
        "col_gen_ms":     "Generation ms ↕",
        "col_questions":  "Questions ↕",
        # ── heatmap ──────────────────────────────────────────────────────────
        "heatmap_desc":   (
            "Most actionable comparison: which <strong>embedding model + chunk size</strong> "
            "combination retrieves the most relevant context (CR) while answering on-topic (AR)? "
            "Each cell averages across all LLMs."
        ),
        "heatmap_legend": "AR = Answer Relevancy &nbsp;|&nbsp; CR = Context Recall",
        # ── breakdown ────────────────────────────────────────────────────────
        "breakdown_desc": "Average metrics for each level of each dimension across all questions.",
        "h_by_llm":       "By LLM Model",
        "h_by_chunk":     "By Chunk Configuration",
        "h_by_embed":     "By Embedding Model",
        "col_dims":       "Dims",
        "col_size":       "Size (chars)",
        "col_n":          "N",
        # ── format & category ────────────────────────────────────────────────
        "format_cat_desc": (
            "<strong>Format</strong> is how source documents were processed — markdown preserves "
            "structure (headings, lists, tables); plaintext strips formatting. "
            "<strong>Category</strong> is the knowledge-base topic of each question."
        ),
        "h_doc_format":   "Document Format Comparison",
        "h_cat":          "By Question Category",
        "format_note":    "Markdown typically scores higher. Structured headings help embeddings find precise passages.",
        "cat_note":       "Categories with higher recall tend to have more focused, self-contained documents.",
        # ── charts ───────────────────────────────────────────────────────────
        "chart_llm":      "By LLM",
        "chart_embed":    "By Embedding",
        "chart_chunk":    "By Chunk Size",
        # ── full rankings ────────────────────────────────────────────────────
        "filter_all_llm":   "All LLMs",
        "filter_all_embed": "All Embeddings",
        "filter_all_chunk": "All Chunks",
        # ── per-question ─────────────────────────────────────────────────────
        "per_q_desc":    "For each question, the combination with the highest average score.",
        "search_q":      "Search question text…",
        "filter_all_cat":"All categories",
        "col_question":  "Question",
        "col_category":  "Category",
        "col_best_combo":"Best Combination",
        # ── best card ────────────────────────────────────────────────────────
        "best_label":    "LLM × Embedding × Chunk Size",
        "best_avg_label":"Average (all metrics)",
        "best_ret_label":"Avg retrieval time",
        "best_gen_label":"Avg generation time",
        "best_q_label":  "Questions covered",
        # ── observations ─────────────────────────────────────────────────────
        "obs": {
            "cr_bottleneck": (
                "<strong>Context Recall is the main bottleneck</strong> — overall avg CR is "
                "<em>{cr:.4f}</em>, below the 0.50 target. The retriever consistently fails to "
                "surface all relevant information the answer needs. Optimise retrieval first."
            ),
            "best_embed": (
                "<strong>Best embedding for recall: {embed}</strong> (CR: {cr}, AR: {ar}) — "
                "its multilingual 1024-dim representation suits Spanish poverty-programme content well. "
                "<strong>Weakest: {worst_embed}</strong> (CR: {wcr})."
            ),
            "best_chunk": (
                "<strong>Best chunk size: {chunk} ({size} chars)</strong> (CR: {cr}) — "
                "medium chunks consistently outperform both small and large. "
                "Small chunks (512) split context too aggressively; large chunks (2048) add noise."
            ),
            "best_llm_ar": (
                "<strong>Most relevant answers: {llm_ar}</strong> (AR: {ar}) — "
                "generates the most on-topic responses."
            ),
            "best_llm_cr": (
                "<strong>Best context coverage: {llm_cr}</strong> (CR: {cr}) — "
                "best at incorporating retrieved context into answers."
            ),
            "best_llm_faith": (
                "<strong>Most faithful LLM: {llm_f}</strong> (Faithfulness: {f}) — "
                "stays closest to the retrieved facts; lowest hallucination risk."
            ),
            "recommended": (
                "<strong>Recommended combination: {combo}</strong> — "
                "Faithfulness: {faith}, AR: {ar}, CP: {cp}, CR: {cr}, Avg: {avg}."
            ),
            "format": (
                "<strong>Document format matters:</strong> Markdown chunks outperform plaintext "
                "(CR +{diff:.4f}). Structured headings and lists help the embedding model identify "
                "semantic boundaries and retrieve more targeted passages."
            ),
            "improve_cr": (
                "<strong>To improve CR further:</strong> increase k_retrieved (currently 8), add a "
                "re-ranking step, or experiment with hybrid search (dense + BM25). "
                "The gap between AR ({ar:.4f}) and CR ({cr:.4f}) indicates a retrieval problem, "
                "not an LLM problem."
            ),
        },
    },

    "es": {
        # ── page chrome ─────────────────────────────────────────────────────
        "title":          "Reporte de Evaluación RAG — Semáforo de Pobreza",
        "subtitle":       "LLM Juez: {judge_llm} &nbsp;|&nbsp; Embeddings Juez: bge-m3 &nbsp;|&nbsp; Generado: {generated_at}",
        "lang_label":     "ES",
        # ── info notice ──────────────────────────────────────────────────────
        "notice": (
            "<strong>ℹ {n_scored:,} ejecuciones evaluadas</strong> en {n_combos} combinaciones "
            "LLM × embedding × tamaño de fragmento y {n_q} preguntas. "
            "La puntuación promedio usa todas las métricas disponibles (Fidelidad, Relevancia de "
            "Respuesta, Precisión de Contexto, Cobertura de Contexto) con peso igual."
        ),
        # ── stat boxes ───────────────────────────────────────────────────────
        "lbl_runs":       "Ejecuciones evaluadas",
        "lbl_combos":     "Combinaciones",
        "lbl_questions":  "Preguntas",
        "lbl_best_avg":   "Mejor puntaje promedio",
        "lbl_avg_faith":  "Fidelidad promedio",
        "lbl_avg_ar":     "Relevancia resp. prom.",
        "lbl_avg_cr":     "Cobertura ctx. prom.",
        "lbl_avg_cp":     "Precisión ctx. prom.",
        # ── section headers ──────────────────────────────────────────────────
        "h_metric_guide":    "Guía de Referencia de Métricas",
        "h_observations":    "Observaciones Clave y Recomendaciones",
        "h_best":            "Mejor Combinación General",
        "h_top10":           "Top 10 Combinaciones",
        "h_heatmap":         "Mapa de Calor: Embedding × Tamaño de Fragmento",
        "h_breakdown":       "Desglose por Dimensión",
        "h_format_cat":      "Formato de Documento y Categoría de Pregunta",
        "h_charts":          "Puntaje Promedio por Dimensión — Gráficos",
        "h_full_rankings":   "Ranking Completo — Todas las Combinaciones",
        "h_per_question":    "Mejor Combinación por Pregunta",
        # ── metric cards ─────────────────────────────────────────────────────
        "m_faith_name":   "Fidelidad (Faithfulness)",
        "m_faith_what":   "Mide si la respuesta generada está <strong>fundamentada en el contexto recuperado</strong> — es decir, no alucina información.",
        "m_faith_how":    "El LLM juez extrae todas las afirmaciones atómicas de la respuesta y verifica si cada una está respaldada por los fragmentos recuperados.",
        "m_faith_formula":"afirmaciones_respaldadas / total_afirmaciones",
        "m_faith_interp": (
            "<span class='tag tag-green'>≥ 0.80 Excelente</span> La respuesta se mantiene dentro de los hechos recuperados.<br>"
            "<span class='tag tag-yellow'>0.60–0.79 Bueno</span> Afirmaciones menores sin respaldo.<br>"
            "<span class='tag tag-orange'>0.40–0.59 Moderado</span> Algunas alucinaciones presentes.<br>"
            "<span class='tag tag-red'>&lt; 0.40 Deficiente</span> Riesgo significativo de alucinación.<br>"
            "<strong>Factores que influyen:</strong> tamaño del LLM, instrucciones del prompt, claridad del contexto."
        ),
        "m_ar_name":      "Relevancia de Respuesta (Answer Relevancy)",
        "m_ar_what":      "Mide qué tan <strong>pertinente es la respuesta</strong> a la pregunta original — no su corrección, sino su relevancia temática.",
        "m_ar_how":       "El LLM juez genera preguntas sintéticas a partir de la respuesta y mide la similitud coseno entre éstas y la pregunta original.",
        "m_ar_formula":   "promedio coseno_sim(pregunta_original, preguntas_sintéticas)",
        "m_ar_interp":    (
            "<span class='tag tag-green'>≥ 0.80 Excelente</span> Respuesta muy enfocada y pertinente.<br>"
            "<span class='tag tag-yellow'>0.60–0.79 Bueno</span> Mayormente relevante con pequeñas desviaciones.<br>"
            "<span class='tag tag-orange'>0.40–0.59 Moderado</span> La respuesta aborda aspectos diferentes.<br>"
            "<span class='tag tag-red'>&lt; 0.40 Deficiente</span> La respuesta no es pertinente.<br>"
            "<strong>Factores que influyen:</strong> calidad del LLM, diseño del prompt, modelo de embedding del juez."
        ),
        "m_cp_name":      "Precisión de Contexto (Context Precision)",
        "m_cp_what":      "Mide la <strong>relación señal-ruido de la recuperación</strong> — ¿los fragmentos recuperados son realmente útiles para responder la pregunta?",
        "m_cp_how":       "Para cada fragmento recuperado, el juez verifica si es relevante para la pregunta dada la respuesta de referencia. Penaliza fragmentos irrelevantes que aparecen antes que los relevantes.",
        "m_cp_formula":   "precisión_ponderada@k (sensible al ranking)",
        "m_cp_interp":    (
            "<span class='tag tag-green'>≥ 0.80 Excelente</span> La mayoría de fragmentos recuperados son útiles.<br>"
            "<span class='tag tag-yellow'>0.50–0.79 Bueno</span> Algo de ruido pero mayormente relevante.<br>"
            "<span class='tag tag-orange'>0.30–0.49 Moderado</span> Muchos fragmentos irrelevantes recuperados.<br>"
            "<span class='tag tag-red'>&lt; 0.30 Deficiente</span> La recuperación es principalmente ruido.<br>"
            "<strong>Factores que influyen:</strong> modelo de embedding, tamaño de fragmento, k, re-ranking."
        ),
        "m_cr_name":      "Cobertura de Contexto (Context Recall)",
        "m_cr_what":      "Mide la <strong>completitud de la recuperación</strong> — ¿fue recuperada toda la información necesaria para responder la pregunta?",
        "m_cr_how":       "Cada oración de la respuesta de referencia se verifica contra el contexto recuperado para ver si está cubierta.",
        "m_cr_formula":   "oraciones_cubiertas / total_oraciones_en_referencia",
        "m_cr_interp":    (
            "<span class='tag tag-green'>≥ 0.50 Bueno</span> La mayoría de la información necesaria fue recuperada.<br>"
            "<span class='tag tag-yellow'>0.30–0.49 Aceptable</span> Información clave recuperada, algunas brechas.<br>"
            "<span class='tag tag-red'>&lt; 0.30 Brecha de recuperación</span> Muchos pasajes relevantes no fueron recuperados.<br>"
            "<strong>Factores que influyen:</strong> modelo de embedding, tamaño de fragmento, k (aumentar para mejorar)."
        ),
        # ── top10 table ──────────────────────────────────────────────────────
        "top10_desc":     (
            "Ordenado por el promedio de todas las métricas disponibles (Fidelidad, Relevancia, "
            "Precisión de Contexto, Cobertura de Contexto). Clic en encabezados para ordenar. "
            "Color: <span class='badge badge-high'>≥0.70</span> "
            "<span class='badge badge-mid'>0.50–0.69</span> "
            "<span class='badge badge-low'>0.30–0.49</span> "
            "<span class='badge badge-poor'>&lt;0.30</span>"
        ),
        "col_rank":       "#",
        "col_llm":        "LLM",
        "col_embed":      "Embedding",
        "col_chunk":      "Fragmento",
        "col_faith":      "Fidelidad ↕",
        "col_ar":         "Rel. Respuesta ↕",
        "col_cp":         "Prec. Contexto ↕",
        "col_cr":         "Cob. Contexto ↕",
        "col_avg":        "Prom. Score ↕",
        "col_ret_ms":     "Recuperación ms ↕",
        "col_gen_ms":     "Generación ms ↕",
        "col_questions":  "Preguntas ↕",
        # ── heatmap ──────────────────────────────────────────────────────────
        "heatmap_desc":   (
            "Comparación más accionable: ¿qué combinación de <strong>modelo de embedding + "
            "tamaño de fragmento</strong> recupera el contexto más relevante (CR) mientras "
            "genera respuestas pertinentes (AR)? Cada celda promedia sobre todos los LLMs."
        ),
        "heatmap_legend": "AR = Relevancia de Respuesta &nbsp;|&nbsp; CR = Cobertura de Contexto",
        # ── breakdown ────────────────────────────────────────────────────────
        "breakdown_desc": "Métricas promedio para cada nivel de cada dimensión en todas las preguntas.",
        "h_by_llm":       "Por Modelo LLM",
        "h_by_chunk":     "Por Configuración de Fragmento",
        "h_by_embed":     "Por Modelo de Embedding",
        "col_dims":       "Dims",
        "col_size":       "Tamaño (chars)",
        "col_n":          "N",
        # ── format & category ────────────────────────────────────────────────
        "format_cat_desc": (
            "<strong>Formato</strong> es cómo se procesaron los documentos fuente — markdown "
            "preserva la estructura (encabezados, listas, tablas); texto plano elimina el formato. "
            "<strong>Categoría</strong> es el tema de cada pregunta en la base de conocimiento."
        ),
        "h_doc_format":   "Comparación de Formato de Documento",
        "h_cat":          "Por Categoría de Pregunta",
        "format_note":    "Markdown típicamente obtiene puntajes más altos. Los encabezados estructurados ayudan a los embeddings a encontrar pasajes precisos.",
        "cat_note":       "Las categorías con mayor cobertura tienden a tener documentos más enfocados y autónomos.",
        # ── charts ───────────────────────────────────────────────────────────
        "chart_llm":      "Por LLM",
        "chart_embed":    "Por Embedding",
        "chart_chunk":    "Por Tamaño de Fragmento",
        # ── full rankings ────────────────────────────────────────────────────
        "filter_all_llm":   "Todos los LLMs",
        "filter_all_embed": "Todos los Embeddings",
        "filter_all_chunk": "Todos los Fragmentos",
        # ── per-question ─────────────────────────────────────────────────────
        "per_q_desc":    "Para cada pregunta, la combinación con el mayor puntaje promedio.",
        "search_q":      "Buscar texto de pregunta…",
        "filter_all_cat":"Todas las categorías",
        "col_question":  "Pregunta",
        "col_category":  "Categoría",
        "col_best_combo":"Mejor Combinación",
        # ── best card ────────────────────────────────────────────────────────
        "best_label":    "LLM × Embedding × Tamaño de Fragmento",
        "best_avg_label":"Promedio (todas las métricas)",
        "best_ret_label":"Tiempo de recuperación prom.",
        "best_gen_label":"Tiempo de generación prom.",
        "best_q_label":  "Preguntas cubiertas",
        # ── observations ─────────────────────────────────────────────────────
        "obs": {
            "cr_bottleneck": (
                "<strong>La Cobertura de Contexto es el principal cuello de botella</strong> — "
                "el promedio global de CR es <em>{cr:.4f}</em>, por debajo del objetivo de 0.50. "
                "El recuperador no logra consistentemente traer toda la información relevante. "
                "Priorizar la optimización de la recuperación."
            ),
            "best_embed": (
                "<strong>Mejor embedding para cobertura: {embed}</strong> (CR: {cr}, AR: {ar}) — "
                "su representación multilingüe de 1024 dimensiones se adapta bien al contenido "
                "en español sobre programas de reducción de pobreza. "
                "<strong>Más débil: {worst_embed}</strong> (CR: {wcr})."
            ),
            "best_chunk": (
                "<strong>Mejor tamaño de fragmento: {chunk} ({size} chars)</strong> (CR: {cr}) — "
                "los fragmentos medianos superan consistentemente a los pequeños y grandes. "
                "Los fragmentos pequeños (512) dividen demasiado el contexto; los grandes (2048) agregan ruido."
            ),
            "best_llm_ar": (
                "<strong>Respuestas más relevantes: {llm_ar}</strong> (AR: {ar}) — "
                "genera las respuestas más pertinentes a la pregunta."
            ),
            "best_llm_cr": (
                "<strong>Mejor cobertura de contexto: {llm_cr}</strong> (CR: {cr}) — "
                "el más efectivo para incorporar el contexto recuperado en sus respuestas."
            ),
            "best_llm_faith": (
                "<strong>LLM más fiel: {llm_f}</strong> (Fidelidad: {f}) — "
                "se mantiene más cerca de los hechos recuperados; menor riesgo de alucinación."
            ),
            "recommended": (
                "<strong>Combinación recomendada: {combo}</strong> — "
                "Fidelidad: {faith}, Rel. Resp.: {ar}, Prec. Ctx.: {cp}, Cob. Ctx.: {cr}, Prom.: {avg}."
            ),
            "format": (
                "<strong>El formato del documento importa:</strong> Los fragmentos en Markdown "
                "superan al texto plano (CR +{diff:.4f}). Los encabezados y listas estructurados "
                "ayudan al modelo de embedding a identificar límites semánticos y recuperar "
                "pasajes más específicos."
            ),
            "improve_cr": (
                "<strong>Para mejorar CR:</strong> aumentar k_retrieved (actualmente 8), añadir "
                "un paso de re-ranking, o experimentar con búsqueda híbrida (densa + BM25). "
                "La brecha entre AR ({ar:.4f}) y CR ({cr:.4f}) indica un problema de recuperación, "
                "no del LLM."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "db_rag"),
        user=os.getenv("DB_USER", "sgadmin"),
        password=os.getenv("DB_PASSWORD", "sg4dm1n!"),
    )


def fetch_data(conn, questions_limit=None) -> list[dict]:
    q_filter = f"AND kb.id <= {questions_limit}" if questions_limit else ""
    sql = f"""
        SELECT
            lm.model_name          AS llm,
            em.model_name          AS embed,
            cc.name                AS chunk,
            kb.question,
            kb.category,
            es.faithfulness,
            es.answer_relevancy,
            es.context_precision,
            es.context_recall,
            ROUND(
                (
                    (COALESCE(es.faithfulness,0) + COALESCE(es.answer_relevancy,0)
                     + COALESCE(es.context_precision,0) + COALESCE(es.context_recall,0))
                    / NULLIF(
                        (CASE WHEN es.faithfulness        IS NOT NULL THEN 1 ELSE 0 END
                         + CASE WHEN es.answer_relevancy  IS NOT NULL THEN 1 ELSE 0 END
                         + CASE WHEN es.context_precision IS NOT NULL THEN 1 ELSE 0 END
                         + CASE WHEN es.context_recall    IS NOT NULL THEN 1 ELSE 0 END),
                        0
                    )
                )::numeric, 4
            )                      AS avg_score,
            er.retrieval_time_ms,
            er.generation_time_ms
        FROM eval_runs er
        JOIN llm_models       lm ON er.llm_model_id       = lm.id
        JOIN embedding_models em ON er.embedding_model_id = em.id
        JOIN chunk_configs    cc ON er.chunk_config_id    = cc.id
        JOIN knowledge_base   kb ON er.knowledge_base_id  = kb.id
        JOIN eval_scores      es ON er.id                 = es.eval_run_id
        WHERE er.status = 'success'
          AND es.status = 'success'
          {q_filter}
        ORDER BY avg_score DESC NULLS LAST
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_score_summary(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT status, COUNT(*) FROM eval_scores GROUP BY status")
        return {row[0]: row[1] for row in cur.fetchall()}


def fetch_extra_breakdowns(conn, questions_limit=None) -> dict:
    qf      = f"AND kb.id <= {questions_limit}" if questions_limit else ""
    qf_nojoin = (
        f"AND er.knowledge_base_id IN (SELECT id FROM knowledge_base ORDER BY id LIMIT {questions_limit})"
        if questions_limit else ""
    )

    def run(sql):
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _metrics_cols(alias="es"):
        return f"""
            ROUND(AVG({alias}.answer_relevancy)::numeric, 4)  AS avg_ar,
            ROUND(AVG({alias}.context_recall)::numeric, 4)    AS avg_cr,
            ROUND(AVG({alias}.faithfulness)::numeric, 4)      AS avg_faith,
            ROUND(AVG({alias}.context_precision)::numeric, 4) AS avg_cp
        """

    embed_chunk = run(f"""
        SELECT em.model_name AS embed, cc.name AS chunk,
               cc.chunk_size, cc.overlap, COUNT(*) AS n, {_metrics_cols()}
        FROM eval_scores es
        JOIN eval_runs er        ON es.eval_run_id       = er.id
        JOIN embedding_models em ON er.embedding_model_id = em.id
        JOIN chunk_configs cc    ON er.chunk_config_id    = cc.id
        WHERE es.status = 'success' {qf_nojoin}
        GROUP BY em.model_name, cc.name, cc.chunk_size, cc.overlap
        ORDER BY avg_cr DESC NULLS LAST
    """)

    by_llm = run(f"""
        SELECT lm.model_name AS llm, COUNT(*) AS n, {_metrics_cols()}
        FROM eval_scores es
        JOIN eval_runs er  ON es.eval_run_id  = er.id
        JOIN llm_models lm ON er.llm_model_id = lm.id
        WHERE es.status = 'success' {qf_nojoin}
        GROUP BY lm.model_name
        ORDER BY avg_cr DESC NULLS LAST
    """)

    by_embed = run(f"""
        SELECT em.model_name AS embed, em.dimensions, COUNT(*) AS n, {_metrics_cols()}
        FROM eval_scores es
        JOIN eval_runs er        ON es.eval_run_id       = er.id
        JOIN embedding_models em ON er.embedding_model_id = em.id
        WHERE es.status = 'success' {qf_nojoin}
        GROUP BY em.model_name, em.dimensions
        ORDER BY avg_cr DESC NULLS LAST
    """)

    by_chunk = run(f"""
        SELECT cc.name AS chunk, cc.chunk_size, cc.overlap, COUNT(*) AS n, {_metrics_cols()}
        FROM eval_scores es
        JOIN eval_runs er     ON es.eval_run_id    = er.id
        JOIN chunk_configs cc ON er.chunk_config_id = cc.id
        WHERE es.status = 'success' {qf_nojoin}
        GROUP BY cc.name, cc.chunk_size, cc.overlap
        ORDER BY avg_cr DESC NULLS LAST
    """)

    by_format = run(f"""
        SELECT
            COALESCE(
                (SELECT c.format FROM chunks c
                 JOIN chunk_configs cc2 ON c.chunk_config_id = cc2.id
                 WHERE cc2.id = er.chunk_config_id LIMIT 1),
                'unknown'
            ) AS format,
            COUNT(*) AS n, {_metrics_cols()}
        FROM eval_scores es
        JOIN eval_runs er ON es.eval_run_id = er.id
        WHERE es.status = 'success' {qf_nojoin}
        GROUP BY format
        ORDER BY avg_cr DESC NULLS LAST
    """)

    by_category = run(f"""
        SELECT kb.category, COUNT(*) AS n, {_metrics_cols()}
        FROM eval_scores es
        JOIN eval_runs er      ON es.eval_run_id      = er.id
        JOIN knowledge_base kb ON er.knowledge_base_id = kb.id
        WHERE es.status = 'success' {qf}
        GROUP BY kb.category
        ORDER BY avg_cr DESC NULLS LAST
    """)

    return {
        "embed_chunk": embed_chunk,
        "by_llm":      by_llm,
        "by_embed":    by_embed,
        "by_chunk":    by_chunk,
        "by_format":   by_format,
        "by_category": by_category,
    }


# ---------------------------------------------------------------------------
# Data aggregation helpers
# ---------------------------------------------------------------------------

def safe_float(v, digits=4):
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, digits)
    except (TypeError, ValueError):
        return None


def combo_key(row):
    return f"{row['llm']} | {row['embed']} | {row['chunk']}"


def aggregate_combos(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall", "avg_score")
    sums = defaultdict(lambda: {m: 0.0 for m in METRICS} | {
        "retrieval_time_ms": 0.0, "generation_time_ms": 0.0, "n": 0,
        "llm": "", "embed": "", "chunk": "",
    })
    counts = defaultdict(lambda: {m: 0 for m in METRICS})

    for r in rows:
        k = combo_key(r)
        d = sums[k]; d["llm"] = r["llm"]; d["embed"] = r["embed"]; d["chunk"] = r["chunk"]
        for m in METRICS:
            if r[m] is not None:
                d[m] += float(r[m]); counts[k][m] += 1
        d["retrieval_time_ms"]  += float(r["retrieval_time_ms"]  or 0)
        d["generation_time_ms"] += float(r["generation_time_ms"] or 0)
        d["n"] += 1

    result = []
    for k, d in sums.items():
        n = d["n"] or 1
        entry = {
            "combo": k, "llm": d["llm"], "embed": d["embed"], "chunk": d["chunk"],
            "retrieval_time_ms":  round(d["retrieval_time_ms"]  / n, 1),
            "generation_time_ms": round(d["generation_time_ms"] / n, 1),
            "n_questions": n,
        }
        for m in METRICS:
            cnt = counts[k][m] or 1
            entry[m] = round(d[m] / cnt, 4) if counts[k][m] else None
        result.append(entry)

    result.sort(key=lambda x: x["avg_score"] or 0, reverse=True)
    return result


def per_question_best(rows: list[dict]) -> list[dict]:
    from collections import defaultdict
    by_q: dict = defaultdict(list)
    for r in rows:
        by_q[r["question"]].append(r)

    result = []
    for question, qrows in by_q.items():
        best = max(qrows, key=lambda r: float(r["avg_score"]) if r["avg_score"] else 0.0)
        result.append({
            "question":          question,
            "category":          best["category"] or "",
            "best_combo":        combo_key(best),
            "avg_score":         safe_float(best["avg_score"]),
            "faithfulness":      safe_float(best["faithfulness"]),
            "answer_relevancy":  safe_float(best["answer_relevancy"]),
            "context_precision": safe_float(best["context_precision"]),
            "context_recall":    safe_float(best["context_recall"]),
        })
    result.sort(key=lambda x: x["avg_score"] or 0.0, reverse=True)
    return result


def avg_by_dim(combos: list[dict], dim: str) -> list[dict]:
    from collections import defaultdict
    sums: dict = defaultdict(list)
    for c in combos:
        if c["avg_score"] is not None:
            sums[c[dim]].append(c["avg_score"])
    return sorted(
        [{"label": k, "score": round(sum(v) / len(v), 4)} for k, v in sums.items()],
        key=lambda x: x["score"], reverse=True,
    )


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{float(v):.4f}"


def pct(v) -> str:
    if v is None:
        return "—"
    return f"{float(v)*100:.1f}%"


def _color(v) -> str:
    if v is None:
        return "#f8f9fa"
    f = float(v)
    if f >= 0.7: return "#d4edda"
    if f >= 0.5: return "#fff3cd"
    if f >= 0.3: return "#fde8c8"
    return "#f8d7da"


def _heat(v, lo=0.0, hi=0.5) -> str:
    if v is None:
        return "#eeeeee"
    f = float(v)
    t = max(0.0, min(1.0, (f - lo) / (hi - lo)))
    if t < 0.5:
        r, g = 220, int(180 * t * 2)
    else:
        r, g = int(220 * (1 - (t - 0.5) * 2)), 180
    return f"rgb({r},{g},80)"


def _embed_chunk_heatmap_html(embed_chunk: list[dict], T: dict) -> str:
    embeds = sorted(set(r["embed"] for r in embed_chunk))
    chunks = ["small", "medium", "large"]
    chunk_labels = {
        "small":  "Small\n(512/64)",
        "medium": "Medium\n(1024/128)",
        "large":  "Large\n(2048/256)",
    }
    idx = {(r["embed"], r["chunk"]): r for r in embed_chunk}

    rows_html = ""
    for embed in embeds:
        rows_html += f"<tr><td style='font-weight:600;white-space:nowrap'>{embed}</td>"
        for chunk in chunks:
            r = idx.get((embed, chunk))
            if r:
                ar = float(r["avg_ar"]) if r["avg_ar"] else 0
                cr = float(r["avg_cr"]) if r["avg_cr"] else 0
                rows_html += (
                    f"<td style='text-align:center;padding:.5rem'>"
                    f"<span style='display:block;background:{_heat(ar,0.5,0.85)};border-radius:3px;"
                    f"padding:.2rem .4rem;margin-bottom:.2rem;font-size:.82rem'>AR: {ar:.4f}</span>"
                    f"<span style='display:block;background:{_heat(cr,0.05,0.50)};border-radius:3px;"
                    f"padding:.2rem .4rem;font-size:.82rem'>CR: {cr:.4f}</span>"
                    f"<span style='font-size:.7rem;color:#888'>n={r['n']}</span></td>"
                )
            else:
                rows_html += "<td style='text-align:center;color:#bbb'>—</td>"
        rows_html += "</tr>"

    header = "".join(
        f"<th style='text-align:center;background:#1a3c5e;color:#fff;padding:.5rem .75rem'>"
        f"{chunk_labels[c]}</th>"
        for c in chunks
    )
    return f"""
    <div style="overflow-x:auto">
    <table style="border-collapse:collapse;width:100%;font-size:.85rem">
      <thead><tr>
        <th style="background:#1a3c5e;color:#fff;padding:.5rem .75rem;text-align:left">Embedding</th>
        {header}
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    <p style="font-size:.78rem;color:#777;margin-top:.5rem">
      {T['heatmap_legend']} &nbsp;|&nbsp;
      Color: <span style="background:#dc4040;color:#fff;padding:.1rem .3rem;border-radius:2px">low</span>
      → <span style="background:#e8b450;color:#fff;padding:.1rem .3rem;border-radius:2px">mid</span>
      → <span style="background:#28a745;color:#fff;padding:.1rem .3rem;border-radius:2px">high</span>
    </p>
    """


def _dim_table_html(rows: list[dict], label_col: str, label_title: str,
                    extra_col: str = None, extra_title: str = None,
                    T: dict = None) -> str:
    T = T or {}
    col_n     = T.get("col_n", "N")
    col_faith = T.get("col_faith", "Faith.")
    col_ar    = T.get("col_ar",    "Ans. Rel.")
    col_cp    = T.get("col_cp",    "Ctx. Prec.")
    col_cr    = T.get("col_cr",    "Ctx. Recall")

    # Shorten for narrow tables
    col_faith = col_faith.replace(" ↕","")
    col_ar    = col_ar.replace(" ↕","")
    col_cp    = col_cp.replace(" ↕","")
    col_cr    = col_cr.replace(" ↕","")

    header = f"<th>{label_title}</th>"
    if extra_col:
        header += f"<th>{extra_title}</th>"
    header += f"<th>{col_n}</th><th>{col_faith}</th><th>{col_ar}</th><th>{col_cp}</th><th>{col_cr}</th>"

    body = ""
    for r in rows:
        extra_td = f"<td>{r.get(extra_col, '')}</td>" if extra_col else ""
        body += (
            f"<tr><td style='font-weight:600'>{r[label_col]}</td>{extra_td}"
            f"<td style='text-align:center'>{r['n']}</td>"
            f"<td style='text-align:center;background:{_color(r.get('avg_faith'))}'>{fmt(r.get('avg_faith'))}</td>"
            f"<td style='text-align:center;background:{_color(r.get('avg_ar'))}'>{fmt(r.get('avg_ar'))}</td>"
            f"<td style='text-align:center;background:{_color(r.get('avg_cp'))}'>{fmt(r.get('avg_cp'))}</td>"
            f"<td style='text-align:center;background:{_color(r.get('avg_cr'))}'>{fmt(r.get('avg_cr'))}</td>"
            f"</tr>"
        )
    return f"""
    <table style="border-collapse:collapse;width:100%;font-size:.85rem">
      <thead><tr style="background:#1a3c5e;color:#fff">
        {header}
      </tr></thead>
      <tbody style="border:1px solid #ddd">{body}</tbody>
    </table>
    """


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

def build_html(rows, combos, per_q, score_summary, extra, judge_llm,
               generated_at, out_path, lang="en"):
    T          = TRANSLATIONS[lang]
    n_scored   = score_summary.get("success", 0)
    best       = combos[0] if combos else {}
    top10      = combos[:10]

    by_llm_dim   = avg_by_dim(combos, "llm")
    by_embed_dim = avg_by_dim(combos, "embed")
    by_chunk_dim = avg_by_dim(combos, "chunk")

    def chart_data(items):
        return json.dumps({"labels": [x["label"] for x in items],
                           "scores": [x["score"] for x in items]})

    radar_labels = [T["m_faith_name"].split(" (")[0],
                    T["m_ar_name"].split(" (")[0],
                    T["m_cp_name"].split(" (")[0],
                    T["m_cr_name"].split(" (")[0]]
    radar_values = [best.get("faithfulness") or 0, best.get("answer_relevancy") or 0,
                    best.get("context_precision") or 0, best.get("context_recall") or 0]

    all_combos_js = json.dumps(combos)
    per_q_js      = json.dumps(per_q)

    # ── Top-10 rows ─────────────────────────────────────────────────────────
    top10_rows_html = ""
    for i, c in enumerate(top10, 1):
        top10_rows_html += f"""
        <tr>
          <td>{i}</td><td>{c['llm']}</td><td>{c['embed']}</td><td>{c['chunk']}</td>
          <td style="background:{_color(c.get('faithfulness'))};text-align:center">{fmt(c.get('faithfulness'))}</td>
          <td style="background:{_color(c.get('answer_relevancy'))};text-align:center">{fmt(c.get('answer_relevancy'))}</td>
          <td style="background:{_color(c.get('context_precision'))};text-align:center">{fmt(c.get('context_precision'))}</td>
          <td style="background:{_color(c.get('context_recall'))};text-align:center">{fmt(c.get('context_recall'))}</td>
          <td style="background:{_color(c.get('avg_score'))};font-weight:bold;text-align:center">{fmt(c.get('avg_score'))}</td>
          <td style="text-align:center">{int(c['retrieval_time_ms'])} ms</td>
          <td style="text-align:center">{int(c['generation_time_ms'])} ms</td>
          <td style="text-align:center">{c['n_questions']}</td>
        </tr>"""

    by_llm_js   = chart_data(by_llm_dim)
    by_embed_js = chart_data(by_embed_dim)
    by_chunk_js = chart_data(by_chunk_dim)

    # ── Sub-tables ──────────────────────────────────────────────────────────
    fmt_rows    = extra.get("by_format", [])
    cat_rows    = extra.get("by_category", [])
    heatmap_html = _embed_chunk_heatmap_html(extra.get("embed_chunk", []), T)
    llm_table   = _dim_table_html(extra["by_llm"],   "llm",   T["col_llm"],   T=T)
    embed_table = _dim_table_html(extra["by_embed"],  "embed", T["col_embed"], "dimensions", T["col_dims"], T=T)
    chunk_table = _dim_table_html(extra["by_chunk"],  "chunk", T["col_chunk"], "chunk_size",  T["col_size"], T=T)
    fmt_table   = _dim_table_html(fmt_rows,  "format",   T.get("h_doc_format","Format"), T=T)
    cat_table   = _dim_table_html(cat_rows,  "category", T["col_category"], T=T)

    # ── Observations variables ───────────────────────────────────────────────
    by_llm_extra   = extra["by_llm"]
    by_embed_extra = extra["by_embed"]
    by_chunk_extra = extra["by_chunk"]
    overall_avg_ar   = round(sum(float(r["avg_ar"] or 0) for r in by_llm_extra) / max(len(by_llm_extra),1), 4)
    overall_avg_cr   = round(sum(float(r["avg_cr"] or 0) for r in by_llm_extra) / max(len(by_llm_extra),1), 4)
    overall_avg_faith= round(sum(float(r["avg_faith"] or 0) for r in by_llm_extra) / max(len(by_llm_extra),1), 4)
    overall_avg_cp   = round(sum(float(r["avg_cp"] or 0) for r in by_llm_extra) / max(len(by_llm_extra),1), 4)

    best_embed_cr  = max(by_embed_extra, key=lambda r: float(r["avg_cr"] or 0))
    worst_embed_cr = min(by_embed_extra, key=lambda r: float(r["avg_cr"] or 0))
    best_chunk_r   = max(by_chunk_extra, key=lambda r: float(r["avg_cr"] or 0))
    best_llm_ar    = max(by_llm_extra, key=lambda r: float(r["avg_ar"] or 0))
    best_llm_cr    = max(by_llm_extra, key=lambda r: float(r["avg_cr"] or 0))
    best_llm_faith = max(by_llm_extra, key=lambda r: float(r["avg_faith"] or 0))

    fmt_rows_sorted = sorted(fmt_rows, key=lambda r: float(r["avg_cr"] or 0), reverse=True)
    fmt_diff_cr = (
        float(fmt_rows_sorted[0]["avg_cr"]) - float(fmt_rows_sorted[1]["avg_cr"])
        if len(fmt_rows_sorted) >= 2 else 0
    )

    obs = T["obs"]
    obs_html = f"""
    <li>{obs['cr_bottleneck'].format(cr=overall_avg_cr)}</li>
    <li>{obs['best_embed'].format(
        embed=best_embed_cr['embed'], cr=fmt(best_embed_cr['avg_cr']), ar=fmt(best_embed_cr['avg_ar']),
        worst_embed=worst_embed_cr['embed'], wcr=fmt(worst_embed_cr['avg_cr'])
    )}</li>
    <li>{obs['best_chunk'].format(
        chunk=best_chunk_r['chunk'], size=best_chunk_r.get('chunk_size','—'), cr=fmt(best_chunk_r['avg_cr'])
    )}</li>
    <li>{obs['best_llm_ar'].format(llm_ar=best_llm_ar['llm'], ar=fmt(best_llm_ar['avg_ar']))}</li>
    <li>{obs['best_llm_cr'].format(llm_cr=best_llm_cr['llm'], cr=fmt(best_llm_cr['avg_cr']))}</li>
    <li>{obs['best_llm_faith'].format(llm_f=best_llm_faith['llm'], f=fmt(best_llm_faith['avg_faith']))}</li>
    <li>{obs['recommended'].format(
        combo=best.get('combo','—'),
        faith=fmt(best.get('faithfulness')), ar=fmt(best.get('answer_relevancy')),
        cp=fmt(best.get('context_precision')), cr=fmt(best.get('context_recall')),
        avg=fmt(best.get('avg_score'))
    )}</li>
    <li>{obs['format'].format(diff=fmt_diff_cr)}</li>
    <li>{obs['improve_cr'].format(ar=overall_avg_ar, cr=overall_avg_cr)}</li>
    """

    notice_html = T["notice"].format(
        n_scored=n_scored, n_combos=len(combos), n_q=len(per_q)
    )
    subtitle_html = T["subtitle"].format(judge_llm=judge_llm, generated_at=generated_at)

    html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{T['title']}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f5f7fa;color:#222;line-height:1.5}}
h1{{font-size:1.8rem}}
h2{{font-size:1.25rem;margin:1.5rem 0 .6rem;color:#1a3c5e;border-bottom:2px solid #e0e7ef;padding-bottom:.3rem}}
h3{{font-size:1rem;margin:.75rem 0 .4rem;color:#1a3c5e}}
.header{{background:linear-gradient(135deg,#1a3c5e,#2a5f9e);color:#fff;padding:1.5rem 2rem}}
.header p{{margin-top:.4rem;font-size:.9rem;opacity:.85}}
.content{{max-width:1500px;margin:0 auto;padding:1.5rem}}
.card{{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.1);padding:1.25rem 1.5rem;margin-bottom:1.5rem}}
.card-note{{background:#fffbea;border-left:4px solid #f0a500;border-radius:0 8px 8px 0}}
.card-obs{{background:#f0f7ff;border-left:4px solid #1a3c5e;border-radius:0 8px 8px 0}}
.stats-row{{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem}}
.stat-box{{flex:1;min-width:120px;background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.1);padding:1rem;text-align:center}}
.stat-box .num{{font-size:1.8rem;font-weight:700;color:#1a3c5e}}
.stat-box .lbl{{font-size:.75rem;color:#666;margin-top:.2rem}}
.best-card{{display:flex;gap:1.5rem;flex-wrap:wrap;align-items:flex-start}}
.best-meta{{flex:1;min-width:220px}}
.best-meta .combo-tag{{font-size:1rem;font-weight:600;color:#1a3c5e;background:#e8f0fe;padding:.5rem .75rem;border-radius:6px;margin:.5rem 0}}
.best-meta table{{border-collapse:collapse;width:100%;margin-top:.5rem}}
.best-meta td{{padding:.3rem .5rem;font-size:.9rem;border-bottom:1px solid #f0f0f0}}
.best-meta td:first-child{{color:#555;width:55%}}
.best-meta td:last-child{{font-weight:700;text-align:right}}
.radar-wrap{{flex:0 0 300px}}
.charts-row{{display:flex;gap:1rem;flex-wrap:wrap}}
.chart-box{{flex:1;min-width:240px}}
.two-col{{display:flex;gap:1.5rem;flex-wrap:wrap}}
.two-col>*{{flex:1;min-width:280px}}
table.data{{width:100%;border-collapse:collapse;font-size:.85rem}}
table.data th{{background:#1a3c5e;color:#fff;padding:.5rem .6rem;text-align:left;cursor:pointer;user-select:none;white-space:nowrap}}
table.data th:hover{{background:#2a5c8e}}
table.data td{{padding:.4rem .6rem;border-bottom:1px solid #eee}}
table.data tr:hover td{{background:#f0f4ff}}
.filter-row{{display:flex;gap:.75rem;flex-wrap:wrap;margin-bottom:.75rem}}
.filter-row select,.filter-row input{{padding:.4rem .6rem;border:1px solid #ccc;border-radius:5px;font-size:.85rem;background:#fff}}
.badge{{display:inline-block;padding:.15rem .45rem;border-radius:4px;font-size:.75rem;font-weight:600}}
.badge-high{{background:#d4edda;color:#155724}}
.badge-mid{{background:#fff3cd;color:#856404}}
.badge-low{{background:#fde8c8;color:#7a4100}}
.badge-poor{{background:#f8d7da;color:#721c24}}
.obs-list{{list-style:none;padding:0;margin:.5rem 0}}
.obs-list li{{padding:.4rem 0 .4rem 1.4rem;position:relative;border-bottom:1px solid #e8f0fe;font-size:.92rem}}
.obs-list li:last-child{{border-bottom:none}}
.obs-list li::before{{content:"▸";position:absolute;left:0;color:#1a3c5e;font-weight:700}}
.obs-list li strong{{color:#1a3c5e}}
.metric-def{{display:flex;gap:1rem;flex-wrap:wrap;margin-top:.75rem}}
.metric-card{{flex:1;min-width:220px;background:#f8fbff;border:1px solid #d0e4f7;border-radius:7px;padding:1rem}}
.metric-card .metric-name{{font-weight:700;color:#1a3c5e;font-size:1rem;margin-bottom:.3rem}}
.metric-card .metric-what{{font-size:.84rem;color:#333;margin-bottom:.4rem}}
.metric-card .metric-how{{font-size:.78rem;color:#555;margin-bottom:.4rem}}
.metric-card .metric-formula{{font-size:.78rem;color:#555;background:#e8f0fe;padding:.3rem .5rem;border-radius:4px;margin:.4rem 0;font-family:monospace}}
.metric-card .metric-interp{{font-size:.80rem;color:#444;line-height:1.6}}
.tag{{display:inline-block;padding:.1rem .4rem;border-radius:3px;font-size:.75rem;font-weight:600;margin-right:.25rem}}
.tag-green{{background:#d4edda;color:#155724}}
.tag-yellow{{background:#fff3cd;color:#856404}}
.tag-orange{{background:#fde8c8;color:#7a4100}}
.tag-red{{background:#f8d7da;color:#721c24}}
</style>
</head>
<body>
<div class="header">
  <h1>{T['title']} <span style="font-size:.7rem;background:rgba(255,255,255,.2);padding:.2rem .5rem;border-radius:4px;vertical-align:middle">{T['lang_label']}</span></h1>
  <p>{subtitle_html}</p>
</div>

<div class="content">

<!-- ── INFO NOTICE ──────────────────────────────────────────────── -->
<div class="card card-note" style="padding:.85rem 1.25rem;margin-bottom:1rem">
  {notice_html}
</div>

<!-- ── STAT BOXES ────────────────────────────────────────────────── -->
<div class="stats-row">
  <div class="stat-box"><div class="num">{n_scored:,}</div><div class="lbl">{T['lbl_runs']}</div></div>
  <div class="stat-box"><div class="num">{len(combos)}</div><div class="lbl">{T['lbl_combos']}</div></div>
  <div class="stat-box"><div class="num">{len(per_q)}</div><div class="lbl">{T['lbl_questions']}</div></div>
  <div class="stat-box"><div class="num" style="font-size:1.4rem">{fmt(best.get('avg_score'))}</div><div class="lbl">{T['lbl_best_avg']}</div></div>
  <div class="stat-box"><div class="num" style="font-size:1.4rem">{overall_avg_faith:.4f}</div><div class="lbl">{T['lbl_avg_faith']}</div></div>
  <div class="stat-box"><div class="num" style="font-size:1.4rem">{overall_avg_ar:.4f}</div><div class="lbl">{T['lbl_avg_ar']}</div></div>
  <div class="stat-box"><div class="num" style="font-size:1.4rem">{overall_avg_cp:.4f}</div><div class="lbl">{T['lbl_avg_cp']}</div></div>
  <div class="stat-box"><div class="num" style="font-size:1.4rem">{overall_avg_cr:.4f}</div><div class="lbl">{T['lbl_avg_cr']}</div></div>
</div>

<!-- ── METRIC GUIDE ───────────────────────────────────────────────── -->
<div class="card">
  <h2>{T['h_metric_guide']}</h2>
  <div class="metric-def">

    <div class="metric-card">
      <div class="metric-name">🔒 {T['m_faith_name']}</div>
      <div class="metric-what">{T['m_faith_what']}</div>
      <div class="metric-how"><em>{T['m_faith_how']}</em></div>
      <div class="metric-formula">formula: {T['m_faith_formula']}</div>
      <div class="metric-interp">{T['m_faith_interp']}</div>
    </div>

    <div class="metric-card">
      <div class="metric-name">🎯 {T['m_ar_name']}</div>
      <div class="metric-what">{T['m_ar_what']}</div>
      <div class="metric-how"><em>{T['m_ar_how']}</em></div>
      <div class="metric-formula">formula: {T['m_ar_formula']}</div>
      <div class="metric-interp">{T['m_ar_interp']}</div>
    </div>

    <div class="metric-card">
      <div class="metric-name">🎚️ {T['m_cp_name']}</div>
      <div class="metric-what">{T['m_cp_what']}</div>
      <div class="metric-how"><em>{T['m_cp_how']}</em></div>
      <div class="metric-formula">formula: {T['m_cp_formula']}</div>
      <div class="metric-interp">{T['m_cp_interp']}</div>
    </div>

    <div class="metric-card">
      <div class="metric-name">📡 {T['m_cr_name']}</div>
      <div class="metric-what">{T['m_cr_what']}</div>
      <div class="metric-how"><em>{T['m_cr_how']}</em></div>
      <div class="metric-formula">formula: {T['m_cr_formula']}</div>
      <div class="metric-interp">{T['m_cr_interp']}</div>
    </div>

  </div>
</div>

<!-- ── KEY OBSERVATIONS ───────────────────────────────────────────── -->
<div class="card card-obs">
  <h2>{T['h_observations']}</h2>
  <ul class="obs-list">{obs_html}</ul>
</div>

<!-- ── BEST COMBINATION ───────────────────────────────────────────── -->
<div class="card">
  <h2>{T['h_best']}</h2>
  <div class="best-card">
    <div class="best-meta">
      <div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;color:#888">{T['best_label']}</div>
      <div class="combo-tag">{best.get('combo', '—')}</div>
      <table>
        <tr><td>🔒 {T['m_faith_name'].split(' (')[0]}</td><td style="color:{('#155724' if (best.get('faithfulness') or 0)>=0.7 else '#856404')}">{fmt(best.get('faithfulness'))}</td></tr>
        <tr><td>🎯 {T['m_ar_name'].split(' (')[0]}</td><td style="color:{('#155724' if (best.get('answer_relevancy') or 0)>=0.7 else '#856404')}">{fmt(best.get('answer_relevancy'))}</td></tr>
        <tr><td>🎚️ {T['m_cp_name'].split(' (')[0]}</td><td style="color:{('#155724' if (best.get('context_precision') or 0)>=0.5 else '#856404')}">{fmt(best.get('context_precision'))}</td></tr>
        <tr><td>📡 {T['m_cr_name'].split(' (')[0]}</td><td style="color:{('#155724' if (best.get('context_recall') or 0)>=0.5 else '#856404')}">{fmt(best.get('context_recall'))}</td></tr>
        <tr><td><strong>{T['best_avg_label']}</strong></td><td><strong>{fmt(best.get('avg_score'))}</strong></td></tr>
        <tr><td>{T['best_ret_label']}</td><td>{best.get('retrieval_time_ms', '—')} ms</td></tr>
        <tr><td>{T['best_gen_label']}</td><td>{best.get('generation_time_ms', '—')} ms</td></tr>
        <tr><td>{T['best_q_label']}</td><td>{best.get('n_questions', '—')}</td></tr>
      </table>
    </div>
    <div class="radar-wrap">
      <canvas id="radarChart" width="300" height="300"></canvas>
    </div>
  </div>
</div>

<!-- ── TOP 10 TABLE ───────────────────────────────────────────────── -->
<div class="card">
  <h2>{T['h_top10']}</h2>
  <p style="font-size:.83rem;color:#666;margin-bottom:.75rem">{T['top10_desc']}</p>
  <div style="overflow-x:auto">
  <table class="data" id="top10Table">
    <thead><tr>
      <th>{T['col_rank']}</th>
      <th onclick="sortTable('top10Table',1)">{T['col_llm']} ↕</th>
      <th onclick="sortTable('top10Table',2)">{T['col_embed']} ↕</th>
      <th onclick="sortTable('top10Table',3)">{T['col_chunk']} ↕</th>
      <th onclick="sortTable('top10Table',4)">{T['col_faith']}</th>
      <th onclick="sortTable('top10Table',5)">{T['col_ar']}</th>
      <th onclick="sortTable('top10Table',6)">{T['col_cp']}</th>
      <th onclick="sortTable('top10Table',7)">{T['col_cr']}</th>
      <th onclick="sortTable('top10Table',8)">{T['col_avg']}</th>
      <th onclick="sortTable('top10Table',9)">{T['col_ret_ms']}</th>
      <th onclick="sortTable('top10Table',10)">{T['col_gen_ms']}</th>
      <th onclick="sortTable('top10Table',11)">{T['col_questions']}</th>
    </tr></thead>
    <tbody>{top10_rows_html}</tbody>
  </table>
  </div>
</div>

<!-- ── EMBEDDING × CHUNK HEATMAP ──────────────────────────────────── -->
<div class="card">
  <h2>{T['h_heatmap']}</h2>
  <p style="font-size:.85rem;color:#555;margin-bottom:.75rem">{T['heatmap_desc']}</p>
  {heatmap_html}
</div>

<!-- ── BREAKDOWN BY DIMENSION ─────────────────────────────────────── -->
<div class="card">
  <h2>{T['h_breakdown']}</h2>
  <p style="font-size:.83rem;color:#666;margin-bottom:1rem">{T['breakdown_desc']}</p>
  <div class="two-col">
    <div>
      <h3>{T['h_by_llm']}</h3>
      {llm_table}
    </div>
    <div>
      <h3>{T['h_by_chunk']}</h3>
      {chunk_table}
    </div>
  </div>
  <div style="margin-top:1.25rem">
    <h3>{T['h_by_embed']}</h3>
    {embed_table}
  </div>
</div>

<!-- ── FORMAT & CATEGORY ──────────────────────────────────────────── -->
<div class="card">
  <h2>{T['h_format_cat']}</h2>
  <p style="font-size:.83rem;color:#666;margin-bottom:1rem">{T['format_cat_desc']}</p>
  <div class="two-col">
    <div>
      <h3>{T['h_doc_format']}</h3>
      {fmt_table}
      <p style="font-size:.78rem;color:#777;margin-top:.4rem">{T['format_note']}</p>
    </div>
    <div>
      <h3>{T['h_cat']}</h3>
      {cat_table}
      <p style="font-size:.78rem;color:#777;margin-top:.4rem">{T['cat_note']}</p>
    </div>
  </div>
</div>

<!-- ── COMPARISON CHARTS ───────────────────────────────────────────── -->
<div class="card">
  <h2>{T['h_charts']}</h2>
  <div class="charts-row">
    <div class="chart-box"><h3>{T['chart_llm']}</h3><canvas id="chartLLM" height="200"></canvas></div>
    <div class="chart-box"><h3>{T['chart_embed']}</h3><canvas id="chartEmbed" height="200"></canvas></div>
    <div class="chart-box"><h3>{T['chart_chunk']}</h3><canvas id="chartChunk" height="200"></canvas></div>
  </div>
</div>

<!-- ── FULL RANKINGS ──────────────────────────────────────────────── -->
<div class="card">
  <h2>{T['h_full_rankings']}</h2>
  <div class="filter-row">
    <select id="filterLLM" onchange="renderCombos()"><option value="">{T['filter_all_llm']}</option></select>
    <select id="filterEmbed" onchange="renderCombos()"><option value="">{T['filter_all_embed']}</option></select>
    <select id="filterChunk" onchange="renderCombos()"><option value="">{T['filter_all_chunk']}</option></select>
  </div>
  <div style="overflow-x:auto">
  <table class="data" id="combosTable">
    <thead><tr>
      <th onclick="sortDynTable('combo')">Combination ↕</th>
      <th onclick="sortDynTable('faithfulness')">{T['col_faith']}</th>
      <th onclick="sortDynTable('answer_relevancy')">{T['col_ar']}</th>
      <th onclick="sortDynTable('context_precision')">{T['col_cp']}</th>
      <th onclick="sortDynTable('context_recall')">{T['col_cr']}</th>
      <th onclick="sortDynTable('avg_score')">{T['col_avg']}</th>
      <th onclick="sortDynTable('retrieval_time_ms')">{T['col_ret_ms']}</th>
      <th onclick="sortDynTable('generation_time_ms')">{T['col_gen_ms']}</th>
      <th onclick="sortDynTable('n_questions')">{T['col_questions']}</th>
    </tr></thead>
    <tbody id="combosBody"></tbody>
  </table>
  </div>
</div>

<!-- ── PER-QUESTION TABLE ──────────────────────────────────────────── -->
<div class="card">
  <h2>{T['h_per_question']}</h2>
  <p style="font-size:.83rem;color:#666;margin-bottom:.75rem">{T['per_q_desc']}</p>
  <div class="filter-row">
    <input id="searchQ" type="text" placeholder="{T['search_q']}" oninput="renderPerQ()" style="width:300px"/>
    <select id="filterCat" onchange="renderPerQ()"><option value="">{T['filter_all_cat']}</option></select>
  </div>
  <div style="overflow-x:auto">
  <table class="data" id="perQTable">
    <thead><tr>
      <th>{T['col_question']}</th><th>{T['col_category']}</th><th>{T['col_best_combo']}</th>
      <th onclick="sortPerQ('avg_score')">{T['col_avg']}</th>
      <th onclick="sortPerQ('faithfulness')">{T['col_faith']}</th>
      <th onclick="sortPerQ('answer_relevancy')">{T['col_ar']}</th>
      <th onclick="sortPerQ('context_precision')">{T['col_cp']}</th>
      <th onclick="sortPerQ('context_recall')">{T['col_cr']}</th>
    </tr></thead>
    <tbody id="perQBody"></tbody>
  </table>
  </div>
</div>

</div><!-- /content -->

<script>
const ALL_COMBOS={all_combos_js};
const ALL_PERQ={per_q_js};

new Chart(document.getElementById('radarChart').getContext('2d'),{{
  type:'radar',
  data:{{labels:{json.dumps(radar_labels)},datasets:[{{
    label:'{best.get("combo","Best").replace("'","\\'")}',
    data:{json.dumps(radar_values)},
    backgroundColor:'rgba(26,60,94,0.18)',
    borderColor:'#1a3c5e',
    pointBackgroundColor:'#1a3c5e'
  }}]}},
  options:{{responsive:false,scales:{{r:{{min:0,max:1,ticks:{{stepSize:0.2}}}}}},plugins:{{legend:{{position:'bottom'}}}}}}
}});

function makeBar(id,dataJson){{
  const d=JSON.parse(dataJson);
  new Chart(document.getElementById(id).getContext('2d'),{{
    type:'bar',
    data:{{labels:d.labels,datasets:[{{label:'Avg Score',data:d.scores,
      backgroundColor:d.scores.map(s=>s>=0.7?'#28a745':s>=0.5?'#ffc107':s>=0.3?'#fd7e14':'#dc3545')}}]}},
    options:{{indexAxis:'y',responsive:true,scales:{{x:{{min:0,max:1}}}},plugins:{{legend:{{display:false}}}}}}
  }});
}}
makeBar('chartLLM',   '{by_llm_js}');
makeBar('chartEmbed', '{by_embed_js}');
makeBar('chartChunk', '{by_chunk_js}');

let sortDirs={{}};
function sortTable(tableId,colIdx){{
  const tbl=document.getElementById(tableId),tbody=tbl.tBodies[0];
  const rows=Array.from(tbody.rows);
  const dir=sortDirs[tableId+colIdx]=-(sortDirs[tableId+colIdx]||-1);
  rows.sort((a,b)=>{{
    const va=a.cells[colIdx].innerText.replace('—',''),vb=b.cells[colIdx].innerText.replace('—','');
    const fa=parseFloat(va),fb=parseFloat(vb);
    return(!isNaN(fa)&&!isNaN(fb))?dir*(fa-fb):dir*va.localeCompare(vb);
  }});
  rows.forEach(r=>tbody.appendChild(r));
}}

let comboSortKey='avg_score',comboSortDir=-1;
function populateFilters(){{
  const llms=[...new Set(ALL_COMBOS.map(c=>c.llm))].sort();
  const embeds=[...new Set(ALL_COMBOS.map(c=>c.embed))].sort();
  const chunks=[...new Set(ALL_COMBOS.map(c=>c.chunk))].sort();
  llms.forEach(v=>filterLLM.add(new Option(v,v)));
  embeds.forEach(v=>filterEmbed.add(new Option(v,v)));
  chunks.forEach(v=>filterChunk.add(new Option(v,v)));
  const cats=[...new Set(ALL_PERQ.map(r=>r.category).filter(Boolean))].sort();
  cats.forEach(v=>filterCat.add(new Option(v,v)));
}}
function badge(v){{
  if(v===null||v===undefined||v==='')return'—';
  const f=parseFloat(v);
  if(isNaN(f))return'—';
  const cls=f>=0.7?'badge-high':f>=0.5?'badge-mid':f>=0.3?'badge-low':'badge-poor';
  return`<span class="badge ${{cls}}">${{f.toFixed(4)}}</span>`;
}}
function renderCombos(){{
  const llmF=filterLLM.value,embedF=filterEmbed.value,chunkF=filterChunk.value;
  let data=ALL_COMBOS.filter(c=>(!llmF||c.llm===llmF)&&(!embedF||c.embed===embedF)&&(!chunkF||c.chunk===chunkF));
  data.sort((a,b)=>comboSortDir*((b[comboSortKey]||0)-(a[comboSortKey]||0)));
  document.getElementById('combosBody').innerHTML=data.map(c=>`
    <tr>
      <td>${{c.combo}}</td>
      <td>${{badge(c.faithfulness)}}</td>
      <td>${{badge(c.answer_relevancy)}}</td>
      <td>${{badge(c.context_precision)}}</td>
      <td>${{badge(c.context_recall)}}</td>
      <td><strong>${{badge(c.avg_score)}}</strong></td>
      <td style="text-align:center">${{c.retrieval_time_ms?c.retrieval_time_ms.toFixed(0):'—'}}</td>
      <td style="text-align:center">${{c.generation_time_ms?c.generation_time_ms.toFixed(0):'—'}}</td>
      <td style="text-align:center">${{c.n_questions}}</td>
    </tr>`).join('');
}}
function sortDynTable(key){{
  if(comboSortKey===key)comboSortDir*=-1;else{{comboSortKey=key;comboSortDir=-1;}}
  renderCombos();
}}

let perQSortKey='avg_score',perQSortDir=-1;
function renderPerQ(){{
  const q=document.getElementById('searchQ').value.toLowerCase();
  const cat=document.getElementById('filterCat').value;
  let data=ALL_PERQ.filter(r=>(!q||r.question.toLowerCase().includes(q))&&(!cat||r.category===cat));
  data.sort((a,b)=>perQSortDir*((b[perQSortKey]||0)-(a[perQSortKey]||0)));
  document.getElementById('perQBody').innerHTML=data.map(r=>`
    <tr>
      <td style="max-width:300px;white-space:normal;font-size:.82rem">${{r.question}}</td>
      <td><span style="font-size:.8rem">${{r.category}}</span></td>
      <td style="font-size:.8rem">${{r.best_combo}}</td>
      <td>${{badge(r.avg_score)}}</td>
      <td>${{badge(r.faithfulness)}}</td>
      <td>${{badge(r.answer_relevancy)}}</td>
      <td>${{badge(r.context_precision)}}</td>
      <td>${{badge(r.context_recall)}}</td>
    </tr>`).join('');
}}
function sortPerQ(key){{
  if(perQSortKey===key)perQSortDir*=-1;else{{perQSortKey=key;perQSortDir=-1;}}
  renderPerQ();
}}

populateFilters();renderCombos();renderPerQ();
</script>
</body>
</html>"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info(f"Report written to: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate interactive HTML report from eval_scores (EN + ES)."
    )
    parser.add_argument(
        "--lang",
        default="both",
        choices=["en", "es", "both"],
        help="Language: en, es, or both (default: both).",
    )
    parser.add_argument(
        "--out-prefix",
        default="reports/rag_evaluation_report",
        help="Output file prefix. Files will be named <prefix>_en.html / <prefix>_es.html.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        metavar="N",
        help="Limit report to the first N questions in the knowledge base (by ID). Default: all.",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    load_dotenv()
    args = parse_args()

    logging.info("Connecting to database...")
    conn = get_db_connection()

    try:
        rows          = fetch_data(conn, questions_limit=args.questions)
        score_summary = fetch_score_summary(conn)
        extra         = fetch_extra_breakdowns(conn, questions_limit=args.questions)

        if not rows:
            logging.warning("No successfully scored rows found. Run run_ragas.py first.")
            return

        logging.info(f"Fetched {len(rows):,} scored rows.")
        combos = aggregate_combos(rows)
        per_q  = per_question_best(rows)
        logging.info(f"Aggregated {len(combos)} combinations, {len(per_q)} questions.")

        langs = ["en", "es"] if args.lang == "both" else [args.lang]
        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for lang in langs:
            out_path = f"{args.out_prefix}_{lang}.html"
            build_html(
                rows=rows, combos=combos, per_q=per_q,
                score_summary=score_summary,
                extra=extra,
                judge_llm="gpt-4o-mini (Faithfulness, CtxPrecision) + gpt-oss:20b (AR, CtxRecall)",
                generated_at=ts,
                out_path=out_path,
                lang=lang,
            )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
