#!/usr/bin/env python3
"""Genera SQL de exportación para knowledge_base + eval_runs + eval_scores (qwen3:8b|bge-m3|medium)."""
import json, os, sys
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    dbname=os.getenv("DB_NAME", "db_rag"),
    user=os.getenv("DB_USER", "sgadmin"),
    password=os.getenv("DB_PASSWORD", "sg4dm1n!"),
)

def ql(v):
    """Escape de texto usando dollar-quoting para evitar problemas con comillas."""
    if v is None:
        return "NULL"
    s = str(v).replace("$$", "$ $")
    return f"$${s}$$"

def fn(v):
    return "NULL" if v is None else str(round(float(v), 6))

lines = []
lines += [
    "-- ============================================================",
    "-- EXPORTACIÓN: Evaluación qwen3:8b | bge-m3 | medium",
    "-- Fecha: 2026-04-29 | 108 preguntas · 108 eval_runs · 108 eval_scores",
    "-- Métricas promedio: relevancy=0.809  recall=0.480  faithfulness=0.827  precision=0.760",
    "-- ============================================================",
    "",
    "BEGIN;",
    "",
]

# ── 1. knowledge_base ─────────────────────────────────────────
lines += ["-- ── 1. knowledge_base ──────────────────────────────────────────", ""]
with conn.cursor() as cur:
    cur.execute("SELECT id, question, answer, category, phase, source_document, created_at FROM knowledge_base ORDER BY id")
    for id_, q, a, cat, ph, src, cr_at in cur.fetchall():
        lines.append(
            f"INSERT INTO knowledge_base (id, question, answer, category, phase, source_document, created_at) "
            f"VALUES ({id_}, {ql(q)}, {ql(a)}, {ql(cat)}, {ql(ph)}, {ql(src)}, '{cr_at}') "
            f"ON CONFLICT (id) DO NOTHING;"
        )

lines.append("")

# ── IDs de catálogo ───────────────────────────────────────────
with conn.cursor() as cur:
    cur.execute("SELECT id FROM llm_models WHERE model_name='qwen3:8b'")
    llm_id = cur.fetchone()[0]
    cur.execute("SELECT id FROM embedding_models WHERE model_name='bge-m3'")
    emb_id = cur.fetchone()[0]
    cur.execute("SELECT id FROM chunk_configs WHERE name='medium'")
    chk_id = cur.fetchone()[0]

lines += [
    f"-- IDs en BD origen: llm={llm_id}, embed={emb_id}, chunk={chk_id}",
    "-- Si los IDs difieren en destino, ajustar las variables antes de ejecutar.",
    f"-- En destino, verificar: SELECT id FROM llm_models WHERE model_name='qwen3:8b';",
    "",
    "-- ── 2. eval_runs ────────────────────────────────────────────────",
    "",
]

# ── 2. eval_runs ──────────────────────────────────────────────
with conn.cursor() as cur:
    cur.execute("""
        SELECT er.id, er.knowledge_base_id, er.retrieved_contexts, er.k_retrieved,
               er.generated_answer, er.retrieval_time_ms, er.generation_time_ms,
               er.total_time_ms, er.created_at
        FROM eval_runs er
        JOIN llm_models lm ON er.llm_model_id = lm.id
        JOIN embedding_models em ON er.embedding_model_id = em.id
        JOIN chunk_configs cc ON er.chunk_config_id = cc.id
        WHERE lm.model_name='qwen3:8b' AND em.model_name='bge-m3'
          AND cc.name='medium' AND er.status='success'
        ORDER BY er.id
    """)
    for er_id, kb_id, ctx, k, ans, r_ms, g_ms, t_ms, cr_at in cur.fetchall():
        ctx_json = json.dumps(ctx) if isinstance(ctx, (list, dict)) else (ctx or "[]")
        lines.append(
            f"INSERT INTO eval_runs "
            f"(id, llm_model_id, embedding_model_id, chunk_config_id, knowledge_base_id, "
            f"retrieved_contexts, k_retrieved, generated_answer, status, error_message, "
            f"retrieval_time_ms, generation_time_ms, total_time_ms, created_at) VALUES ("
            f"{er_id}, {llm_id}, {emb_id}, {chk_id}, {kb_id}, "
            f"{ql(ctx_json)}, {k}, {ql(ans)}, 'success', NULL, "
            f"{r_ms}, {g_ms}, {t_ms}, '{cr_at}') "
            f"ON CONFLICT (llm_model_id, embedding_model_id, chunk_config_id, knowledge_base_id) DO NOTHING;"
        )

lines += ["", "-- ── 3. eval_scores ──────────────────────────────────────────────", ""]

# ── 3. eval_scores ────────────────────────────────────────────
with conn.cursor() as cur:
    cur.execute("""
        SELECT es.id, es.eval_run_id, es.faithfulness, es.answer_relevancy,
               es.context_precision, es.context_recall, es.judge_llm, es.created_at
        FROM eval_scores es
        JOIN eval_runs er ON es.eval_run_id = er.id
        JOIN llm_models lm ON er.llm_model_id = lm.id
        JOIN embedding_models em ON er.embedding_model_id = em.id
        JOIN chunk_configs cc ON er.chunk_config_id = cc.id
        WHERE lm.model_name='qwen3:8b' AND em.model_name='bge-m3'
          AND cc.name='medium' AND es.status='success'
        ORDER BY es.eval_run_id
    """)
    for es_id, er_id, faith, relev, prec, recall, judge, cr_at in cur.fetchall():
        lines.append(
            f"INSERT INTO eval_scores "
            f"(id, eval_run_id, faithfulness, answer_relevancy, context_precision, context_recall, "
            f"judge_llm, status, created_at) VALUES ("
            f"{es_id}, {er_id}, {fn(faith)}, {fn(relev)}, {fn(prec)}, {fn(recall)}, "
            f"{ql(judge)}, 'success', '{cr_at}') "
            f"ON CONFLICT (eval_run_id) DO NOTHING;"
        )

lines += [
    "",
    "-- ── 4. Resetear secuencias ───────────────────────────────────────",
    "SELECT setval('knowledge_base_id_seq', (SELECT MAX(id) FROM knowledge_base));",
    "SELECT setval('eval_runs_id_seq',      (SELECT MAX(id) FROM eval_runs));",
    "SELECT setval('eval_scores_id_seq',    (SELECT MAX(id) FROM eval_scores));",
    "",
    "COMMIT;",
    "",
    "-- FIN DEL SCRIPT",
]

conn.close()

out = Path(__file__).parent.parent / "postgres" / "migrations" / "006_eval_qwen3_bge_m3_medium.sql"
out.write_text("\n".join(lines), encoding="utf-8")
kb   = sum(1 for l in lines if "INSERT INTO knowledge_base" in l)
er   = sum(1 for l in lines if "INSERT INTO eval_runs" in l)
es   = sum(1 for l in lines if "INSERT INTO eval_scores" in l)
print(f"OK  →  {out.name}")
print(f"     knowledge_base : {kb} filas")
print(f"     eval_runs      : {er} filas")
print(f"     eval_scores    : {es} filas")
print(f"     tamaño         : {out.stat().st_size / 1024:.1f} KB")
