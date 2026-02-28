#!/usr/bin/env python3
"""
Resumable RAG evaluation pipeline for Semáforo de Pobreza.

Iterates over all combinations of:
  - LLM models (from llm_models table)
  - Embedding models (from embedding_models table)
  - Chunk configs (from chunk_configs table)
  - Knowledge base questions (from knowledge_base table)

Each result is stored in eval_runs for downstream RAGAS evaluation.

RAGAS mapping:
  question     → knowledge_base.question
  answer       → eval_runs.generated_answer
  contexts     → [c["chunk_text"] for c in retrieved_contexts]
  ground_truth → knowledge_base.answer
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from itertools import product

import requests
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434"
K_RETRIEVED = 8

RAG_SYSTEM_PROMPT = """Eres Luz, la asistente conversacional del Banco de Soluciones de la Fundacion Paraguaya.
Responde UNICAMENTE con la informacion proporcionada en el CONTEXTO a continuacion.
Si la informacion no esta en el contexto, dilo claramente.
Cita siempre las fuentes (nombres de archivo) cuando respondas.
No inventes datos. Se concisa pero completa. Responde en espanol."""

# ---------------------------------------------------------------------------
# Database helpers  (same pattern as scripts/ingest_all.py)
# ---------------------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "db_rag"),
        user=os.getenv("DB_USER", "sgadmin"),
        password=os.getenv("DB_PASSWORD", "sg4dm1n!"),
    )


def load_catalog(conn) -> dict:
    """Return all catalog rows needed for building the eval matrix."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, model_name FROM llm_models ORDER BY id")
        llm_models = cur.fetchall()

        cur.execute("SELECT id, model_name, table_name FROM embedding_models ORDER BY id")
        embed_models = cur.fetchall()

        cur.execute("SELECT id, name FROM chunk_configs ORDER BY id")
        chunk_configs = cur.fetchall()

        cur.execute("SELECT id, question FROM knowledge_base ORDER BY id")
        kb_questions = cur.fetchall()

    return {
        "llm_models":   llm_models,    # [(id, model_name), ...]
        "embed_models": embed_models,  # [(id, model_name, table_name), ...]
        "chunk_configs": chunk_configs, # [(id, name), ...]
        "kb_questions": kb_questions,  # [(id, question), ...]
    }


def already_done(conn, llm_id: int, embed_id: int, chunk_id: int, kb_id: int) -> bool:
    """Check if a successful run already exists for this combination."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM eval_runs
            WHERE llm_model_id = %s
              AND embedding_model_id = %s
              AND chunk_config_id = %s
              AND knowledge_base_id = %s
              AND status = 'success'
            """,
            (llm_id, embed_id, chunk_id, kb_id),
        )
        return cur.fetchone() is not None


def upsert_eval_run(conn, llm_id, embed_id, chunk_id, kb_id, payload: dict) -> None:
    """Insert or update an eval_run row."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_runs
                (llm_model_id, embedding_model_id, chunk_config_id, knowledge_base_id,
                 retrieved_contexts, k_retrieved, generated_answer,
                 status, error_message,
                 retrieval_time_ms, generation_time_ms, total_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (llm_model_id, embedding_model_id, chunk_config_id, knowledge_base_id)
            DO UPDATE SET
                retrieved_contexts = EXCLUDED.retrieved_contexts,
                k_retrieved        = EXCLUDED.k_retrieved,
                generated_answer   = EXCLUDED.generated_answer,
                status             = EXCLUDED.status,
                error_message      = EXCLUDED.error_message,
                retrieval_time_ms  = EXCLUDED.retrieval_time_ms,
                generation_time_ms = EXCLUDED.generation_time_ms,
                total_time_ms      = EXCLUDED.total_time_ms,
                created_at         = CURRENT_TIMESTAMP
            """,
            (
                llm_id, embed_id, chunk_id, kb_id,
                json.dumps(payload.get("retrieved_contexts")),
                payload.get("k_retrieved", K_RETRIEVED),
                payload.get("generated_answer"),
                payload.get("status", "error"),
                payload.get("error_message"),
                payload.get("retrieval_time_ms"),
                payload.get("generation_time_ms"),
                payload.get("total_time_ms"),
            ),
        )
    conn.commit()

# ---------------------------------------------------------------------------
# Ollama helpers  (same pattern as src/tools.py)
# ---------------------------------------------------------------------------

def get_query_embedding(model: str, query: str, base_url: str) -> list:
    """Embed a query string using Ollama /api/embed."""
    resp = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": query},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    embeddings = data.get("embeddings") or data.get("embedding")
    if isinstance(embeddings, list) and isinstance(embeddings[0], list):
        return embeddings[0]
    return embeddings


def is_model_available(model_name: str) -> bool:
    """Check if an Ollama model is available locally."""
    result = subprocess.run(
        ["ollama", "show", model_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0

# ---------------------------------------------------------------------------
# Retrieval  (replicates src/pgvector_manager.py:search_similar_new_schema)
# ---------------------------------------------------------------------------

def retrieve_contexts(
    conn,
    query: str,
    embed_model: str,
    embed_table: str,
    chunk_config_name: str,
    base_url: str,
    k: int = K_RETRIEVED,
) -> tuple[list[dict], int]:
    """
    Embed the query, then retrieve top-k similar chunks.

    Returns:
        (contexts, retrieval_time_ms)
        contexts: list of dicts with chunk_id, chunk_text, filename, format, distance
    """
    t0 = time.monotonic()
    embedding = get_query_embedding(embed_model, query, base_url)
    embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"

    with conn.cursor() as cur:
        query_sql = f"""
            SELECT
                c.id          AS chunk_id,
                c.chunk_text,
                d.filename,
                c.format,
                e.embedding <=> %s::vector AS distance
            FROM {embed_table} e
            JOIN chunks c        ON e.chunk_id      = c.id
            JOIN chunk_configs cc ON c.chunk_config_id = cc.id
            JOIN documents d     ON c.document_id    = d.id
            WHERE cc.name = %s
            ORDER BY e.embedding <=> %s::vector
            LIMIT %s
        """
        cur.execute(query_sql, (embedding_str, chunk_config_name, embedding_str, k))
        rows = cur.fetchall()

    contexts = [
        {
            "chunk_id":   row[0],
            "chunk_text": row[1],
            "filename":   row[2],
            "format":     row[3],
            "distance":   float(row[4]),
        }
        for row in rows
    ]
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return contexts, elapsed_ms

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_answer(
    llm_model: str,
    question: str,
    contexts: list[dict],
    base_url: str,
) -> tuple[str, int]:
    """
    Build a RAG prompt from contexts and call Ollama /api/generate.

    Returns:
        (answer_text, generation_time_ms)
    """
    context_block = "\n\n---\n\n".join(
        f"[Fuente: {c['filename']} | formato: {c['format']} | dist: {c['distance']:.4f}]\n{c['chunk_text']}"
        for c in contexts
    )

    full_prompt = (
        f"{RAG_SYSTEM_PROMPT}\n\n"
        f"## CONTEXTO (documentos encontrados):\n\n{context_block}\n\n"
        f"## PREGUNTA:\n{question}\n\n"
        f"## RESPUESTA:"
    )

    t0 = time.monotonic()
    resp = requests.post(
        f"{base_url}/api/generate",
        json={
            "model":  llm_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=300,
    )
    resp.raise_for_status()
    answer = resp.json().get("response", "").strip()
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    return answer, elapsed_ms

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RAG evaluations over all model/chunk/question combinations."
    )
    parser.add_argument(
        "--llm",
        metavar="MODEL",
        action="append",
        dest="llm_filter",
        default=None,
        help="LLM model name(s) to include (repeatable). Default: all.",
    )
    parser.add_argument(
        "--embed",
        metavar="MODEL",
        action="append",
        dest="embed_filter",
        default=None,
        help="Embedding model name(s) to include (repeatable). Default: all.",
    )
    parser.add_argument(
        "--chunk",
        metavar="NAME",
        action="append",
        dest="chunk_filter",
        default=None,
        help="Chunk config name(s) to include: small/medium/large (repeatable). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print combination counts only; do not write to DB.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    load_dotenv()
    args = parse_args()
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL_DEFAULT)

    logging.info("Connecting to database...")
    conn = get_db_connection()

    try:
        catalog = load_catalog(conn)

        # Apply CLI filters
        llm_models = catalog["llm_models"]
        if args.llm_filter:
            llm_models = [(i, n) for i, n in llm_models if n in args.llm_filter]

        embed_models = catalog["embed_models"]
        if args.embed_filter:
            embed_models = [(i, n, t) for i, n, t in embed_models if n in args.embed_filter]

        chunk_configs = catalog["chunk_configs"]
        if args.chunk_filter:
            chunk_configs = [(i, n) for i, n in chunk_configs if n in args.chunk_filter]

        kb_questions = catalog["kb_questions"]

        total = len(llm_models) * len(embed_models) * len(chunk_configs) * len(kb_questions)

        logging.info(
            f"Eval matrix: {len(llm_models)} LLMs × {len(embed_models)} embeddings × "
            f"{len(chunk_configs)} chunks × {len(kb_questions)} questions = {total:,} runs"
        )

        if args.dry_run:
            logging.info("--dry-run: no DB writes. Exiting.")
            return

        # Verify which LLM models are actually available
        available_llms = set()
        for llm_id, llm_name in llm_models:
            if is_model_available(llm_name):
                available_llms.add(llm_name)
                logging.info(f"  LLM OK: {llm_name}")
            else:
                logging.warning(f"  LLM UNAVAILABLE (will skip): {llm_name}")

        # Build all combinations
        combos = list(product(llm_models, embed_models, chunk_configs, kb_questions))

        errors = 0
        skipped = 0

        with tqdm(total=len(combos), desc="Eval runs", unit="run") as pbar:
            for (llm_id, llm_name), (embed_id, embed_name, embed_table), \
                    (chunk_id, chunk_name), (kb_id, question) in combos:

                pbar.set_postfix(
                    llm=llm_name[:12],
                    embed=embed_name[:10],
                    chunk=chunk_name,
                    errors=errors,
                )

                # Skip unavailable LLMs
                if llm_name not in available_llms:
                    pbar.update(1)
                    skipped += 1
                    continue

                # Skip already-successful runs (resumability)
                if already_done(conn, llm_id, embed_id, chunk_id, kb_id):
                    pbar.update(1)
                    skipped += 1
                    continue

                t_total_start = time.monotonic()

                try:
                    # --- Retrieval ---
                    contexts, retrieval_ms = retrieve_contexts(
                        conn, question, embed_name, embed_table, chunk_name, base_url
                    )

                    # --- Generation ---
                    answer, generation_ms = generate_answer(llm_name, question, contexts, base_url)

                    total_ms = int((time.monotonic() - t_total_start) * 1000)

                    upsert_eval_run(conn, llm_id, embed_id, chunk_id, kb_id, {
                        "retrieved_contexts":  contexts,
                        "k_retrieved":         K_RETRIEVED,
                        "generated_answer":    answer,
                        "status":              "success",
                        "error_message":       None,
                        "retrieval_time_ms":   retrieval_ms,
                        "generation_time_ms":  generation_ms,
                        "total_time_ms":       total_ms,
                    })

                except Exception as exc:
                    errors += 1
                    total_ms = int((time.monotonic() - t_total_start) * 1000)
                    logging.warning(
                        f"  ERROR llm={llm_name} embed={embed_name} chunk={chunk_name} "
                        f"kb_id={kb_id}: {exc}"
                    )
                    try:
                        upsert_eval_run(conn, llm_id, embed_id, chunk_id, kb_id, {
                            "retrieved_contexts":  None,
                            "k_retrieved":         K_RETRIEVED,
                            "generated_answer":    None,
                            "status":              "error",
                            "error_message":       str(exc)[:500],
                            "retrieval_time_ms":   None,
                            "generation_time_ms":  None,
                            "total_time_ms":       total_ms,
                        })
                    except Exception as db_exc:
                        logging.error(f"  Could not store error row: {db_exc}")
                        conn.rollback()

                pbar.update(1)

        # Final summary
        with conn.cursor() as cur:
            cur.execute("SELECT status, COUNT(*) FROM eval_runs GROUP BY status ORDER BY status")
            rows = cur.fetchall()

        logging.info("=" * 60)
        logging.info("Evaluation complete. eval_runs summary:")
        for status, count in rows:
            logging.info(f"  {status}: {count:,}")
        logging.info(f"  Skipped (already done / unavailable): {skipped:,}")
        logging.info(f"  Errors this run: {errors:,}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
