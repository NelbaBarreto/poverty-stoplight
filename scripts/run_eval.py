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
from itertools import product

import requests
import psycopg2
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

WEBCHAT_API_URL = os.getenv("WEBCHAT_API_URL", "http://localhost:8000")
WEBCHAT_SECRET  = os.getenv("WEBCHAT_SECRET",  "W3bCh4tFup4")
K_RETRIEVED     = 8

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


def get_active_version_id(conn) -> int:
    """Return the id of the currently active version_agente."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM version_agente WHERE activa = TRUE ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("No hay ninguna version_agente activa en la BD.")
        return row[0]


def already_done(conn, llm_id: int, embed_id: int, chunk_id: int, kb_id: int, version_id: int) -> bool:
    """Check if a successful run already exists for this combination + version."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM eval_runs
            WHERE llm_model_id = %s
              AND embedding_model_id = %s
              AND chunk_config_id = %s
              AND knowledge_base_id = %s
              AND version_agente_id = %s
              AND status = 'success'
            """,
            (llm_id, embed_id, chunk_id, kb_id, version_id),
        )
        return cur.fetchone() is not None


def upsert_eval_run(conn, llm_id, embed_id, chunk_id, kb_id, version_id: int, payload: dict) -> None:
    """Insert or update an eval_run row."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_runs
                (llm_model_id, embedding_model_id, chunk_config_id, knowledge_base_id,
                 version_agente_id,
                 retrieved_contexts, k_retrieved, generated_answer,
                 status, error_message,
                 retrieval_time_ms, generation_time_ms, total_time_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (llm_model_id, embedding_model_id, chunk_config_id, knowledge_base_id, version_agente_id)
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
                llm_id, embed_id, chunk_id, kb_id, version_id,
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
# Webchat API helpers
# ---------------------------------------------------------------------------

def is_api_available() -> bool:
    try:
        resp = requests.get(f"{WEBCHAT_API_URL}/api/debug/ping", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def _get_token() -> str:
    """Fetch the daily auth token from the debug endpoint."""
    resp = requests.get(f"{WEBCHAT_API_URL}/api/debug/token", timeout=10)
    resp.raise_for_status()
    return resp.json()["expected_token"]


def _create_session(token: str) -> str:
    resp = requests.post(
        f"{WEBCHAT_API_URL}/api/session",
        headers={"Authorization": f"Bearer {token}"},
        json={"origin": "run_eval"},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["session_id"]


def call_webchat(question: str, token: str) -> dict:
    """
    Create a fresh session and call POST /api/chat — same pipeline as the web UI.
    Returns { response, sources }.
    """
    session_id = _create_session(token)
    resp = requests.post(
        f"{WEBCHAT_API_URL}/api/chat",
        headers={"Authorization": f"Bearer {token}"},
        json={"session_id": session_id, "message": question},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()

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
        "--version-id",
        metavar="ID",
        type=int,
        default=None,
        help="version_agente.id to tag runs with. Default: active version.",
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

    args = parse_args()

    logging.info(f"WEBCHAT_API_URL={WEBCHAT_API_URL}")
    if not is_api_available():
        logging.error("Webchat API no disponible. Verificá que esté corriendo.")
        sys.exit(1)
    logging.info("Webchat API OK")
    token = _get_token()
    logging.info("Token diario obtenido")

    logging.info("Connecting to database...")
    conn = get_db_connection()

    try:
        version_id = args.version_id if args.version_id is not None else get_active_version_id(conn)
        logging.info(f"Using version_agente_id={version_id}")

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

                # Skip already-successful runs (resumability)
                if already_done(conn, llm_id, embed_id, chunk_id, kb_id, version_id):
                    pbar.update(1)
                    skipped += 1
                    continue

                t_total_start = time.monotonic()

                try:
                    data = call_webchat(question, token)
                    total_ms = int((time.monotonic() - t_total_start) * 1000)

                    upsert_eval_run(conn, llm_id, embed_id, chunk_id, kb_id, version_id, {
                        "retrieved_contexts":  data.get("sources", []),
                        "k_retrieved":         K_RETRIEVED,
                        "generated_answer":    data.get("response", ""),
                        "status":              "success",
                        "error_message":       None,
                        "retrieval_time_ms":   data.get("retrieval_time_ms"),
                        "generation_time_ms":  data.get("generation_time_ms"),
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
                        upsert_eval_run(conn, llm_id, embed_id, chunk_id, kb_id, version_id, {
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
        logging.info(f"  Skipped (already done): {skipped:,}")
        logging.info(f"  Errors this run: {errors:,}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
