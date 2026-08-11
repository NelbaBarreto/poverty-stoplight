#!/usr/bin/env python3
"""
Resumable RAGAS scoring pipeline for Semáforo de Pobreza eval_runs.

Default mode (Ollama): scores Answer Relevancy + Context Recall for rows
with no eval_scores entry yet.

OpenAI mode (--openai): scores Faithfulness + Context Precision for rows
that already have eval_scores but still have NULL faithfulness, using
gpt-4o-mini as judge LLM and text-embedding-3-small as judge embeddings.

vLLM mode (--vllm): scores ALL 4 metrics in one pass using the local
vLLM server (Qwen3.5-27B-FP8) as judge — no OpenAI API key required.
The vLLM server exposes an OpenAI-compatible API so RAGAS uses ChatOpenAI
pointed at the local endpoint. Requires VLLM_BASE_URL and EMBED_BASE_URL
in .env (or env vars). Embedding server is expected at EMBED_BASE_URL
(port 8801); falls back to sentence-transformers locally if not set.

Usage:
    python scripts/run_ragas.py --dry-run
    python scripts/run_ragas.py --llm llama3.2:3b --embed bge-m3 --chunk small --batch-size 10
    python scripts/run_ragas.py --batch-size 50
    python scripts/run_ragas.py --openai --questions 30 --batch-size 20
    python scripts/run_ragas.py --vllm --version-id 5 --batch-size 10
    python scripts/run_ragas.py --vllm --dry-run
"""

import os
import sys
import json
import math
import logging
import argparse

import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUDGE_LLM        = "gpt-oss:20b"
JUDGE_EMBED      = "bge-m3"
DEFAULT_BATCH    = 50
OLLAMA_BASE_URL  = "http://localhost:11434"
VLLM_BASE_URL    = "http://localhost:8800"
EMBED_BASE_URL   = ""   # vLLM embedding server, e.g. http://localhost:8801

# HuggingFace name for bge-m3 (used in sentence-transformers fallback)
BGE_M3_HF = "BAAI/bge-m3"

# ---------------------------------------------------------------------------
# DB helpers  (same pattern as run_eval.py)
# ---------------------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "db_rag"),
        user=os.getenv("DB_USER", "sgadmin"),
        password=os.getenv("DB_PASSWORD", "sg4dm1n!"),
    )


def fetch_pending(conn, llm_filter, embed_filter, chunk_filter, questions_limit=None, version_id=None) -> list[dict]:
    """Return all eval_runs that still need RAGAS scoring."""
    where_clauses = [
        "er.status = 'success'",
        "es.id IS NULL",
    ]
    params: list = []

    if llm_filter:
        where_clauses.append("lm.model_name = ANY(%s)")
        params.append(llm_filter)
    if embed_filter:
        where_clauses.append("em.model_name = ANY(%s)")
        params.append(embed_filter)
    if chunk_filter:
        where_clauses.append("cc.name = ANY(%s)")
        params.append(chunk_filter)
    if version_id is not None:
        where_clauses.append("er.version_agente_id = %s")
        params.append(version_id)
    if questions_limit:
        where_clauses.append(
            "er.knowledge_base_id IN (SELECT id FROM knowledge_base ORDER BY id LIMIT %s)"
        )
        params.append(questions_limit)

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT
            er.id            AS eval_run_id,
            kb.question,
            er.generated_answer,
            er.retrieved_contexts,
            kb.answer        AS ground_truth
        FROM eval_runs er
        JOIN knowledge_base   kb ON er.knowledge_base_id   = kb.id
        JOIN llm_models       lm ON er.llm_model_id        = lm.id
        JOIN embedding_models em ON er.embedding_model_id  = em.id
        JOIN chunk_configs    cc ON er.chunk_config_id     = cc.id
        LEFT JOIN eval_scores es ON er.id                  = es.eval_run_id
        WHERE {where_sql}
        ORDER BY er.id
    """
    with conn.cursor() as cur:
        cur.execute(sql, params if params else None)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    return rows


def upsert_score(conn, eval_run_id: int, payload: dict) -> None:
    """Insert or update a row in eval_scores."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_scores
                (eval_run_id, faithfulness, answer_relevancy,
                 context_precision, context_recall,
                 judge_llm, status, error_message)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (eval_run_id) DO UPDATE SET
                faithfulness      = EXCLUDED.faithfulness,
                answer_relevancy  = EXCLUDED.answer_relevancy,
                context_precision = EXCLUDED.context_precision,
                context_recall    = EXCLUDED.context_recall,
                judge_llm         = EXCLUDED.judge_llm,
                status            = EXCLUDED.status,
                error_message     = EXCLUDED.error_message,
                created_at        = CURRENT_TIMESTAMP
            """,
            (
                eval_run_id,
                payload.get("faithfulness"),
                payload.get("answer_relevancy"),
                payload.get("context_precision"),
                payload.get("context_recall"),
                JUDGE_LLM,
                payload.get("status", "error"),
                payload.get("error_message"),
            ),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# OpenAI mode — fetch rows that need faithfulness / context precision
# ---------------------------------------------------------------------------

def fetch_pending_faithfulness(conn, llm_filter, embed_filter, chunk_filter, questions_limit=None, version_id=None) -> list[dict]:
    """Return scored eval_runs that still have NULL faithfulness OR NULL context_precision."""
    where_clauses = [
        "er.status = 'success'",
        "es.status = 'success'",
        "(es.faithfulness IS NULL OR es.context_precision IS NULL)",
    ]
    params: list = []

    if llm_filter:
        where_clauses.append("lm.model_name = ANY(%s)")
        params.append(llm_filter)
    if embed_filter:
        where_clauses.append("em.model_name = ANY(%s)")
        params.append(embed_filter)
    if chunk_filter:
        where_clauses.append("cc.name = ANY(%s)")
        params.append(chunk_filter)
    if version_id is not None:
        where_clauses.append("er.version_agente_id = %s")
        params.append(version_id)
    if questions_limit:
        where_clauses.append(
            "er.knowledge_base_id IN (SELECT id FROM knowledge_base ORDER BY id LIMIT %s)"
        )
        params.append(questions_limit)

    where_sql = " AND ".join(where_clauses)
    sql = f"""
        SELECT
            er.id            AS eval_run_id,
            kb.question,
            er.generated_answer,
            er.retrieved_contexts,
            kb.answer        AS ground_truth
        FROM eval_runs er
        JOIN knowledge_base   kb ON er.knowledge_base_id   = kb.id
        JOIN llm_models       lm ON er.llm_model_id        = lm.id
        JOIN embedding_models em ON er.embedding_model_id  = em.id
        JOIN chunk_configs    cc ON er.chunk_config_id     = cc.id
        JOIN eval_scores      es ON er.id                  = es.eval_run_id
        WHERE {where_sql}
        ORDER BY er.id
    """
    with conn.cursor() as cur:
        cur.execute(sql, params if params else None)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def upsert_faithfulness(conn, eval_run_id: int, faithfulness: float, context_precision: float) -> None:
    """Update only faithfulness and context_precision on an existing eval_scores row."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE eval_scores
               SET faithfulness      = %s,
                   context_precision = %s
             WHERE eval_run_id = %s
            """,
            (faithfulness, context_precision, eval_run_id),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# RAGAS helpers
# ---------------------------------------------------------------------------

def build_ragas_objects(base_url: str):
    """Instantiate RAGAS judge LLM, embeddings, metrics, and RunConfig."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        LLMContextRecall,
    )
    from ragas.run_config import RunConfig
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    judge_llm = LangchainLLMWrapper(
        ChatOllama(model=JUDGE_LLM, temperature=0, base_url=base_url)
    )
    judge_emb = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=JUDGE_EMBED, base_url=base_url)
    )
    # Final metric selection:
    #   AnswerRelevancy          — embedding-based, reliable, ~77% non-null coverage
    #   LLMContextRecall         — LLM-based, reliable, ~84% non-null coverage
    # Dropped (all produce NaN with local LLMs / missing required fields):
    #   Faithfulness             — needs structured claim-verdict JSON from LLM
    #   LLMContextPrecisionWithReference — same JSON-format dependency
    #   NonLLMContextPrecisionWithReference — needs 'reference_contexts' column
    metrics = [
        AnswerRelevancy(),
        LLMContextRecall(),
    ]
    # max_workers=1 forces sequential async execution — no parallelism
    # storms against Ollama which handles one request at a time.
    # timeout=300s gives each individual LLM call 5 min to respond.
    run_config = RunConfig(max_workers=1, timeout=300, max_retries=3, max_wait=30)
    return judge_llm, judge_emb, metrics, run_config


def build_openai_ragas_objects():
    """Instantiate OpenAI judge LLM, embeddings, and metrics for Faithfulness + ContextPrecision."""
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference
    from ragas.run_config import RunConfig
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model="gpt-4o-mini", temperature=0)
    )
    judge_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )
    metrics = [
        Faithfulness(),
        LLMContextPrecisionWithReference(),
    ]
    run_config = RunConfig(max_workers=4, timeout=120, max_retries=3, max_wait=30)
    return judge_llm, judge_emb, metrics, run_config


def build_vllm_ragas_objects(vllm_url: str, embed_url: str):
    """
    Instantiate RAGAS judge using local vLLM server (OpenAI-compatible).
    Scores ALL 4 metrics in one pass — no OpenAI API key needed.

    LLM judge : llm_factory → vllm_url/v1  (Qwen3.5-27B-FP8)
    Embeddings: OpenAIEmbeddings → embed_url/v1  (bge-m3 via vLLM)
                Falls back to HuggingFaceEmbeddings locally if embed_url empty.
    """
    from ragas.metrics.collections import (
        Faithfulness,
        AnswerRelevancy,
        LLMContextRecall,
        LLMContextPrecisionWithReference,
    )
    from ragas.run_config import RunConfig
    from ragas.llms import llm_factory
    from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
    from openai import OpenAI

    # Detect model name from vLLM /v1/models
    import requests as _req
    try:
        r = _req.get(f"{vllm_url}/v1/models", headers={"Authorization": "Bearer EMPTY"}, timeout=10)
        model_name = r.json()["data"][0]["id"]
    except Exception:
        model_name = "unknown"
    logging.info(f"[vllm] judge model detected: {model_name}")

    # LLM judge via llm_factory (new RAGAS API)
    openai_client = OpenAI(base_url=f"{vllm_url}/v1", api_key="EMPTY")
    judge_llm = llm_factory(model_name, client=openai_client, max_tokens=4096)

    if embed_url:
        try:
            r = _req.get(f"{embed_url}/v1/models", headers={"Authorization": "Bearer EMPTY"}, timeout=10)
            embed_model_name = r.json()["data"][0]["id"]
        except Exception:
            embed_model_name = BGE_M3_HF
        logging.info(f"[vllm] embed model: {embed_model_name} at {embed_url}")
        embed_client = OpenAI(base_url=f"{embed_url}/v1", api_key="EMPTY")
        judge_emb = RagasOpenAIEmbeddings(model=embed_model_name, client=embed_client)
    else:
        logging.info(f"[vllm] no EMBED_BASE_URL — using HuggingFaceEmbeddings ({BGE_M3_HF}) locally")
        from ragas.embeddings import HuggingFaceEmbeddings as RagasHFEmbeddings
        judge_emb = RagasHFEmbeddings(model_name=BGE_M3_HF)

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextRecall(),
        LLMContextPrecisionWithReference(),
    ]
    run_config = RunConfig(max_workers=2, timeout=300, max_retries=3, max_wait=60)
    return judge_llm, judge_emb, metrics, run_config, model_name


def score_vllm_batch(rows: list[dict], judge_llm, judge_emb, metrics, run_config) -> list[dict]:
    """Score all 4 RAGAS metrics using vLLM judge."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    samples = []
    for row in rows:
        contexts_raw = row["retrieved_contexts"]
        if isinstance(contexts_raw, str):
            contexts_raw = json.loads(contexts_raw)
        contexts = [c["chunk_text"] for c in (contexts_raw or [])]
        samples.append(SingleTurnSample(
            user_input=row["question"] or "",
            response=row["generated_answer"] or "",
            retrieved_contexts=contexts,
            reference=row["ground_truth"] or "",
        ))

    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=False,
    )
    df = result.to_pandas()

    output = []
    for i, row in enumerate(rows):
        r = df.iloc[i]
        output.append({
            "eval_run_id":       row["eval_run_id"],
            "faithfulness":      _safe_float(r.get("faithfulness")),
            "answer_relevancy":  _safe_float(r.get("answer_relevancy")),
            "context_recall":    _safe_float(r.get("context_recall")),
            "context_precision": _safe_float(r.get("llm_context_precision_with_reference")),
            "status":            "success",
            "error_message":     None,
        })
    return output


def score_vllm_single(row: dict, judge_llm, judge_emb, metrics, run_config) -> dict:
    try:
        return score_vllm_batch([row], judge_llm, judge_emb, metrics, run_config)[0]
    except Exception as exc:
        return {
            "eval_run_id":       row["eval_run_id"],
            "faithfulness":      None,
            "answer_relevancy":  None,
            "context_recall":    None,
            "context_precision": None,
            "status":            "error",
            "error_message":     str(exc)[:500],
        }


def score_faithfulness_batch(rows: list[dict], judge_llm, judge_emb, metrics, run_config) -> list[dict]:
    """Score faithfulness + context_precision for a batch using OpenAI judge."""
    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    samples = []
    for row in rows:
        contexts_raw = row["retrieved_contexts"]
        if isinstance(contexts_raw, str):
            contexts_raw = json.loads(contexts_raw)
        contexts = [c["chunk_text"] for c in (contexts_raw or [])]

        samples.append(SingleTurnSample(
            user_input=row["question"] or "",
            response=row["generated_answer"] or "",
            retrieved_contexts=contexts,
            reference=row["ground_truth"] or "",
        ))

    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=False,
    )
    df = result.to_pandas()

    output = []
    for i, row in enumerate(rows):
        r = df.iloc[i]
        output.append({
            "eval_run_id":       row["eval_run_id"],
            "faithfulness":      _safe_float(r.get("faithfulness")),
            "context_precision": _safe_float(r.get("llm_context_precision_with_reference")),
            "status":            "success",
            "error_message":     None,
        })
    return output


def score_faithfulness_single(row: dict, judge_llm, judge_emb, metrics, run_config) -> dict:
    """Score a single row for faithfulness, returning error payload on failure."""
    try:
        return score_faithfulness_batch([row], judge_llm, judge_emb, metrics, run_config)[0]
    except Exception as exc:
        return {
            "eval_run_id":       row["eval_run_id"],
            "faithfulness":      None,
            "context_precision": None,
            "status":            "error",
            "error_message":     str(exc)[:500],
        }


def _safe_float(val) -> float | None:
    """Return float or None for NaN/None values."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def score_batch(rows: list[dict], judge_llm, judge_emb, metrics, run_config) -> list[dict]:
    """
    Score a batch of eval_run rows with RAGAS.

    Returns a list of dicts with keys:
        eval_run_id, faithfulness, answer_relevancy,
        context_precision, context_recall, status, error_message
    """
    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    samples = []
    for row in rows:
        contexts_raw = row["retrieved_contexts"]
        if isinstance(contexts_raw, str):
            contexts_raw = json.loads(contexts_raw)
        contexts = [c["chunk_text"] for c in (contexts_raw or [])]

        samples.append(SingleTurnSample(
            user_input=row["question"] or "",
            response=row["generated_answer"] or "",
            retrieved_contexts=contexts,
            reference=row["ground_truth"] or "",
        ))

    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=False,
    )
    df = result.to_pandas()

    output = []
    for i, row in enumerate(rows):
        r = df.iloc[i]
        output.append({
            "eval_run_id":        row["eval_run_id"],
            "faithfulness":       None,   # dropped — NaN with local LLMs
            "answer_relevancy":   _safe_float(r.get("answer_relevancy")),
            "context_precision":  None,   # dropped — no reliable metric available
            "context_recall":     _safe_float(r.get("context_recall")),
            "status":             "success",
            "error_message":      None,
        })
    return output


def score_single(row: dict, judge_llm, judge_emb, metrics, run_config) -> dict:
    """Score a single row, returning error payload on failure."""
    try:
        results = score_batch([row], judge_llm, judge_emb, metrics, run_config)
        return results[0]
    except Exception as exc:
        return {
            "eval_run_id":       row["eval_run_id"],
            "faithfulness":      None,
            "answer_relevancy":  None,
            "context_precision": None,
            "context_recall":    None,
            "status":            "error",
            "error_message":     str(exc)[:500],
        }


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Score eval_runs with RAGAS and persist to eval_scores."
    )
    parser.add_argument(
        "--llm",
        metavar="MODEL",
        action="append",
        dest="llm_filter",
        default=None,
        help="Filter eval_runs by LLM model name (repeatable). Default: all.",
    )
    parser.add_argument(
        "--embed",
        metavar="MODEL",
        action="append",
        dest="embed_filter",
        default=None,
        help="Filter by embedding model name (repeatable). Default: all.",
    )
    parser.add_argument(
        "--chunk",
        metavar="NAME",
        action="append",
        dest="chunk_filter",
        default=None,
        help="Filter by chunk config name: small/medium/large (repeatable). Default: all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Rows per RAGAS call (default: {DEFAULT_BATCH}).",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        metavar="N",
        help="Limit to the first N questions in the knowledge base (by ID). Default: all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count pending rows only; do not score or write to DB.",
    )
    parser.add_argument(
        "--version-id",
        metavar="ID",
        type=int,
        default=None,
        help="Filter eval_runs by version_agente.id. Default: all versions.",
    )
    parser.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI (gpt-4o-mini) to score Faithfulness + Context Precision "
             "for rows that already have Answer Relevancy / Context Recall scored.",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use local vLLM server (Qwen3.5-27B) as judge to score ALL 4 metrics "
             "in one pass. Uses VLLM_BASE_URL and EMBED_BASE_URL from .env. "
             "Scores rows with no eval_scores entry yet (same filter as default mode).",
    )
    parser.add_argument(
        "--vllm-url",
        metavar="URL",
        default=None,
        help="Override VLLM_BASE_URL for this run (e.g. http://10.1.50.50:8800).",
    )
    parser.add_argument(
        "--embed-url",
        metavar="URL",
        default=None,
        help="Override EMBED_BASE_URL for this run (e.g. http://10.1.50.50:8801).",
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
    base_url  = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL)
    vllm_url  = args.vllm_url  or os.getenv("VLLM_BASE_URL",  VLLM_BASE_URL)
    embed_url = args.embed_url or os.getenv("EMBED_BASE_URL", EMBED_BASE_URL)

    logging.info("Connecting to database...")
    conn = get_db_connection()

    try:
        if args.vllm:
            # ── vLLM mode: score ALL 4 metrics with local Qwen3.5-27B ────────
            pending = fetch_pending(
                conn,
                llm_filter=args.llm_filter,
                embed_filter=args.embed_filter,
                chunk_filter=args.chunk_filter,
                questions_limit=args.questions,
                version_id=args.version_id,
            )

            logging.info(f"{len(pending):,} rows pending scoring (vLLM mode — all 4 metrics).")

            if args.dry_run:
                logging.info("--dry-run: no scoring or DB writes. Exiting.")
                return

            if not pending:
                logging.info("Nothing to score. Exiting.")
                return

            logging.info(f"Loading vLLM judge: {vllm_url}  embed: {embed_url or 'sentence-transformers'}")
            judge_llm, judge_emb, metrics, run_config, model_name = build_vllm_ragas_objects(vllm_url, embed_url)

            batch_size = args.batch_size
            batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
            errors = 0
            scored = 0

            with tqdm(total=len(pending), desc="RAGAS vLLM scoring", unit="run") as pbar:
                for batch in batches:
                    try:
                        scored_batch = score_vllm_batch(batch, judge_llm, judge_emb, metrics, run_config)
                    except Exception as batch_exc:
                        logging.warning(
                            f"Batch of {len(batch)} failed ({batch_exc}); retrying individually."
                        )
                        scored_batch = [
                            score_vllm_single(row, judge_llm, judge_emb, metrics, run_config)
                            for row in batch
                        ]

                    for payload in scored_batch:
                        try:
                            upsert_score(conn, payload["eval_run_id"], {
                                **payload,
                                "judge_llm_override": model_name,
                            })
                            if payload["status"] == "success":
                                scored += 1
                            else:
                                errors += 1
                                logging.warning(
                                    f"  eval_run_id={payload['eval_run_id']} error: "
                                    f"{payload.get('error_message', '')}"
                                )
                        except Exception as db_exc:
                            errors += 1
                            logging.error(
                                f"  DB write failed for eval_run_id={payload['eval_run_id']}: {db_exc}"
                            )
                            conn.rollback()

                    pbar.update(len(batch))

            logging.info("=" * 60)
            logging.info("vLLM RAGAS scoring complete.")
            logging.info(f"  Scored this run: {scored:,}   Errors: {errors:,}")

        elif args.openai:
            # ── OpenAI mode: score Faithfulness + Context Precision ──────────
            pending = fetch_pending_faithfulness(
                conn,
                llm_filter=args.llm_filter,
                embed_filter=args.embed_filter,
                chunk_filter=args.chunk_filter,
                questions_limit=args.questions,
                version_id=args.version_id,
            )

            logging.info(f"{len(pending):,} rows pending faithfulness scoring.")

            if args.dry_run:
                logging.info("--dry-run: no scoring or DB writes. Exiting.")
                return

            if not pending:
                logging.info("Nothing to score. Exiting.")
                return

            logging.info("Loading OpenAI judge: gpt-4o-mini + text-embedding-3-small")
            judge_llm, judge_emb, metrics, run_config = build_openai_ragas_objects()

            batch_size = args.batch_size
            batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
            errors = 0
            scored = 0

            with tqdm(total=len(pending), desc="Faithfulness scoring", unit="run") as pbar:
                for batch in batches:
                    try:
                        scored_batch = score_faithfulness_batch(batch, judge_llm, judge_emb, metrics, run_config)
                    except Exception as batch_exc:
                        logging.warning(
                            f"Batch of {len(batch)} failed ({batch_exc}); retrying individually."
                        )
                        scored_batch = [
                            score_faithfulness_single(row, judge_llm, judge_emb, metrics, run_config)
                            for row in batch
                        ]

                    for payload in scored_batch:
                        try:
                            upsert_faithfulness(
                                conn,
                                payload["eval_run_id"],
                                payload.get("faithfulness"),
                                payload.get("context_precision"),
                            )
                            scored += 1
                        except Exception as db_exc:
                            errors += 1
                            logging.error(
                                f"  DB write failed for eval_run_id={payload['eval_run_id']}: {db_exc}"
                            )
                            conn.rollback()

                    pbar.update(len(batch))

            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM eval_scores WHERE faithfulness IS NOT NULL"
                )
                n_faith = cur.fetchone()[0]

            logging.info("=" * 60)
            logging.info("Faithfulness scoring complete.")
            logging.info(f"  Rows with faithfulness score: {n_faith:,}")
            logging.info(f"  Scored this run: {scored:,}")
            logging.info(f"  Errors this run: {errors:,}")

        else:
            # ── Default Ollama mode: score Answer Relevancy + Context Recall ─
            pending = fetch_pending(
                conn,
                llm_filter=args.llm_filter,
                embed_filter=args.embed_filter,
                chunk_filter=args.chunk_filter,
                questions_limit=args.questions,
                version_id=args.version_id,
            )

            logging.info(f"{len(pending):,} rows pending scoring.")

            if args.dry_run:
                logging.info("--dry-run: no scoring or DB writes. Exiting.")
                return

            if not pending:
                logging.info("Nothing to score. Exiting.")
                return

            logging.info(f"Loading RAGAS judge: LLM={JUDGE_LLM}, embed={JUDGE_EMBED}")
            judge_llm, judge_emb, metrics, run_config = build_ragas_objects(base_url)

            batch_size = args.batch_size
            batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

            errors = 0
            scored = 0

            with tqdm(total=len(pending), desc="RAGAS scoring", unit="run") as pbar:
                for batch in batches:
                    try:
                        scored_batch = score_batch(batch, judge_llm, judge_emb, metrics, run_config)
                    except Exception as batch_exc:
                        logging.warning(
                            f"Batch of {len(batch)} failed ({batch_exc}); retrying individually."
                        )
                        scored_batch = [
                            score_single(row, judge_llm, judge_emb, metrics, run_config) for row in batch
                        ]

                    for payload in scored_batch:
                        try:
                            upsert_score(conn, payload["eval_run_id"], payload)
                            if payload["status"] == "success":
                                scored += 1
                            else:
                                errors += 1
                                logging.warning(
                                    f"  eval_run_id={payload['eval_run_id']} error: "
                                    f"{payload.get('error_message', '')}"
                                )
                        except Exception as db_exc:
                            errors += 1
                            logging.error(
                                f"  DB write failed for eval_run_id={payload['eval_run_id']}: {db_exc}"
                            )
                            conn.rollback()

                    pbar.update(len(batch))

            # Final summary
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status, COUNT(*) FROM eval_scores GROUP BY status ORDER BY status"
                )
                rows = cur.fetchall()

            logging.info("=" * 60)
            logging.info("Scoring complete. eval_scores summary:")
            for status, count in rows:
                logging.info(f"  {status}: {count:,}")
            logging.info(f"  Scored this run: {scored:,}")
            logging.info(f"  Errors this run: {errors:,}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
