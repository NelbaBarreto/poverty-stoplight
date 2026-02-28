#!/usr/bin/env python3
"""
Pipeline de ingesta multi-embedding multi-chunk para Semáforo de Pobreza.

Etapas:
  1. Procesar documentos (Docling para PDFs, pandas para CSV, texto para MD)
  2. Crear chunks para cada combinación doc × formato × chunk_config
  3. Generar embeddings para cada modelo Ollama disponible
  4. Crear índices IVFFlat en las tablas de embeddings
"""

import os
import re
import sys
import math
import logging
import subprocess
from pathlib import Path

import psycopg2
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from docling.document_converter import DocumentConverter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent.parent / "data"

EMBEDDING_MODELS = [
    {"model_name": "bge-m3",                 "table_name": "embeddings_bge_m3",   "dimensions": 1024},
    {"model_name": "nomic-embed-text",        "table_name": "embeddings_nomic",    "dimensions": 768},
    {"model_name": "mxbai-embed-large",       "table_name": "embeddings_mxbai",   "dimensions": 1024},
    {"model_name": "all-minilm",              "table_name": "embeddings_minilm",  "dimensions": 384},
    {"model_name": "snowflake-arctic-embed",  "table_name": "embeddings_snowflake","dimensions": 1024},
]

CHUNK_CONFIGS = [
    {"name": "small",  "chunk_size": 512,  "overlap": 64},
    {"name": "medium", "chunk_size": 1024, "overlap": 128},
    {"name": "large",  "chunk_size": 2048, "overlap": 256},
]

FORMATS = ["markdown", "plaintext"]

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".md"}

EMBED_BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "db_rag"),
        user=os.getenv("DB_USER", "sgadmin"),
        password=os.getenv("DB_PASSWORD", "sg4dm1n!"),
    )


def get_chunk_config_ids(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT name, id FROM chunk_configs")
        return dict(cur.fetchall())


def get_or_create_document(conn, filename: str, file_type: str, file_path: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM documents WHERE filename = %s", (filename,))
        row = cur.fetchone()
        if row:
            return row[0]
        cur.execute(
            "INSERT INTO documents (filename, file_type, file_path) VALUES (%s, %s, %s) RETURNING id",
            (filename, file_type, str(file_path)),
        )
        doc_id = cur.fetchone()[0]
    conn.commit()
    return doc_id


def chunks_exist(conn, doc_id: int, config_id: int, format_: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM chunks WHERE document_id=%s AND chunk_config_id=%s AND format=%s LIMIT 1",
            (doc_id, config_id, format_),
        )
        return cur.fetchone() is not None


def insert_chunks(conn, doc_id: int, config_id: int, format_: str, texts: list) -> list:
    chunk_ids = []
    with conn.cursor() as cur:
        for idx, text in enumerate(texts):
            cur.execute(
                "INSERT INTO chunks (document_id, chunk_config_id, format, chunk_index, chunk_text)"
                " VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (doc_id, config_id, format_, idx, text),
            )
            chunk_ids.append(cur.fetchone()[0])
    conn.commit()
    return chunk_ids


def get_embedded_chunk_ids(conn, table_name: str) -> set:
    with conn.cursor() as cur:
        cur.execute(f"SELECT chunk_id FROM {table_name}")
        return {row[0] for row in cur.fetchall()}


def create_ivfflat_indexes(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks")
        total_chunks = cur.fetchone()[0]

    # lists ≈ sqrt(rows), min 1
    lists = max(1, math.isqrt(total_chunks))
    logging.info(f"Creating IVFFlat indexes with lists={lists} ({total_chunks} chunks)")

    with conn.cursor() as cur:
        for model in EMBEDDING_MODELS:
            table = model["table_name"]
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            row_count = cur.fetchone()[0]
            if row_count == 0:
                logging.info(f"  Skipping index for {table} (empty)")
                continue
            idx_lists = max(1, math.isqrt(row_count))
            idx_name = f"idx_{table}_embedding"
            cur.execute(f"DROP INDEX IF EXISTS {idx_name}")
            cur.execute(
                f"CREATE INDEX {idx_name} ON {table}"
                f" USING ivfflat (embedding vector_cosine_ops) WITH (lists = {idx_lists})"
            )
            logging.info(f"  Created index on {table} (lists={idx_lists})")
    conn.commit()

# ---------------------------------------------------------------------------
# Document processors
# ---------------------------------------------------------------------------

def process_pdf(file_path: Path) -> dict:
    converter = DocumentConverter()
    result = converter.convert(str(file_path))
    return {
        "markdown":  result.document.export_to_markdown(),
        "plaintext": result.document.export_to_text(),
    }


def process_csv(file_path: Path) -> dict:
    """Process CSV row-by-row: one chunk per row using key: value format."""
    df = pd.read_csv(file_path)
    TEXT_COLS = ["title", "description", "content_text", "dimension",
                 "indicators_names", "country", "lang"]
    available = [c for c in TEXT_COLS if c in df.columns]
    rows = []
    for _, row in df.iterrows():
        parts = []
        for col in available:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                parts.append(f"{col}: {str(val).strip()}")
        if parts:
            rows.append("\n".join(parts))
    return {"rows": rows}


def process_md(file_path: Path) -> dict:
    content = file_path.read_text(encoding="utf-8")
    plaintext = re.sub(r"[#*`\[\]()_~>]", "", content)
    plaintext = re.sub(r"\n{3,}", "\n\n", plaintext).strip()
    return {"markdown": content, "plaintext": plaintext}


PROCESSORS = {
    ".pdf": (process_pdf, "pdf"),
    ".csv": (process_csv, "csv"),
    ".md":  (process_md,  "md"),
}

# ---------------------------------------------------------------------------
# Ollama helper
# ---------------------------------------------------------------------------

def is_model_available(model_name: str) -> bool:
    result = subprocess.run(
        ["ollama", "show", model_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def stage1_process_documents(conn) -> dict:
    logging.info("=" * 60)
    logging.info("Stage 1: Processing documents")
    logging.info("=" * 60)

    files = sorted(
        f for f in DATA_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    logging.info(f"Found {len(files)} supported files in {DATA_DIR}")

    results = {}
    for file_path in tqdm(files, desc="Documents"):
        ext = file_path.suffix.lower()
        processor_fn, file_type = PROCESSORS[ext]
        try:
            # Check DB first — skip heavy processing if already ingested
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM documents WHERE filename = %s", (file_path.name,))
                row = cur.fetchone()
            if row:
                doc_id = row[0]
                logging.info(f"  SKIP (already in DB) {file_path.name} (id={doc_id})")
                # Still need texts for chunking stage if chunks don't exist yet
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM chunks WHERE document_id = %s LIMIT 1", (doc_id,))
                    has_chunks = cur.fetchone() is not None
                if has_chunks:
                    results[file_path.name] = {"doc_id": doc_id, "texts": {}, "file_type": file_type}
                    continue
                # Chunks missing — re-process to get texts
                logging.info(f"  Re-processing for chunking: {file_path.name}")

            texts = processor_fn(file_path)
            doc_id = get_or_create_document(conn, file_path.name, file_type, str(file_path))
            results[file_path.name] = {"doc_id": doc_id, "texts": texts, "file_type": file_type}
            logging.info(f"  OK  {file_path.name} (id={doc_id})")
        except Exception as exc:
            logging.error(f"  ERR {file_path.name}: {exc}")

    return results


def stage2_create_chunks(conn, doc_results: dict, config_ids: dict) -> None:
    logging.info("=" * 60)
    logging.info("Stage 2: Creating chunks")
    logging.info("=" * 60)

    skipped = created = 0

    # CSV files: one chunk per row (row-per-chunk strategy, uses medium/plaintext)
    csv_docs = {fn: d for fn, d in doc_results.items() if d["file_type"] == "csv"}
    other_docs = {fn: d for fn, d in doc_results.items() if d["file_type"] != "csv"}

    for fname, data in tqdm(csv_docs.items(), desc="CSV rows"):
        doc_id    = data["doc_id"]
        rows      = data["texts"].get("rows", [])
        config_id = config_ids["medium"]
        if not rows:
            continue
        if chunks_exist(conn, doc_id, config_id, "plaintext"):
            skipped += 1
            continue
        insert_chunks(conn, doc_id, config_id, "plaintext", rows)
        created += len(rows)
        logging.info(f"  {fname}: {len(rows)} row-chunks")

    # Non-CSV files: standard split × format × config
    combos = [
        (fname, data, fmt, cfg)
        for fname, data in other_docs.items()
        for fmt in FORMATS
        for cfg in CHUNK_CONFIGS
    ]
    for fname, data, fmt, cfg in tqdm(combos, desc="Chunks"):
        doc_id    = data["doc_id"]
        config_id = config_ids[cfg["name"]]
        text      = data["texts"].get(fmt, "")

        if not text.strip():
            continue

        if chunks_exist(conn, doc_id, config_id, fmt):
            skipped += 1
            continue

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["overlap"],
        )
        parts = splitter.split_text(text)
        insert_chunks(conn, doc_id, config_id, fmt, parts)
        created += len(parts)

    logging.info(f"  Created {created} chunks, skipped {skipped} existing combos")


def stage3_embed_chunks(conn) -> None:
    logging.info("=" * 60)
    logging.info("Stage 3: Generating embeddings")
    logging.info("=" * 60)

    with conn.cursor() as cur:
        cur.execute("SELECT id, chunk_text FROM chunks ORDER BY id")
        all_chunks = cur.fetchall()
    logging.info(f"Total chunks: {len(all_chunks)}")

    ollama_base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    for model_info in EMBEDDING_MODELS:
        model_name = model_info["model_name"]
        table_name = model_info["table_name"]

        if not is_model_available(model_name):
            logging.warning(f"  SKIP {model_name}: not available in Ollama")
            continue

        already_done = get_embedded_chunk_ids(conn, table_name)
        pending = [(cid, txt) for cid, txt in all_chunks if cid not in already_done]

        if not pending:
            logging.info(f"  {model_name}: all chunks already embedded, skipping")
            continue

        logging.info(f"  {model_name}: embedding {len(pending)} chunks -> {table_name}")
        embedder = OllamaEmbeddings(model=model_name, base_url=ollama_base)

        with tqdm(total=len(pending), desc=f"  {model_name}", unit="chunk") as pbar:
            for i in range(0, len(pending), EMBED_BATCH_SIZE):
                batch      = pending[i : i + EMBED_BATCH_SIZE]
                chunk_ids  = [c[0] for c in batch]
                texts      = [c[1] for c in batch]

                try:
                    vectors = embedder.embed_documents(texts)
                    with conn.cursor() as cur:
                        for cid, vec in zip(chunk_ids, vectors):
                            cur.execute(
                                f"INSERT INTO {table_name} (chunk_id, embedding)"
                                f" VALUES (%s, %s) ON CONFLICT (chunk_id) DO NOTHING",
                                (cid, vec),
                            )
                    conn.commit()
                    pbar.update(len(batch))
                except Exception:
                    conn.rollback()
                    # Retry each chunk individually; truncate if context length exceeded
                    for cid, text in zip(chunk_ids, texts):
                        for attempt, candidate in enumerate([text, text[:1500], text[:500]]):
                            try:
                                vec = embedder.embed_query(candidate)
                                with conn.cursor() as cur:
                                    cur.execute(
                                        f"INSERT INTO {table_name} (chunk_id, embedding)"
                                        f" VALUES (%s, %s) ON CONFLICT (chunk_id) DO NOTHING",
                                        (cid, vec),
                                    )
                                conn.commit()
                                if attempt > 0:
                                    logging.debug(f"    chunk {cid}: embedded after truncation to {len(candidate)} chars")
                                break
                            except Exception as exc2:
                                conn.rollback()
                                if attempt == 2:
                                    logging.warning(f"    chunk {cid}: skipped after 3 attempts ({exc2})")
                    pbar.update(len(batch))


def stage4_create_indexes(conn) -> None:
    logging.info("=" * 60)
    logging.info("Stage 4: Creating IVFFlat indexes")
    logging.info("=" * 60)
    create_ivfflat_indexes(conn)


def print_summary(conn) -> None:
    logging.info("=" * 60)
    logging.info("Summary")
    logging.info("=" * 60)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM documents")
        logging.info(f"  documents:  {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM chunks")
        logging.info(f"  chunks:     {cur.fetchone()[0]}")
        for m in EMBEDDING_MODELS:
            cur.execute(f"SELECT COUNT(*) FROM {m['table_name']}")
            logging.info(f"  {m['table_name']}: {cur.fetchone()[0]}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    load_dotenv()

    logging.info("Connecting to database...")
    conn = get_db_connection()

    try:
        config_ids = get_chunk_config_ids(conn)

        doc_results = stage1_process_documents(conn)
        stage2_create_chunks(conn, doc_results, config_ids)
        stage3_embed_chunks(conn)
        stage4_create_indexes(conn)
        print_summary(conn)

        logging.info("Ingestion complete!")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
